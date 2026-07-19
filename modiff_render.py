import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
from tools.transformers3d import get_z_rot

import joblib

motionfix_pth = "/Users/sxu/Downloads/motionfix-dataset/motionfix.pth.tar"

data_dict = joblib.load(motionfix_pth)

source = data_dict["005454"]["motion_source"]
# include rots, trans, joint_positions, timestamp

target = data_dict["005454"]["motion_target"]

C.smplx_models = "/Users/sxu/Projects/body_models/"
# Motions are grounded explicitly below; keep one shared floor fixed at Y=0.
C.auto_set_floor = False


FPS=30
INPUT_IS_Z_UP = True
JOINT_POSITIONS_ARE_ROOT_RELATIVE = False
# A softer MotionFix-like palette.  Keep the material bright instead of raising
# scene ambient light too far: excessive ambient light removes facial shading.
SOURCE_COLOR = (0.78, 0.28, 0.24, 1.0)
TARGET_COLOR = (0.26, 0.68, 0.36, 1.0)
BODY_AMBIENT = 0.45
BODY_DIFFUSE = 0.65
# if true: is the relative position, we need to add the trans aka 整体位移 这里只保存了相对根节点的局部位置
# if false: 已经是包含了trans的joint position了，是世界坐标 关节位置包含了人物的整体移动，不要再加trans了
# 加trans的时候是把每一帧的trans驾上来 如果已经是世界坐标再加就会重复计算人的位移trans joints_positions[:, 0] 如果和trans的移动轨迹基本相似，则为False
# joint_positions[:, 0]: root trajectory

# ---------------------------------------------------------------------
# Rotation conversion
# ---------------------------------------------------------------------

def to_axis_angle(rots, n_joints):
    """Return rotations with shape (F, J, 3) in axis-angle format."""

    rots = np.asarray(rots)

    # Flattened axis-angle: (F, J*3)
    if rots.ndim == 2:
        if rots.shape[1] != n_joints * 3:
            raise ValueError(
                f"Expected flattened rotations of shape (F, {n_joints * 3}), "
                f"but got {rots.shape}"
            )
        return rots.reshape(len(rots), n_joints, 3)

    # Axis-angle: (F, J, 3)
    if rots.ndim == 3 and rots.shape[-1] == 3:
        return rots

    # Quaternion in XYZW order: (F, J, 4)
    if rots.ndim == 3 and rots.shape[-1] == 4:
        return Rotation.from_quat(
            rots.reshape(-1, 4)
        ).as_rotvec().reshape(len(rots), n_joints, 3)

    # Rotation matrices: (F, J, 3, 3)
    if rots.ndim == 4 and rots.shape[-2:] == (3, 3):
        return Rotation.from_matrix(
            rots.reshape(-1, 3, 3)
        ).as_rotvec().reshape(len(rots), n_joints, 3)

    raise ValueError(f"Unsupported rotation shape: {rots.shape}")


# ---------------------------------------------------------------------
# Load motion
# ---------------------------------------------------------------------

rots = np.asarray(source["rots"])
trans = np.asarray(source["trans"], dtype=np.float32)
joint_positions = np.asarray(source["joint_positions"], dtype=np.float32)

# print("joint_position: ", joint_positions[:, 0])
# print("trans: ", trans)

if trans.ndim != 2 or trans.shape[1] != 3:
    raise ValueError(f"Expected trans shape (F, 3), got {trans.shape}")

if joint_positions.ndim != 3 or joint_positions.shape[2] != 3:
    raise ValueError(
        f"Expected joint_positions shape (F, J, 3), "
        f"got {joint_positions.shape}"
    )

n_frames, n_joints, _ = joint_positions.shape

if len(trans) != n_frames:
    raise ValueError("rots, trans and joint_positions must have the same frame count")

# Standard body-only layouts.
if n_joints == 22:
    model_type = "smplh"
elif n_joints == 24:
    model_type = "smpl"
else:
    raise ValueError(
        f"Cannot infer body model from {n_joints} joints. "
        "Expected 22 for SMPL-X body or 24 for SMPL."
    )

axis_angle = to_axis_angle(rots, n_joints).astype(np.float32) # this is already aa format rots

if axis_angle.shape[0] != n_frames:
    raise ValueError("rots and joint_positions have different frame counts")

# ---------------------------------------------------------------------
# Create SMPL sequence
# ---------------------------------------------------------------------

smpl_layer = SMPLLayer(
    model_type=model_type,
    gender="neutral",
    device=C.device,
)

expected_joints = smpl_layer.bm.NUM_BODY_JOINTS + 1
if n_joints != expected_joints:
    raise ValueError(
        f"{model_type} expects {expected_joints} root/body joints, "
        f"but the motion contains {n_joints}"
    )

# Joint zero is the root. Remaining joints are body rotations.
poses_root = axis_angle[:, 0]
poses_body = axis_angle[:, 1:].reshape(n_frames, -1)

smpl_seq = SMPLSequence(
    poses_body=poses_body,
    poses_root=poses_root,
    trans=trans,
    smpl_layer=smpl_layer,
    z_up=INPUT_IS_Z_UP,
    name="Source motion",
    color=SOURCE_COLOR,
)

first_frame = SMPLSequence(
    poses_body=poses_body[0:1],
    poses_root=poses_root[0:1],
    trans=trans[0:1],
    smpl_layer=smpl_layer,
    z_up=INPUT_IS_Z_UP,
    name="Source first frame",
    color=SOURCE_COLOR,
)

# The dataset stores translation at the SMPL root, not at the feet. Source and
# target can therefore have different vertical origins when their poses differ.
# Use one constant offset per complete sequence so both share the same ground
# without introducing per-frame vertical jitter.
source_ground_offset = -float(smpl_seq.vertices[..., 2].min().item())
source_position = np.array([0.0, source_ground_offset, 0.0], dtype=np.float32)
smpl_seq.position = source_position
first_frame.position = source_position

# Build the edited/target motion as a second overlaid sequence, matching the
# red-source / green-target convention used by MotionFix.
target_rots = np.asarray(target["rots"])
target_trans = np.asarray(target["trans"], dtype=np.float32)
target_joint_positions = np.asarray(target["joint_positions"], dtype=np.float32)

if target_trans.ndim != 2 or target_trans.shape[1] != 3:
    raise ValueError(f"Expected target trans shape (F, 3), got {target_trans.shape}")
if target_joint_positions.ndim != 3 or target_joint_positions.shape[2] != 3:
    raise ValueError(
        "Expected target joint_positions shape (F, J, 3), "
        f"got {target_joint_positions.shape}"
    )

target_n_frames, target_n_joints, _ = target_joint_positions.shape
if target_n_joints != n_joints:
    raise ValueError(
        f"Source and target joint counts differ: {n_joints} vs {target_n_joints}"
    )
if len(target_trans) != target_n_frames:
    raise ValueError("Target rotations, translations and joints must have the same frame count")

target_axis_angle = to_axis_angle(target_rots, target_n_joints).astype(np.float32)
target_poses_root = target_axis_angle[:, 0]
target_poses_body = target_axis_angle[:, 1:].reshape(target_n_frames, -1)

target_seq = SMPLSequence(
    poses_body=target_poses_body,
    poses_root=target_poses_root,
    trans=target_trans,
    smpl_layer=smpl_layer,
    z_up=INPUT_IS_Z_UP,
    name="Target motion",
    color=TARGET_COLOR,
)

for sequence in (smpl_seq, first_frame, target_seq):
    sequence.mesh_seq.material.ambient = BODY_AMBIENT
    sequence.mesh_seq.material.diffuse = BODY_DIFFUSE

target_ground_offset = -float(target_seq.vertices[..., 2].min().item())
target_seq.position = np.array(
    [0.0, target_ground_offset, 0.0],
    dtype=np.float32,
)


def sequence_bounds_in_viewer(sequence, ground_offset):
    """Bounds over all frames after converting input Z-up to viewer Y-up."""

    vertices = sequence.vertices
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    else:
        vertices = np.asarray(vertices)
    bounds_min = np.array(
        [
            vertices[..., 0].min(),
            vertices[..., 2].min() + ground_offset,
            -vertices[..., 1].max(),
        ],
        dtype=np.float32,
    )
    bounds_max = np.array(
        [
            vertices[..., 0].max(),
            vertices[..., 2].max() + ground_offset,
            -vertices[..., 1].min(),
        ],
        dtype=np.float32,
    )
    return bounds_min, bounds_max


source_bounds = sequence_bounds_in_viewer(smpl_seq, source_ground_offset)
target_bounds = sequence_bounds_in_viewer(target_seq, target_ground_offset)
motion_bounds_min = np.minimum(source_bounds[0], target_bounds[0])
motion_bounds_max = np.maximum(source_bounds[1], target_bounds[1])
motion_bounds_center = (motion_bounds_min + motion_bounds_max) * 0.5
motion_half_extents = (motion_bounds_max - motion_bounds_min) * 0.5

# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------

# Only the first/root joint describes the body's global orientation. Passing
# all joints to get_z_rot would produce (J, 3, 3), and therefore a non-scalar
# relative camera position.
root_z_rotation = get_z_rot(torch.from_numpy(poses_root[0]), in_format="aa")
heading = -root_z_rotation[:, 1]

parser = argparse.ArgumentParser(description="Preview or export a MotionFix source/target pair")
parser.add_argument(
    "--mode",
    choices=("preview", "frame", "video", "both"),
    default="preview",
    help="preview opens the interactive AITViewer window (default)",
)
parser.add_argument(
    "--output",
    default=None,
    help="output path for frame/video mode (defaults: test.png or test.mp4)",
)
parser.add_argument(
    "--camera-distance",
    type=float,
    default=None,
    help="fixed horizontal distance; default fits all source/target frames automatically",
)
parser.add_argument("--camera-height", type=float, default=1.5)
parser.add_argument("--camera-fov", type=float, default=45.0)
parser.add_argument("--camera-margin", type=float, default=1.2)
parser.add_argument("--frame", type=int, default=0, help="initial/ exported frame index")
parser.add_argument("--width", type=int, default=720, help="native render width")
parser.add_argument("--height", type=int, default=720, help="native render height")
parser.add_argument("--samples", type=int, default=8, help="MSAA sample count")
parser.add_argument(
    "--export-scale",
    type=float,
    default=None,
    help="export viewport scale; default uses the display pixel ratio (2x on Retina)",
)
parser.add_argument("--ambient-strength", type=float, default=1.2)
parser.add_argument("--light-strength", type=float, default=0.9)
parser.add_argument(
    "--video-quality",
    choices=("high", "medium", "low"),
    default="high",
    help="AITViewer encoder quality; high uses CRF 23",
)
parser.add_argument(
    "--video-crf",
    type=int,
    default=None,
    help="custom H.264 CRF via lossless PNG frames; try 18 for a sharper video",
)
parser.add_argument(
    "--video-variants",
    choices=("overlap", "source", "target", "all"),
    default="overlap",
    help="which visibility variant to export; all writes overlap/source/target videos",
)
args = parser.parse_args()

print(
    f"Ground offsets (viewer Y): source={source_ground_offset:.3f} m, "
    f"target={target_ground_offset:.3f} m"
)

if args.width <= 0 or args.height <= 0:
    raise ValueError("--width and --height must be positive")
if args.export_scale is not None and args.export_scale <= 0:
    raise ValueError("--export-scale must be positive")

renderer = Viewer(
    size=(args.width, args.height),
    samples=args.samples,
)
print(
    "Render buffer: "
    f"window={renderer.window_size}, framebuffer={renderer.wnd.buffer_size}, "
    f"pixel_ratio={renderer.wnd.pixel_ratio}"
)
logical_window_size = tuple(renderer.window_size)
framebuffer_size = tuple(renderer.wnd.buffer_size)
max_export_scale = min(
    framebuffer_size[0] / logical_window_size[0],
    framebuffer_size[1] / logical_window_size[1],
)
export_scale = float(renderer.wnd.pixel_ratio) if args.export_scale is None else args.export_scale
if export_scale > max_export_scale + 1e-6:
    raise ValueError(
        f"--export-scale {export_scale:g} exceeds the available framebuffer scale "
        f"{max_export_scale:g}; reduce the scale or increase the viewer size"
    )


logical_frame_reader = renderer.get_current_frame_as_image


def get_export_frame_as_image(alpha=False):
    """Read physical framebuffer pixels without AITViewer's Retina downscale."""

    fmt = "RGBA" if alpha else "RGB"
    components = 4 if alpha else 3
    width, height = renderer.window_size
    viewport = (0, 0, width, height)
    image = Image.frombytes(
        fmt,
        (width, height),
        renderer.wnd.fbo.read(
            viewport=viewport,
            alignment=1,
            components=components,
        ),
    )
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def set_export_viewport(enabled):
    """Use the full Retina framebuffer for export, then restore the GUI viewport."""

    if enabled:
        size = (
            int(round(logical_window_size[0] * export_scale)),
            int(round(logical_window_size[1] * export_scale)),
        )
    else:
        size = logical_window_size
    renderer.window_size = size
    renderer._resize_viewports()
    renderer.wnd.fbo.viewport = (0, 0, size[0], size[1])
    renderer.get_current_frame_as_image = (
        get_export_frame_as_image if enabled else logical_frame_reader
    )
    return size


renderer.playback_fps = FPS
renderer.scene.add(smpl_seq, target_seq)
renderer.scene.origin.enabled = False
renderer.scene.light_mode = "default"
renderer.scene.ambient_strength = args.ambient_strength
for light in renderer.scene.lights:
    light.strength = args.light_strength
renderer.scene.background_color = (0.93, 0.93, 0.93, 1.0)
renderer.scene.floor.c1 = (0.68, 0.68, 0.68, 1.0)
renderer.scene.floor.c2 = (0.52, 0.52, 0.52, 1.0)

# Keep the camera fixed in world space. Unlike lock_to_node(relative_position),
# camera-height is an absolute height above the shared Y=0 ground. This gives a
# shallow downward pitch and keeps the horizon visible like the MotionFix render.
camera_target = motion_bounds_center.copy()
camera_direction = np.array(
    [heading[0].item(), 0.0, -heading[1].item()],
    dtype=np.float32,
)
camera_direction /= np.linalg.norm(camera_direction)

# Fit the complete source+target AABB in the perspective frustum. The nearest
# corner is used for a conservative fit and camera-margin leaves room around it.
camera_right = np.array(
    [camera_direction[2], 0.0, -camera_direction[0]],
    dtype=np.float32,
)
half_width = float(np.dot(np.abs(camera_right), motion_half_extents))
half_depth = float(np.dot(np.abs(camera_direction), motion_half_extents))
tan_half_vertical = np.tan(np.deg2rad(args.camera_fov) * 0.5)
aspect = renderer.window_size[0] / renderer.window_size[1]
tan_half_horizontal = tan_half_vertical * aspect
auto_camera_distance = half_depth + args.camera_margin * max(
    motion_half_extents[1] / tan_half_vertical,
    half_width / tan_half_horizontal,
)
camera_distance = (
    auto_camera_distance
    if args.camera_distance is None
    else args.camera_distance
)

camera_position = camera_target + camera_direction * camera_distance
camera_position[1] = args.camera_height

print(
    f"Camera: distance={camera_distance:.3f} m "
    f"({'auto' if args.camera_distance is None else 'manual'}), "
    f"height={args.camera_height:.3f} m, fov={args.camera_fov:.1f} deg"
)

camera = PinholeCamera(
    camera_position,
    camera_target,
    renderer.window_size[0],
    renderer.window_size[1],
    fov=args.camera_fov,
    viewer=renderer,
)
renderer.scene.add(camera)
renderer.set_temp_camera(camera)
renderer.scene.current_frame_id = args.frame


def export_video(output_path, ffmpeg_executable):
    """Export directly with AITViewer or through PNG frames for custom CRF."""

    if args.video_crf is None:
        renderer.export_video(
            output_path=output_path,
            output_fps=FPS,
            quality=args.video_quality,
            ensure_no_overwrite=False,
        )
        return

    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be between 0 and 51")

    with tempfile.TemporaryDirectory(prefix="motionfix_frames_") as frame_root:
        renderer.export_video(
            output_path=None,
            frame_dir=frame_root,
            output_fps=FPS,
        )
        frame_dir = os.path.join(frame_root, "0000")
        subprocess.run(
            [
                ffmpeg_executable,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                os.path.join(frame_dir, "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(args.video_crf),
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
        )
    print(f"Video saved to {os.path.abspath(output_path)} (CRF {args.video_crf})")


def set_video_variant(variant):
    """Change visibility only; camera, floor, colors and grounding stay fixed."""

    smpl_seq.enabled = variant in ("overlap", "source")
    target_seq.enabled = variant in ("overlap", "target")


def variant_output_paths(output_path):
    if args.video_variants != "all":
        return [(args.video_variants, output_path)]

    path = Path(output_path)
    suffix = path.suffix or ".mp4"
    stem = path.stem if path.suffix else path.name
    return [
        (variant, str(path.with_name(f"{stem}_{variant}{suffix}")))
        for variant in ("overlap", "source", "target")
    ]

if args.mode == "preview":
    print("Opening AITViewer preview: source=red, target=green")
    renderer.run()
elif args.mode == "frame":
    renderer._init_scene()
    export_size = set_export_viewport(True)
    print(f"Export resolution: {export_size[0]}x{export_size[1]}")
    renderer.export_frame(file_path=args.output or "test.png")
else:
    ffmpeg_executable = shutil.which("ffmpeg")
    environment_bin = os.path.dirname(sys.executable)
    environment_ffmpeg = os.path.join(environment_bin, "ffmpeg")
    if ffmpeg_executable is None and os.path.isfile(environment_ffmpeg):
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
        ffmpeg_executable = environment_ffmpeg
    if ffmpeg_executable is None:
        raise RuntimeError(
            "Video export requires ffmpeg on PATH. Install it with: "
            "conda install -n vml -c conda-forge ffmpeg"
        )
    renderer._init_scene()
    export_size = set_export_viewport(True)
    print(f"Export resolution: {export_size[0]}x{export_size[1]}")
    outputs = variant_output_paths(args.output or "test.mp4")
    for variant, output_path in outputs:
        set_video_variant(variant)
        print(f"Exporting {variant} video -> {output_path}")
        export_video(output_path, ffmpeg_executable)
    set_video_variant("overlap")
    if args.mode == "both":
        set_export_viewport(False)
        renderer.scene.current_frame_id = args.frame
        print("Video saved; opening AITViewer preview")
        renderer.run()
