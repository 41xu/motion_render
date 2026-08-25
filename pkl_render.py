import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import moderngl
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer


DEFAULT_INPUT = Path(__file__).with_name("source.pkl")
# The PromptHMR/mocap pkl stores a full SMPL-X pose (55 joints, axis-angle),
# so this renderer targets SMPL-X only.
_MODEL_CANDIDATES = (
    os.environ.get("SMPLX_MODELS"),
    "/home/ec2-user/nvme-local/sxu/body_models/",
    "/opt/dlami/nvme/sxu/body_models/",
    "/Users/sxu/Projects/body_models/",
)
DEFAULT_MODELS = next(
    (path for path in _MODEL_CANDIDATES if path and Path(path).is_dir()),
    None,
)

# The MotionFix source red and target green, plus a few extras so a
# multi-person pkl can still draw each track distinctly.
NAMED_COLORS = {
    "red": (0.78, 0.28, 0.24, 1.0),
    "green": (0.26, 0.68, 0.36, 1.0),
    "blue": (0.28, 0.45, 0.78, 1.0),
    "yellow": (0.80, 0.62, 0.22, 1.0),
    "purple": (0.55, 0.35, 0.72, 1.0),
}
# Palette order used when a pkl has multiple people (starts with the chosen color).
PERSON_COLORS = tuple(NAMED_COLORS.values())
BODY_AMBIENT = 0.45
BODY_DIFFUSE = 0.65

# Standard SMPL-X axis-angle layout for the flattened 55*3 = 165 pose vector.
POSE_SLICES = {
    "root": slice(0, 3),
    "body": slice(3, 66),
    "jaw": slice(66, 69),
    "leye": slice(69, 72),
    "reye": slice(72, 75),
    "left_hand": slice(75, 120),
    "right_hand": slice(120, 165),
}


def parse_args(description=None):
    parser = argparse.ArgumentParser(
        description=description
        or "Render a PromptHMR-style SMPL-X pkl (smplx_world) with a fixed camera"
    )
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="input .pkl file")
    parser.add_argument(
        "--mode",
        choices=("preview", "frame", "video", "both"),
        default="preview",
        help="both exports the video first and then opens the interactive viewer",
    )
    parser.add_argument("--output", default=None, help="output PNG/MP4 path")
    parser.add_argument(
        "--source",
        default="smplx_world",
        choices=("smplx_world", "smplx_cam"),
        help="which stored pose to render; smplx_world is the grounded world motion",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="render only this person/track id; default renders every detected person",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="directory containing SMPL-X models (or set SMPLX_MODELS)",
    )
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument(
        "--color",
        choices=tuple(NAMED_COLORS),
        default="red",
        help="body color; red is MotionFix source, green is MotionFix target",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--hand-mode",
        choices=("flat", "smplh_mean", "stored"),
        default="flat",
        help="fingers: flat is the zero/straight open hand, smplh_mean uses the "
        "SMPL-H mean hand, stored keeps the pkl hand pose",
    )
    parser.add_argument(
        "--input-coordinates",
        choices=("y_up", "z_up", "camera"),
        default="y_up",
        help="smplx_world is already Y-up and grounded; use camera for smplx_cam",
    )
    parser.add_argument(
        "--keep-start-position",
        action="store_true",
        help="keep the original first-frame horizontal translation instead of centering it",
    )
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    parser.add_argument("--camera-margin", type=float, default=1.2)
    parser.add_argument(
        "--camera-yaw",
        type=float,
        default=0.0,
        help="degrees around world Y; use 180 if the automatically chosen view shows the back",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--width", type=int, default=720, help="GUI logical width; headless output width"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="GUI logical height; headless output height"
    )
    parser.add_argument("--samples", type=int, default=8, help="MSAA sample count")
    parser.add_argument(
        "--export-scale",
        type=float,
        default=None,
        help="GUI default: display pixel ratio; headless default: 1.0",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="render without DISPLAY using an offscreen EGL OpenGL context",
    )
    parser.add_argument(
        "--headless-backend",
        default="egl",
        help="ModernGL standalone context backend used by --headless",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SMPL-X compute device; auto selects CUDA when available",
    )
    parser.add_argument("--ambient-strength", type=float, default=1.2)
    parser.add_argument("--light-strength", type=float, default=0.9)
    parser.add_argument(
        "--video-crf",
        type=int,
        default=18,
        help="H.264 quality; lower is sharper/larger, 18 is visually near-lossless",
    )
    parser.add_argument(
        "--direct-aitviewer",
        action="store_true",
        help="use AITViewer's direct CRF-23 encoder instead of PNG frames + custom CRF",
    )
    args = parser.parse_args()

    if args.models is None:
        parser.error(
            "Could not locate SMPL-X models; pass --models or set SMPLX_MODELS"
        )
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.export_scale is not None and args.export_scale <= 0:
        parser.error("--export-scale must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be between 0 and 51")
    if args.headless and args.mode in ("preview", "both"):
        parser.error("--headless supports --mode frame or --mode video")
    return args


def numpy_data(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def align_to_timeline(values, frames, total):
    """Scatter a per-detection array onto the full timeline, clamp-filling gaps.

    PromptHMR stores each person's pose arrays only for the frames where the
    track was detected. To animate multiple people on one shared timeline we
    place each detection at its frame index and hold the nearest known value in
    any gap so the mesh never collapses to the origin.
    """

    values = np.asarray(values, dtype=np.float32)
    frames = np.asarray(frames).reshape(-1)
    if len(frames) == total and np.array_equal(frames, np.arange(total)):
        return values

    in_range = (frames >= 0) & (frames < total)
    frames = frames[in_range]
    values = values[in_range]
    if len(frames) == 0:
        raise ValueError("no detections fall inside the shared timeline")

    filled = np.zeros((total,) + values.shape[1:], dtype=np.float32)
    known = np.zeros(total, dtype=bool)
    filled[frames] = values
    known[frames] = True

    # Forward-fill, then backfill any leading gap from the first known value so
    # the mesh holds its nearest pose instead of collapsing to the origin.
    last = None
    for i in range(total):
        if known[i]:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    first_known = int(np.argmax(known))
    if first_known > 0:
        filled[:first_known] = filled[first_known]
    return filled


def _load_full_pose(motion, track_id, source_key):
    """Return a (F, 165) axis-angle SMPL-X pose from either 'pose' or 'rotmat'.

    smplx_world stores the flat 55*3 axis-angle pose directly, while smplx_cam
    only keeps the full pose as (F, 55, 3, 3) rotation matrices.
    """

    pose = motion.get("pose")
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float32)
        if pose.ndim == 2 and pose.shape[1] == 165:
            return pose

    rotmat = motion.get("rotmat")
    if rotmat is not None:
        rotmat = np.asarray(rotmat, dtype=np.float32)
        if rotmat.ndim == 4 and rotmat.shape[1:] == (55, 3, 3):
            frames = len(rotmat)
            flat = rotmat.reshape(-1, 3, 3).copy()
            # Undetected/unused joints (e.g. hands when has_hands is False) can be
            # stored as all-zero matrices, which are not valid rotations.
            invalid = np.abs(np.linalg.det(flat)) < 1e-6
            flat[invalid] = np.eye(3, dtype=np.float32)
            axis_angle = Rotation.from_matrix(flat).as_rotvec()
            return axis_angle.reshape(frames, 165).astype(np.float32)

    shapes = {k: np.asarray(v).shape for k, v in motion.items() if hasattr(v, "shape")}
    raise ValueError(
        f"track {track_id} '{source_key}' has no (F, 165) 'pose' or (F, 55, 3, 3) "
        f"'rotmat'; found {shapes}"
    )


def load_people(path, source_key, track):
    payload = joblib.load(path)
    if "people" not in payload or not payload["people"]:
        raise ValueError(f"{path} has no 'people' entry")

    people = payload["people"]
    total = None
    camera = payload.get("camera") or payload.get("camera_world")
    if isinstance(camera, dict) and "pred_cam_R" in camera:
        total = int(len(camera["pred_cam_R"]))

    track_ids = list(people.keys()) if track is None else [track]
    loaded = []
    for track_id in track_ids:
        if track_id not in people:
            raise ValueError(f"track {track_id} not found; available: {list(people.keys())}")
        person = people[track_id]
        if source_key not in person:
            raise ValueError(
                f"track {track_id} has no '{source_key}'; available: {list(person.keys())}"
            )
        motion = person[source_key]
        pose = _load_full_pose(motion, track_id, source_key)
        trans = np.asarray(motion["trans"], dtype=np.float32).reshape(len(pose), 3)
        shape = np.asarray(motion["shape"], dtype=np.float32)
        betas = np.median(shape.reshape(len(shape), -1), axis=0)[:10].astype(np.float32)

        frames = np.asarray(person.get("frames", np.arange(len(pose))))
        span = total if total is not None else int(frames.max()) + 1
        pose = align_to_timeline(pose, frames, span)
        trans = align_to_timeline(trans, frames, span)

        loaded.append(
            {
                "track_id": track_id,
                "pose": pose,
                "trans": trans,
                "betas": betas,
            }
        )
    frame_count = max(len(item["pose"]) for item in loaded)
    return loaded, frame_count


def convert_global_coordinates(root_orient, trans, input_coordinates):
    """Convert global orientation/translation into AITViewer's Y-up world."""

    if input_coordinates == "camera":
        conversion = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    elif input_coordinates == "z_up":
        conversion = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
    else:
        return root_orient.astype(np.float32), trans.astype(np.float32)

    root_matrices = Rotation.from_rotvec(root_orient).as_matrix()
    converted_root = Rotation.from_matrix(conversion[None] @ root_matrices).as_rotvec()
    converted_trans = (conversion @ trans.T).T
    return converted_root.astype(np.float32), converted_trans.astype(np.float32)


def body_facing_direction(joints):
    """Estimate the first-frame body's front from hips and shoulders."""

    joints = numpy_data(joints)
    sample = joints[: min(10, len(joints))]
    left_to_right = (sample[:, 2] - sample[:, 1]) + (sample[:, 17] - sample[:, 16])
    left_to_right = left_to_right.mean(axis=0)
    forward = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), left_to_right)
    forward[1] = 0.0
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (forward / norm).astype(np.float32)


def find_ffmpeg():
    executable = shutil.which("ffmpeg")
    environment_bin = os.path.dirname(sys.executable)
    environment_ffmpeg = os.path.join(environment_bin, "ffmpeg")
    if executable is None and os.path.isfile(environment_ffmpeg):
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
        executable = environment_ffmpeg
    if executable is None:
        raise RuntimeError(
            "Video export requires ffmpeg. Install it with: "
            "conda install -c conda-forge ffmpeg"
        )
    return executable


def create_egl_headless_viewer(size, samples, backend):
    """Create AITViewer's headless window with an explicit EGL backend."""

    original_create_context = moderngl.create_standalone_context

    def create_context_with_backend(*context_args, **context_kwargs):
        context_kwargs.setdefault("backend", backend)
        return original_create_context(*context_args, **context_kwargs)

    moderngl.create_standalone_context = create_context_with_backend
    try:
        return HeadlessRenderer(size=size, samples=samples)
    finally:
        moderngl.create_standalone_context = original_create_context


def describe_opengl(viewer, reject_software=False):
    gl_vendor = str(viewer.ctx.info.get("GL_VENDOR", "unknown"))
    gl_renderer = str(viewer.ctx.info.get("GL_RENDERER", "unknown"))
    gl_version = str(viewer.ctx.info.get("GL_VERSION", "unknown"))
    print(f"OpenGL: vendor={gl_vendor}, renderer={gl_renderer}, version={gl_version}")
    if reject_software and any(
        marker in gl_renderer.lower()
        for marker in ("llvmpipe", "softpipe", "swrast", "software rasterizer")
    ):
        raise RuntimeError(
            "Headless OpenGL is using a software renderer instead of the GPU: "
            f"{gl_renderer}"
        )


def load_smplh_mean_hands(gender):
    """Return the SMPL-H mean (left, right) hand pose as (45,) axis-angle each.

    SMPL-H and SMPL-X share the same 15-joint MANO hand, so this relaxed mean
    pose drops straight into the SMPL-X hand slots. It is a good default when
    the pkl has no real hand tracking (has_hands is False).
    """

    import smplx

    smplh_gender = "female" if gender == "neutral" else gender
    body_model = smplx.create(
        C.smplx_models,
        model_type="smplh",
        gender=smplh_gender,
        use_pca=False,
        flat_hand_mean=False,
    )
    left = numpy_data(body_model.left_hand_mean).reshape(-1)[:45].astype(np.float32)
    right = numpy_data(body_model.right_hand_mean).reshape(-1)[:45].astype(np.float32)
    return left, right


def resolve_hands(pose, hand_mode, smplh_mean):
    frames = len(pose)
    if hand_mode == "stored":
        return pose[:, POSE_SLICES["left_hand"]], pose[:, POSE_SLICES["right_hand"]]
    if hand_mode == "smplh_mean":
        left, right = smplh_mean
        return np.tile(left, (frames, 1)), np.tile(right, (frames, 1))
    zeros_hand = np.zeros((frames, 45), dtype=np.float32)
    return zeros_hand, zeros_hand


def build_sequence(person, args, smpl_layer, color, common_trans_shift, smplh_mean):
    pose = person["pose"]
    root, trans = convert_global_coordinates(
        pose[:, POSE_SLICES["root"]], person["trans"], args.input_coordinates
    )
    trans = trans - common_trans_shift

    left_hand, right_hand = resolve_hands(pose, args.hand_mode, smplh_mean)
    sequence = SMPLSequence(
        poses_body=pose[:, POSE_SLICES["body"]],
        poses_root=root,
        poses_left_hand=left_hand,
        poses_right_hand=right_hand,
        poses_jaw=np.zeros((len(pose), 3), dtype=np.float32),
        betas=person["betas"],
        trans=trans,
        smpl_layer=smpl_layer,
        z_up=False,
        name=f"track {person['track_id']}",
        color=color,
    )
    sequence.mesh_seq.material.ambient = BODY_AMBIENT
    sequence.mesh_seq.material.diffuse = BODY_DIFFUSE
    return sequence


def main(load_people_fn=None, description=None):
    args = parse_args(description=description)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    C.smplx_models = str(Path(args.models).expanduser())
    C.auto_set_floor = False
    if args.device == "auto":
        compute_device = "cuda:0" if torch.cuda.is_available() else str(C.device)
    else:
        compute_device = args.device

    loader = load_people if load_people_fn is None else load_people_fn
    people, frame_count = loader(input_path, args.source, args.track)

    # Center the horizontal start using the first rendered person's first frame.
    common_trans_shift = np.zeros(3, dtype=np.float32)
    if not args.keep_start_position:
        first_trans = people[0]["trans"][0]
        common_trans_shift = np.array([first_trans[0], 0.0, first_trans[2]], dtype=np.float32)

    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender=args.gender,
        num_betas=10,
        device=compute_device,
    )

    smplh_mean = (
        load_smplh_mean_hands(args.gender)
        if args.hand_mode == "smplh_mean"
        else (None, None)
    )

    # Start the palette at the chosen color; extra people cycle through the rest.
    chosen = NAMED_COLORS[args.color]
    palette = (chosen,) + tuple(c for c in PERSON_COLORS if c != chosen)

    sequences = []
    for index, person in enumerate(people):
        color = palette[index % len(palette)]
        sequences.append(
            build_sequence(person, args, smpl_layer, color, common_trans_shift, smplh_mean)
        )

    # Ground every rendered body on one shared floor at Y=0 using a single
    # offset over all frames and people (avoids per-frame vertical jitter).
    all_min_y = min(float(numpy_data(seq.vertices)[..., 1].min()) for seq in sequences)
    ground_offset = -all_min_y
    ground_position = np.array([0.0, ground_offset, 0.0], dtype=np.float32)

    bounds_min = None
    bounds_max = None
    for seq in sequences:
        seq.position = ground_position
        vertices = numpy_data(seq.vertices)
        seq_min = vertices.min(axis=(0, 1)) + ground_position
        seq_max = vertices.max(axis=(0, 1)) + ground_position
        bounds_min = seq_min if bounds_min is None else np.minimum(bounds_min, seq_min)
        bounds_max = seq_max if bounds_max is None else np.maximum(bounds_max, seq_max)
    bounds_center = (bounds_min + bounds_max) * 0.5
    half_extents = (bounds_max - bounds_min) * 0.5

    heading = body_facing_direction(sequences[0].joints)
    if args.camera_yaw:
        heading = Rotation.from_euler("y", args.camera_yaw, degrees=True).apply(heading)
    heading[1] = 0.0
    heading /= np.linalg.norm(heading)

    logical_size = (args.width, args.height)
    if args.headless:
        export_scale = 1.0 if args.export_scale is None else args.export_scale
        export_size = (
            int(round(args.width * export_scale)),
            int(round(args.height * export_scale)),
        )
        viewer = create_egl_headless_viewer(
            export_size, args.samples, args.headless_backend
        )
    else:
        viewer = Viewer(size=logical_size, samples=args.samples)
    describe_opengl(viewer, reject_software=args.headless)
    viewer.playback_fps = args.fps
    for seq in sequences:
        viewer.scene.add(seq)
    viewer.scene.origin.enabled = False
    viewer.scene.light_mode = "default"
    viewer.scene.ambient_strength = args.ambient_strength
    for light in viewer.scene.lights:
        light.strength = args.light_strength
    viewer.scene.background_color = (0.93, 0.93, 0.93, 1.0)
    viewer.scene.floor.c1 = (0.68, 0.68, 0.68, 1.0)
    viewer.scene.floor.c2 = (0.52, 0.52, 0.52, 1.0)

    camera_right = np.array([heading[2], 0.0, -heading[0]], dtype=np.float32)
    half_width = float(np.dot(np.abs(camera_right), half_extents))
    half_depth = float(np.dot(np.abs(heading), half_extents))
    tan_half_vertical = np.tan(np.deg2rad(args.camera_fov) * 0.5)
    aspect = args.width / args.height
    tan_half_horizontal = tan_half_vertical * aspect
    auto_distance = half_depth + args.camera_margin * max(
        half_extents[1] / tan_half_vertical,
        half_width / tan_half_horizontal,
    )
    camera_distance = auto_distance if args.camera_distance is None else args.camera_distance
    camera_target = bounds_center.copy()
    camera_position = camera_target + heading * camera_distance
    camera_position[1] = args.camera_height
    camera = PinholeCamera(
        camera_position,
        camera_target,
        viewer.window_size[0],
        viewer.window_size[1],
        fov=args.camera_fov,
        viewer=viewer,
    )
    viewer.scene.add(camera)
    viewer.set_temp_camera(camera)

    if not 0 <= args.frame < frame_count:
        raise ValueError(f"--frame {args.frame} is outside [0, {frame_count - 1}]")
    viewer.scene.current_frame_id = args.frame

    if not args.headless:
        logical_size = tuple(viewer.window_size)
    framebuffer_size = tuple(viewer.wnd.buffer_size)
    pixel_ratio = float(viewer.wnd.pixel_ratio)
    if args.headless:
        export_size = tuple(viewer.window_size)
    else:
        export_scale = pixel_ratio if args.export_scale is None else args.export_scale
        max_scale = min(
            framebuffer_size[0] / logical_size[0],
            framebuffer_size[1] / logical_size[1],
        )
        if export_scale > max_scale + 1e-6:
            raise ValueError(
                f"--export-scale {export_scale:g} exceeds framebuffer scale {max_scale:g}"
            )
        export_size = (
            int(round(logical_size[0] * export_scale)),
            int(round(logical_size[1] * export_scale)),
        )

    print(
        f"Loaded {len(people)} person(s), {frame_count} frames from {input_path.name}\n"
        f"Pose source: {args.source}\n"
        f"SMPL-X compute device: {compute_device}\n"
        f"Ground offset: {ground_offset:.3f} m\n"
        f"Camera: distance={camera_distance:.3f} m, height={args.camera_height:.3f} m, "
        f"yaw={args.camera_yaw:.1f} deg\n"
        f"Render buffer: window={logical_size}, framebuffer={framebuffer_size}, "
        f"pixel_ratio={pixel_ratio:g}"
    )

    logical_frame_reader = viewer.get_current_frame_as_image

    def physical_frame_reader(alpha=False):
        fmt = "RGBA" if alpha else "RGB"
        components = 4 if alpha else 3
        width, height = viewer.window_size
        viewport = (0, 0, width, height)
        image = Image.frombytes(
            fmt,
            (width, height),
            viewer.wnd.fbo.read(viewport=viewport, alignment=1, components=components),
        )
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def set_export_viewport(enabled):
        if args.headless:
            return export_size
        if enabled:
            size = export_size
        else:
            size = logical_size
        viewer.window_size = size
        viewer._resize_viewports()
        viewer.wnd.fbo.viewport = (0, 0, size[0], size[1])
        viewer.get_current_frame_as_image = (
            physical_frame_reader if enabled else logical_frame_reader
        )
        return size

    def export_video(output_path):
        ffmpeg = find_ffmpeg()
        if args.direct_aitviewer:
            viewer.export_video(
                output_path=output_path,
                output_fps=args.fps,
                quality="high",
                ensure_no_overwrite=False,
            )
            return

        with tempfile.TemporaryDirectory(prefix="pkl_render_frames_") as frame_root:
            viewer.export_video(output_path=None, frame_dir=frame_root, output_fps=args.fps)
            frame_dir = os.path.join(frame_root, "0000")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(args.fps),
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
        print(f"Video saved to {Path(output_path).resolve()} (CRF {args.video_crf})")

    if args.mode == "preview":
        print("Opening interactive AITViewer preview")
        viewer.run()
        return

    viewer._init_scene()
    export_size = set_export_viewport(True)
    print(f"Export resolution: {export_size[0]}x{export_size[1]}")

    if args.mode == "frame":
        output = args.output or f"{input_path.stem}.png"
        viewer.export_frame(output)
        print(f"Frame saved to {Path(output).resolve()}")
        viewer.close()
        return

    output = args.output or f"{input_path.stem}.mp4"
    export_video(output)
    if args.mode == "both":
        set_export_viewport(False)
        viewer.scene.current_frame_id = args.frame
        print("Video saved; opening interactive AITViewer preview")
        viewer.run()
    else:
        viewer.close()


if __name__ == "__main__":
    main()
