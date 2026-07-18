import argparse
import json
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
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer


DEFAULT_INPUT = Path(__file__).with_name("Ways_to_Catch_A_Cold_clip1.json")
DEFAULT_MODELS = "/Users/sxu/Projects/body_models/"
MOTION_COLOR = (0.26, 0.68, 0.36, 1.0)
BODY_AMBIENT = 0.45
BODY_DIFFUSE = 0.65


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one SMPL-X JSON motion with a fixed MotionFix-style camera"
    )
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="input motion JSON")
    parser.add_argument(
        "--mode",
        choices=("preview", "frame", "video", "both"),
        default="preview",
        help="both exports the video first and then opens the interactive viewer",
    )
    parser.add_argument("--output", default=None, help="output PNG/MP4 path")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="directory containing SMPL-X models")
    parser.add_argument("--gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--use-face",
        action="store_true",
        help="apply jaw/expression parameters; default keeps a neutral MotionFix-style face",
    )
    parser.add_argument(
        "--input-coordinates",
        choices=("camera", "y_up", "z_up"),
        default="camera",
        help="sample JSON uses camera coordinates: X right, Y down, Z forward",
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
    parser.add_argument("--width", type=int, default=720, help="logical viewer width")
    parser.add_argument("--height", type=int, default=720, help="logical viewer height")
    parser.add_argument("--samples", type=int, default=8, help="MSAA sample count")
    parser.add_argument(
        "--export-scale",
        type=float,
        default=None,
        help="default uses display pixel ratio (2x on Retina, yielding 1440x1440)",
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

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.export_scale is not None and args.export_scale <= 0:
        parser.error("--export-scale must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be between 0 and 51")
    return args


def stacked(records, key, width, default=None):
    values = []
    for frame_id, record in enumerate(records):
        value = record.get(key, default)
        if value is None:
            raise ValueError(f"Frame {frame_id} is missing smplx_params.{key}")
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(value) != width:
            raise ValueError(
                f"Frame {frame_id} smplx_params.{key} has {len(value)} values; expected {width}"
            )
        values.append(value)
    return np.stack(values)


def load_smplx_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Expected a non-empty top-level 'annotations' array")
    annotations = sorted(
        annotations,
        key=lambda item: (item.get("image_id", 0), item.get("file_name", "")),
    )
    records = [annotation.get("smplx_params", {}) for annotation in annotations]

    root = stacked(records, "root_orient", 3)
    body = stacked(records, "pose_body", 63)
    hands = stacked(records, "pose_hand", 90, default=np.zeros(90, dtype=np.float32))
    trans = stacked(records, "trans", 3)
    jaw = stacked(records, "pose_jaw", 3, default=np.zeros(3, dtype=np.float32))

    # SMPL-X model files conventionally expose 10 expression coefficients.  The
    # sample stores 50 entries, with the extra entries padded or model-specific.
    expressions = []
    for record in records:
        expression = np.asarray(record.get("face_expr", np.zeros(10)), dtype=np.float32).reshape(-1)
        if len(expression) < 10:
            expression = np.pad(expression, (0, 10 - len(expression)))
        expressions.append(expression[:10])
    expressions = np.stack(expressions)

    # A constant shape avoids subtle frame-to-frame surface jitter.
    beta_frames = stacked(records, "betas", 10, default=np.zeros(10, dtype=np.float32))
    betas = np.median(beta_frames, axis=0).astype(np.float32)

    return {
        "root": root,
        "body": body,
        "left_hand": hands[:, :45],
        "right_hand": hands[:, 45:],
        "jaw": jaw,
        "expression": expressions,
        "trans": trans,
        "betas": betas,
        "frame_count": len(records),
    }


def convert_global_coordinates(root_orient, trans, input_coordinates):
    """Convert only global orientation/translation into AITViewer's Y-up world."""

    if input_coordinates == "camera":
        # Computer-vision camera coordinates (X right, Y down, Z forward) to
        # a Y-up world. This 180-degree X rotation also cancels the common
        # root_orient ~= [pi, 0, 0] found in camera-space SMPL-X predictions.
        conversion = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    elif input_coordinates == "z_up":
        conversion = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
    else:
        conversion = np.eye(3, dtype=np.float32)

    root_matrices = Rotation.from_rotvec(root_orient).as_matrix()
    converted_root = Rotation.from_matrix(conversion[None] @ root_matrices).as_rotvec()
    converted_trans = (conversion @ trans.T).T
    return converted_root.astype(np.float32), converted_trans.astype(np.float32)


def numpy_data(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def body_facing_direction(joints):
    """Estimate the first-frame body's front from hips and shoulders."""

    joints = numpy_data(joints)
    sample = joints[: min(10, len(joints))]
    # SMPL-X: 1/2 are left/right hip and 16/17 are left/right shoulder.
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
            "conda install -n vml -c conda-forge ffmpeg"
        )
    return executable


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    C.smplx_models = str(Path(args.models).expanduser())
    C.auto_set_floor = False

    motion = load_smplx_json(input_path)
    root, trans = convert_global_coordinates(
        motion["root"], motion["trans"], args.input_coordinates
    )
    if not args.keep_start_position:
        trans[:, 0] -= trans[0, 0]
        trans[:, 2] -= trans[0, 2]

    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender=args.gender,
        num_betas=10,
        num_expression_coeffs=10,
        device=C.device,
    )
    sequence = SMPLSequence(
        poses_body=motion["body"],
        poses_root=root,
        poses_left_hand=motion["left_hand"],
        poses_right_hand=motion["right_hand"],
        poses_jaw=motion["jaw"] if args.use_face else np.zeros_like(motion["jaw"]),
        expression=(
            motion["expression"]
            if args.use_face
            else np.zeros_like(motion["expression"])
        ),
        betas=motion["betas"],
        trans=trans,
        smpl_layer=smpl_layer,
        z_up=False,
        name=input_path.stem,
        color=MOTION_COLOR,
    )
    sequence.mesh_seq.material.ambient = BODY_AMBIENT
    sequence.mesh_seq.material.diffuse = BODY_DIFFUSE

    vertices = numpy_data(sequence.vertices)
    ground_offset = -float(vertices[..., 1].min())
    sequence.position = np.array([0.0, ground_offset, 0.0], dtype=np.float32)
    bounds_min = vertices.min(axis=(0, 1)) + sequence.position
    bounds_max = vertices.max(axis=(0, 1)) + sequence.position
    bounds_center = (bounds_min + bounds_max) * 0.5
    half_extents = (bounds_max - bounds_min) * 0.5

    heading = body_facing_direction(sequence.joints)
    if args.camera_yaw:
        heading = Rotation.from_euler("y", args.camera_yaw, degrees=True).apply(heading)
    heading[1] = 0.0
    heading /= np.linalg.norm(heading)

    viewer = Viewer(size=(args.width, args.height), samples=args.samples)
    viewer.playback_fps = args.fps
    viewer.scene.add(sequence)
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
        args.width,
        args.height,
        fov=args.camera_fov,
        viewer=viewer,
    )
    viewer.scene.add(camera)
    viewer.set_temp_camera(camera)

    if not 0 <= args.frame < motion["frame_count"]:
        raise ValueError(
            f"--frame {args.frame} is outside [0, {motion['frame_count'] - 1}]"
        )
    viewer.scene.current_frame_id = args.frame

    logical_size = tuple(viewer.window_size)
    framebuffer_size = tuple(viewer.wnd.buffer_size)
    pixel_ratio = float(viewer.wnd.pixel_ratio)
    export_scale = pixel_ratio if args.export_scale is None else args.export_scale
    max_scale = min(
        framebuffer_size[0] / logical_size[0],
        framebuffer_size[1] / logical_size[1],
    )
    if export_scale > max_scale + 1e-6:
        raise ValueError(
            f"--export-scale {export_scale:g} exceeds framebuffer scale {max_scale:g}"
        )

    print(
        f"Loaded {motion['frame_count']} SMPL-X frames from {input_path.name}\n"
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
        if enabled:
            size = (
                int(round(logical_size[0] * export_scale)),
                int(round(logical_size[1] * export_scale)),
            )
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

        with tempfile.TemporaryDirectory(prefix="mesh_render_frames_") as frame_root:
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
        return

    output = args.output or f"{input_path.stem}.mp4"
    export_video(output)
    if args.mode == "both":
        set_export_viewport(False)
        viewer.scene.current_frame_id = args.frame
        print("Video saved; opening interactive AITViewer preview")
        viewer.run()


if __name__ == "__main__":
    main()
