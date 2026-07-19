"""Batch render every MotionFix sample: source, target, and overlap videos.

Layout: <output-dir>/<sample_id>_{overlap,source,target}.mp4
Behaviour: skips samples whose three variants already exist. Failures per
sample are caught so one bad entry doesn't stop the run.
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
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
from tools.transformers3d import get_z_rot


C.auto_set_floor = False

FPS = 30
LOCAL_MOTIONFIX_PTH = "/Users/sxu/Downloads/motionfix-dataset/motionfix.pth.tar"
LOCAL_BODY_MODELS = "/Users/sxu/Projects/body_models"
INPUT_IS_Z_UP = True
SOURCE_COLOR = (0.78, 0.28, 0.24, 1.0)
TARGET_COLOR = (0.26, 0.68, 0.36, 1.0)
BODY_AMBIENT = 0.45
BODY_DIFFUSE = 0.65
VARIANTS = ("overlap", "source", "target")


def to_axis_angle(rots, n_joints):
    rots = np.asarray(rots)
    if rots.ndim == 2:
        if rots.shape[1] != n_joints * 3:
            raise ValueError(
                f"Expected flattened rotations of shape (F, {n_joints * 3}), "
                f"but got {rots.shape}"
            )
        return rots.reshape(len(rots), n_joints, 3)
    if rots.ndim == 3 and rots.shape[-1] == 3:
        return rots
    if rots.ndim == 3 and rots.shape[-1] == 4:
        return Rotation.from_quat(
            rots.reshape(-1, 4)
        ).as_rotvec().reshape(len(rots), n_joints, 3)
    if rots.ndim == 4 and rots.shape[-2:] == (3, 3):
        return Rotation.from_matrix(
            rots.reshape(-1, 3, 3)
        ).as_rotvec().reshape(len(rots), n_joints, 3)
    raise ValueError(f"Unsupported rotation shape: {rots.shape}")


def build_sequence(motion, smpl_layer, name, color):
    rots = np.asarray(motion["rots"])
    trans = np.asarray(motion["trans"], dtype=np.float32)
    joint_positions = np.asarray(motion["joint_positions"], dtype=np.float32)

    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Expected trans shape (F, 3), got {trans.shape}")
    if joint_positions.ndim != 3 or joint_positions.shape[2] != 3:
        raise ValueError(
            f"Expected joint_positions shape (F, J, 3), got {joint_positions.shape}"
        )

    n_frames, n_joints, _ = joint_positions.shape
    if len(trans) != n_frames or len(rots) != n_frames:
        raise ValueError("rots, trans and joint_positions must share the frame count")

    axis_angle = to_axis_angle(rots, n_joints).astype(np.float32)
    poses_root = axis_angle[:, 0]
    poses_body = axis_angle[:, 1:].reshape(n_frames, -1)

    seq = SMPLSequence(
        poses_body=poses_body,
        poses_root=poses_root,
        trans=trans,
        smpl_layer=smpl_layer,
        z_up=INPUT_IS_Z_UP,
        name=name,
        color=color,
    )
    seq.mesh_seq.material.ambient = BODY_AMBIENT
    seq.mesh_seq.material.diffuse = BODY_DIFFUSE
    ground_offset = -float(seq.vertices[..., 2].min().item())
    seq.position = np.array([0.0, ground_offset, 0.0], dtype=np.float32)
    return seq, poses_root, ground_offset


def sequence_bounds(seq, ground_offset):
    verts = seq.vertices
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    else:
        verts = np.asarray(verts)
    bmin = np.array(
        [verts[..., 0].min(), verts[..., 2].min() + ground_offset, -verts[..., 1].max()],
        dtype=np.float32,
    )
    bmax = np.array(
        [verts[..., 0].max(), verts[..., 2].max() + ground_offset, -verts[..., 1].min()],
        dtype=np.float32,
    )
    return bmin, bmax


def infer_model_type(n_joints):
    if n_joints == 22:
        return "smplh"
    if n_joints == 24:
        return "smpl"
    raise ValueError(
        f"Cannot infer body model from {n_joints} joints. Expected 22 (SMPL-X body) or 24 (SMPL)."
    )


def _create_egl_headless_renderer(size, samples, backend):
    """Construct AITViewer's headless window while explicitly selecting EGL."""

    # AITViewer's manually-created moderngl_window does not forward ``backend``
    # in every released version. Inject it only while the one context is built.
    original_create_context = moderngl.create_standalone_context

    def create_context_with_backend(*context_args, **context_kwargs):
        context_kwargs.setdefault("backend", backend)
        return original_create_context(*context_args, **context_kwargs)

    moderngl.create_standalone_context = create_context_with_backend
    try:
        return HeadlessRenderer(size=size, samples=samples)
    finally:
        moderngl.create_standalone_context = original_create_context


def create_batch_renderer(args):
    """Create exactly one OpenGL context for the whole batch."""

    if args.export_scale is not None and args.export_scale <= 0:
        raise ValueError("--export-scale must be greater than zero")

    logical_window_size = (args.width, args.height)
    if args.headless:
        # There is no Retina pixel ratio in EGL. The requested framebuffer is
        # the actual exported resolution (or width/height times an explicit scale).
        export_scale = 1.0 if args.export_scale is None else args.export_scale
        export_size = (
            int(round(args.width * export_scale)),
            int(round(args.height * export_scale)),
        )
        renderer = _create_egl_headless_renderer(
            export_size, args.samples, args.headless_backend
        )
        framebuffer_size = tuple(renderer.wnd.buffer_size)
    else:
        renderer = Viewer(size=logical_window_size, samples=args.samples)
        logical_window_size = tuple(renderer.window_size)
        framebuffer_size = tuple(renderer.wnd.buffer_size)
        max_export_scale = min(
            framebuffer_size[0] / logical_window_size[0],
            framebuffer_size[1] / logical_window_size[1],
        )
        export_scale = (
            float(renderer.wnd.pixel_ratio)
            if args.export_scale is None
            else args.export_scale
        )
        if export_scale > max_export_scale + 1e-6:
            raise ValueError(
                f"--export-scale {export_scale:g} exceeds available framebuffer scale "
                f"{max_export_scale:g}"
            )

        export_size = (
            int(round(logical_window_size[0] * export_scale)),
            int(round(logical_window_size[1] * export_scale)),
        )
        renderer.window_size = export_size
        renderer._resize_viewports()
        renderer.wnd.fbo.viewport = (0, 0, export_size[0], export_size[1])

        # The GUI framebuffer can be larger than the logical Retina window.
        # Headless rendering must instead use AITViewer's own MSAA resolve path.
        def get_export_frame_as_image(alpha=False):
            fmt = "RGBA" if alpha else "RGB"
            components = 4 if alpha else 3
            width, height = renderer.window_size
            viewport = (0, 0, width, height)
            image = Image.frombytes(
                fmt,
                (width, height),
                renderer.wnd.fbo.read(
                    viewport=viewport, alignment=1, components=components
                ),
            )
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        renderer.get_current_frame_as_image = get_export_frame_as_image

    gl_vendor = str(renderer.ctx.info.get("GL_VENDOR", "unknown"))
    gl_renderer = str(renderer.ctx.info.get("GL_RENDERER", "unknown"))
    gl_version = str(renderer.ctx.info.get("GL_VERSION", "unknown"))
    print(f"OpenGL: vendor={gl_vendor}, renderer={gl_renderer}, version={gl_version}")
    if args.headless and any(
        marker in gl_renderer.lower()
        for marker in ("llvmpipe", "softpipe", "swrast", "software rasterizer")
    ):
        raise RuntimeError(
            "Headless OpenGL is using a software renderer instead of the GPU: "
            f"{gl_renderer}"
        )
    renderer.playback_fps = FPS
    renderer.scene.origin.enabled = False
    renderer.scene.light_mode = "default"
    renderer.scene.ambient_strength = args.ambient_strength
    for light in renderer.scene.lights:
        light.strength = args.light_strength
    renderer.scene.background_color = (0.93, 0.93, 0.93, 1.0)
    renderer.scene.floor.c1 = (0.68, 0.68, 0.68, 1.0)
    renderer.scene.floor.c2 = (0.52, 0.52, 0.52, 1.0)
    renderer._init_scene()

    print(
        f"Batch render buffer: window={logical_window_size}, "
        f"framebuffer={framebuffer_size}, export={export_size}, "
        f"pixel_ratio={renderer.wnd.pixel_ratio}, headless={args.headless}"
    )
    return renderer


def render_sample(
    sample_id,
    sample,
    output_dir,
    args,
    smpl_layer,
    renderer,
    ffmpeg_executable,
):
    """Render one MotionFix sample. Returns list of variants actually written."""
    output_paths = {v: output_dir / f"{sample_id}_{v}.mp4" for v in VARIANTS}
    variants_to_render = [
        v for v in VARIANTS if args.overwrite or not output_paths[v].exists()
    ]
    if not variants_to_render:
        print(f"[{sample_id}] all outputs exist, skipping")
        return []

    source_seq, poses_root, source_ground = build_sequence(
        sample["motion_source"], smpl_layer, "Source motion", SOURCE_COLOR
    )
    target_seq, _, target_ground = build_sequence(
        sample["motion_target"], smpl_layer, "Target motion", TARGET_COLOR
    )

    if source_seq.n_frames != target_seq.n_frames:
        # not strictly required by aitviewer, but keeps camera fit consistent.
        pass

    source_bmin, source_bmax = sequence_bounds(source_seq, source_ground)
    target_bmin, target_bmax = sequence_bounds(target_seq, target_ground)
    bmin = np.minimum(source_bmin, target_bmin)
    bmax = np.maximum(source_bmax, target_bmax)
    center = (bmin + bmax) * 0.5
    half_extents = (bmax - bmin) * 0.5

    root_z_rotation = get_z_rot(torch.from_numpy(poses_root[0]), in_format="aa")
    heading = -root_z_rotation[:, 1]
    camera_direction = np.array(
        [heading[0].item(), 0.0, -heading[1].item()], dtype=np.float32
    )
    camera_direction /= np.linalg.norm(camera_direction)
    camera_right = np.array(
        [camera_direction[2], 0.0, -camera_direction[0]], dtype=np.float32
    )
    half_width = float(np.dot(np.abs(camera_right), half_extents))
    half_depth = float(np.dot(np.abs(camera_direction), half_extents))
    tan_half_vertical = np.tan(np.deg2rad(args.camera_fov) * 0.5)
    aspect = args.width / args.height
    tan_half_horizontal = tan_half_vertical * aspect
    camera_distance = half_depth + args.camera_margin * max(
        half_extents[1] / tan_half_vertical,
        half_width / tan_half_horizontal,
    )
    camera_position = center + camera_direction * camera_distance
    camera_position[1] = args.camera_height

    renderer.scene.add(source_seq, target_seq)

    camera = PinholeCamera(
        camera_position,
        center,
        renderer.window_size[0],
        renderer.window_size[1],
        fov=args.camera_fov,
        viewer=renderer,
    )
    renderer.scene.add(camera)
    renderer.set_temp_camera(camera)
    renderer.scene.current_frame_id = 0
    # The scene/context already exists. Only newly added nodes need GPU setup.
    renderer.scene.make_renderable(renderer.ctx)

    def export_video(output_path):
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
            renderer.export_video(output_path=None, frame_dir=frame_root, output_fps=FPS)
            frame_dir = os.path.join(frame_root, "0000")
            subprocess.run(
                [
                    ffmpeg_executable, "-y", "-loglevel", "error",
                    "-framerate", str(FPS),
                    "-i", os.path.join(frame_dir, "frame_%06d.png"),
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", str(args.video_crf),
                    "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags", "+faststart",
                    output_path,
                ],
                check=True,
            )

    written = []
    try:
        for variant in variants_to_render:
            source_seq.enabled = variant in ("overlap", "source")
            target_seq.enabled = variant in ("overlap", "target")
            out_path = output_paths[variant]
            print(f"[{sample_id}] rendering {variant} -> {out_path}")
            # Write atomically so a killed process never leaves a corrupt file
            # that would be mistaken for a completed result on resume.
            part_path = out_path.with_name(f".{out_path.stem}.part{out_path.suffix}")
            part_path.unlink(missing_ok=True)
            export_video(str(part_path))
            os.replace(part_path, out_path)
            written.append((variant, str(out_path)))
    finally:
        try:
            renderer.reset_camera()
        except Exception:
            pass
        renderer.scene.remove(source_seq, target_seq, camera)
        renderer.scene.current_frame_id = 0
    return written


def resolve_ffmpeg():
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
    return ffmpeg_executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch render every MotionFix sample as source/target/overlap MP4s"
    )
    parser.add_argument(
        "--motionfix-pth",
        default=os.environ.get("MOTIONFIX_PTH", LOCAL_MOTIONFIX_PTH),
    )
    default_body_models = os.environ.get("SMPLX_MODELS")
    if default_body_models is None:
        default_body_models = (
            LOCAL_BODY_MODELS if Path(LOCAL_BODY_MODELS).is_dir() else str(C.smplx_models)
        )
    parser.add_argument(
        "--body-models",
        default=default_body_models,
        help="SMPL/SMPL-H model directory (or set SMPLX_MODELS)",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-render even when the mp4 for a variant already exists",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="restrict batch to a subset of dataset keys",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
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
        help="ModernGL standalone context backend used by --headless (default: egl)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SMPL compute device; auto selects CUDA when available",
    )
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    parser.add_argument("--camera-margin", type=float, default=1.2)
    parser.add_argument("--ambient-strength", type=float, default=1.2)
    parser.add_argument("--light-strength", type=float, default=0.9)
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=25,
        help="run Python/GPU cache cleanup every N samples; 0 disables it",
    )
    parser.add_argument(
        "--video-quality", choices=("high", "medium", "low"), default="high"
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=None,
        help="custom H.264 CRF via lossless PNG frames; try 18 for a sharper video",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ffmpeg_executable = resolve_ffmpeg()

    C.smplx_models = str(Path(args.body_models).expanduser())
    if args.device == "auto":
        compute_device = "cuda:0" if torch.cuda.is_available() else str(C.device)
    else:
        compute_device = args.device
    print(f"SMPL compute device: {compute_device}")
    print(f"Body models: {C.smplx_models}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MotionFix dataset from {args.motionfix_pth}")
    data_dict = joblib.load(args.motionfix_pth)
    all_ids = (
        list(args.sample_ids) if args.sample_ids else sorted(data_dict.keys())
    )
    print(f"Rendering {len(all_ids)} samples into {output_dir.resolve()}")

    first = data_dict[all_ids[0]]
    n_joints = np.asarray(first["motion_source"]["joint_positions"]).shape[1]
    model_type = infer_model_type(n_joints)
    print(f"Using SMPL layer: model_type={model_type}, joints={n_joints}")
    smpl_layer = SMPLLayer(
        model_type=model_type, gender="neutral", device=compute_device
    )
    renderer = create_batch_renderer(args)

    failures = []
    start_time = time.time()
    try:
        for i, sid in enumerate(all_ids, 1):
            elapsed = time.time() - start_time
            print(f"\n=== [{i}/{len(all_ids)}] {sid} (elapsed {elapsed:.1f}s) ===")
            try:
                render_sample(
                    sid,
                    data_dict[sid],
                    output_dir,
                    args,
                    smpl_layer,
                    renderer,
                    ffmpeg_executable,
                )
            except Exception as exc:
                traceback.print_exc()
                failures.append((sid, str(exc)))

            if args.gc_interval and i % args.gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    finally:
        # Do not destroy/recreate QApplication inside the loop. One final AITViewer
        # cleanup is safe and clears its process-wide shader caches.
        renderer.close()

    total = time.time() - start_time
    print(f"\nDone in {total:.1f}s. Rendered {len(all_ids) - len(failures)}/{len(all_ids)} samples.")
    if failures:
        print(f"{len(failures)} failures:")
        for sid, err in failures:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
