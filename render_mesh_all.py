"""Batch render SMPL-X JSON motions listed in a JSON manifest.

The expected manifest is a list of objects such as::

    [{"id": "animation_example", "motion": "/path/to/example.json"}]

One AITViewer/OpenGL context and one SMPL-X layer are reused for the entire
batch. Existing MP4s are skipped, and each completed output is installed
atomically so an interrupted run can safely be resumed.
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

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

from mesh_render import (
    BODY_AMBIENT,
    BODY_DIFFUSE,
    DEFAULT_MODELS,
    MOTION_COLOR,
    body_facing_direction,
    convert_global_coordinates,
    find_ffmpeg,
    load_smplx_json,
    numpy_data,
)


DEFAULT_MANIFEST = Path(__file__).with_name("motionx++_test_render.json")


def safe_output_stem(sample_id):
    """Keep the ID as the filename while blocking separators/control bytes."""

    stem = re.sub(r"[\\/\x00-\x1f]+", "_", sample_id).strip()
    if stem in ("", ".", ".."):
        stem = "motion"
    return stem or "motion"


def load_manifest(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a top-level JSON array")

    records = []
    seen_ids = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        sample_id = item.get("id")
        motion_path = item.get("motion")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Manifest entry {index} has an invalid 'id'")
        if not isinstance(motion_path, str) or not motion_path:
            raise ValueError(f"Manifest entry {index} has an invalid 'motion' path")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate manifest id: {sample_id}")
        seen_ids.add(sample_id)
        records.append({"id": sample_id, "motion": motion_path})

    output_names = [safe_output_stem(record["id"]) for record in records]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Two manifest IDs map to the same sanitized output filename")
    return records


def resolve_motion_path(record, motion_root):
    """Resolve an original absolute path or remap its global_motion suffix."""

    original = Path(record["motion"]).expanduser()
    if original.is_file() or motion_root is None:
        return original

    root = Path(motion_root).expanduser()
    parts = original.parts
    try:
        anchor = parts.index("global_motion")
        relative = Path(*parts[anchor + 1 :])
    except ValueError:
        relative = Path(original.parent.name) / original.name
    return root / relative


def create_egl_headless_renderer(size, samples, backend):
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


def create_batch_renderer(args):
    if args.export_scale is not None and args.export_scale <= 0:
        raise ValueError("--export-scale must be greater than zero")

    logical_size = (args.width, args.height)
    if args.headless:
        export_scale = 1.0 if args.export_scale is None else args.export_scale
        export_size = (
            int(round(args.width * export_scale)),
            int(round(args.height * export_scale)),
        )
        viewer = create_egl_headless_renderer(
            export_size, args.samples, args.headless_backend
        )
        framebuffer_size = tuple(viewer.wnd.buffer_size)
    else:
        viewer = Viewer(size=logical_size, samples=args.samples)
        logical_size = tuple(viewer.window_size)
        framebuffer_size = tuple(viewer.wnd.buffer_size)
        max_scale = min(
            framebuffer_size[0] / logical_size[0],
            framebuffer_size[1] / logical_size[1],
        )
        export_scale = (
            float(viewer.wnd.pixel_ratio)
            if args.export_scale is None
            else args.export_scale
        )
        if export_scale > max_scale + 1e-6:
            raise ValueError(
                f"--export-scale {export_scale:g} exceeds framebuffer scale "
                f"{max_scale:g}"
            )
        export_size = (
            int(round(logical_size[0] * export_scale)),
            int(round(logical_size[1] * export_scale)),
        )
        viewer.window_size = export_size
        viewer._resize_viewports()
        viewer.wnd.fbo.viewport = (0, 0, export_size[0], export_size[1])

        # Preserve physical Retina pixels instead of letting AITViewer resize
        # them back to the logical window dimensions.
        def physical_frame_reader(alpha=False):
            fmt = "RGBA" if alpha else "RGB"
            components = 4 if alpha else 3
            width, height = viewer.window_size
            viewport = (0, 0, width, height)
            image = Image.frombytes(
                fmt,
                (width, height),
                viewer.wnd.fbo.read(
                    viewport=viewport, alignment=1, components=components
                ),
            )
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        viewer.get_current_frame_as_image = physical_frame_reader

    gl_vendor = str(viewer.ctx.info.get("GL_VENDOR", "unknown"))
    gl_renderer = str(viewer.ctx.info.get("GL_RENDERER", "unknown"))
    gl_version = str(viewer.ctx.info.get("GL_VERSION", "unknown"))
    print(f"OpenGL: vendor={gl_vendor}, renderer={gl_renderer}, version={gl_version}")
    if args.headless and any(
        marker in gl_renderer.lower()
        for marker in ("llvmpipe", "softpipe", "swrast", "software rasterizer")
    ):
        raise RuntimeError(
            "Headless OpenGL is using a software renderer instead of the GPU: "
            f"{gl_renderer}"
        )

    viewer.playback_fps = args.fps
    viewer.scene.origin.enabled = False
    viewer.scene.light_mode = "default"
    viewer.scene.ambient_strength = args.ambient_strength
    for light in viewer.scene.lights:
        light.strength = args.light_strength
    viewer.scene.background_color = (0.93, 0.93, 0.93, 1.0)
    viewer.scene.floor.c1 = (0.68, 0.68, 0.68, 1.0)
    viewer.scene.floor.c2 = (0.52, 0.52, 0.52, 1.0)
    viewer._init_scene()

    print(
        f"Batch render buffer: window={logical_size}, "
        f"framebuffer={framebuffer_size}, export={export_size}, "
        f"pixel_ratio={viewer.wnd.pixel_ratio}, headless={args.headless}"
    )
    return viewer


def build_sequence_and_camera(sample_id, input_path, smpl_layer, viewer, args):
    motion = load_smplx_json(input_path)
    root, trans = convert_global_coordinates(
        motion["root"], motion["trans"], args.input_coordinates
    )
    if not args.keep_start_position:
        trans[:, 0] -= trans[0, 0]
        trans[:, 2] -= trans[0, 2]

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
        name=sample_id,
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
        heading = Rotation.from_euler(
            "y", args.camera_yaw, degrees=True
        ).apply(heading)
    heading[1] = 0.0
    heading /= np.linalg.norm(heading)

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
    camera_distance = (
        auto_distance if args.camera_distance is None else args.camera_distance
    )
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
    return sequence, camera, motion["frame_count"], ground_offset, camera_distance


def export_video(viewer, output_path, args, ffmpeg):
    if args.direct_aitviewer:
        viewer.export_video(
            output_path=str(output_path),
            output_fps=args.fps,
            quality="high",
            ensure_no_overwrite=False,
        )
        return

    temp_prefix = (
        f"mesh_batch_w{args.worker_index}_frames_"
        if args.worker_index is not None
        else "mesh_batch_frames_"
    )
    with tempfile.TemporaryDirectory(
        prefix=temp_prefix, dir=args.frame_temp_dir
    ) as frame_root:
        viewer.export_video(
            output_path=None,
            frame_dir=frame_root,
            output_fps=args.fps,
        )
        frame_dir = os.path.join(frame_root, "0000")
        command = [
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
        ]
        if args.ffmpeg_threads is not None:
            command.extend(["-threads", str(args.ffmpeg_threads)])
        command.extend(
            [
                "-crf",
                str(args.video_crf),
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        subprocess.run(command, check=True)


def render_one(record, input_path, output_path, smpl_layer, viewer, args, ffmpeg):
    sequence = None
    camera = None
    try:
        sequence, camera, frame_count, ground_offset, camera_distance = (
            build_sequence_and_camera(
                record["id"], input_path, smpl_layer, viewer, args
            )
        )
        viewer.scene.add(sequence, camera)
        viewer.set_temp_camera(camera)
        viewer.scene.current_frame_id = 0
        viewer.scene.make_renderable(viewer.ctx)
        print(
            f"[{record['id']}] frames={frame_count}, "
            f"ground={ground_offset:.3f}m, camera_distance={camera_distance:.3f}m"
        )

        part_path = output_path.with_name(
            f".{output_path.stem}.part{output_path.suffix}"
        )
        part_path.unlink(missing_ok=True)
        try:
            export_video(viewer, part_path, args, ffmpeg)
            os.replace(part_path, output_path)
        finally:
            part_path.unlink(missing_ok=True)
    finally:
        if camera is not None:
            try:
                viewer.reset_camera()
            except Exception:
                pass
        nodes = [node for node in (sequence, camera) if node is not None]
        if nodes:
            viewer.scene.remove(*nodes)
        viewer.scene.current_frame_id = 0


def write_failure_log(path, failures):
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def launch_parallel_workers(args):
    """Launch one persistent rendering subprocess per GPU."""

    if not args.headless:
        raise ValueError("--workers greater than 1 requires --headless")
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else list(range(args.workers))
    egl_device_ids = (
        args.egl_device_ids if args.egl_device_ids is not None else gpu_ids
    )
    if len(gpu_ids) < args.workers:
        raise ValueError("--gpu-ids must provide at least one CUDA GPU per worker")
    if len(egl_device_ids) < args.workers:
        raise ValueError(
            "--egl-device-ids must provide at least one EGL device per worker"
        )

    cpu_count = os.cpu_count() or args.workers
    ffmpeg_threads = args.ffmpeg_threads
    if ffmpeg_threads is None:
        ffmpeg_threads = max(1, cpu_count // args.workers)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for worker_index in range(args.workers):
        (output_dir / f"render_failures.worker_{worker_index:02d}.json").unlink(
            missing_ok=True
        )
    (output_dir / "render_failures.json").unlink(missing_ok=True)
    print(
        f"Launching {args.workers} workers across CUDA GPUs "
        f"{gpu_ids[:args.workers]} and EGL devices {egl_device_ids[:args.workers]}"
    )
    print(
        f"CPU allocation: {cpu_count} logical CPUs, "
        f"{ffmpeg_threads} FFmpeg threads per worker"
    )

    processes = []
    for worker_index in range(args.workers):
        cuda_gpu = gpu_ids[worker_index]
        egl_device = egl_device_ids[worker_index]
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_gpu)
        environment["GLCONTEXT_DEVICE_INDEX"] = str(egl_device)
        environment["PYTHONUNBUFFERED"] = "1"

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--workers",
            "1",
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(args.workers),
            "--device",
            "cuda:0",
            "--ffmpeg-threads",
            str(ffmpeg_threads),
        ]
        print(
            f"Starting worker {worker_index}: CUDA GPU {cuda_gpu}, "
            f"EGL device {egl_device}"
        )
        processes.append(subprocess.Popen(command, env=environment))

    try:
        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        print("Interrupted; terminating workers...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return 130

    combined_failures = []
    for worker_index in range(args.workers):
        worker_failure_log = output_dir / f"render_failures.worker_{worker_index:02d}.json"
        if worker_failure_log.is_file():
            try:
                with worker_failure_log.open("r", encoding="utf-8") as handle:
                    combined_failures.extend(json.load(handle))
            except (json.JSONDecodeError, OSError) as exc:
                combined_failures.append(
                    {
                        "worker": worker_index,
                        "error": f"Could not read worker failure log: {exc}",
                    }
                )

    combined_failure_log = output_dir / "render_failures.json"
    if combined_failures:
        write_failure_log(combined_failure_log, combined_failures)
        print(
            f"Parallel run completed with {len(combined_failures)} sample failures: "
            f"{combined_failure_log}"
        )
    else:
        combined_failure_log.unlink(missing_ok=True)

    crashed_workers = [
        index for index, return_code in enumerate(return_codes) if return_code != 0
    ]
    if crashed_workers:
        print(f"Workers with non-zero exit status: {crashed_workers}")
        return 1
    print("All parallel workers finished")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch render SMPL-X JSON motions with the mesh_render camera style"
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(DEFAULT_MANIFEST),
        help="JSON list containing id/motion entries",
    )
    parser.add_argument("--output-dir", default="motionxpp_renders")
    default_models = os.environ.get("SMPLX_MODELS")
    if default_models is None:
        default_models = (
            DEFAULT_MODELS if Path(DEFAULT_MODELS).is_dir() else str(C.smplx_models)
        )
    parser.add_argument(
        "--models",
        default=default_models,
        help="directory containing SMPL-X models (or set SMPLX_MODELS)",
    )
    parser.add_argument(
        "--motion-root",
        default=None,
        help=(
            "optional replacement for the path through global_motion; "
            "keeps category/name.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="render only these manifest IDs",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="zero-based start position after --sample-ids filtering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="render at most this many manifest entries",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of persistent GPU rendering processes (default: 1)",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="physical CUDA GPU IDs assigned to workers (default: 0..workers-1)",
    )
    parser.add_argument(
        "--egl-device-ids",
        nargs="+",
        type=int,
        default=None,
        help="EGL device indices; defaults to the corresponding --gpu-ids",
    )
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--gender", choices=("neutral", "female", "male"), default="neutral"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--use-face", action="store_true")
    parser.add_argument(
        "--input-coordinates",
        choices=("camera", "y_up", "z_up"),
        default="camera",
    )
    parser.add_argument("--keep-start-position", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--camera-fov", type=float, default=45.0)
    parser.add_argument("--camera-margin", type=float, default=1.2)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=8, help="MSAA sample count")
    parser.add_argument(
        "--export-scale",
        type=float,
        default=None,
        help="GUI default: display pixel ratio; headless default: 1.0",
    )
    parser.add_argument("--ambient-strength", type=float, default=1.2)
    parser.add_argument("--light-strength", type=float, default=0.9)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--direct-aitviewer", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--headless-backend", default="egl")
    parser.add_argument(
        "--device",
        default="auto",
        help="SMPL-X compute device; auto selects CUDA when available",
    )
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=25,
        help="run Python/GPU cache cleanup every N entries; 0 disables it",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=None,
        help="x264 threads per worker; parallel mode defaults to CPU count/workers",
    )
    parser.add_argument(
        "--frame-temp-dir",
        default=None,
        help="directory for lossless PNG frames; use fast local NVMe when possible",
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
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.gpu_ids is not None and any(gpu_id < 0 for gpu_id in args.gpu_ids):
        parser.error("--gpu-ids must be non-negative")
    if args.egl_device_ids is not None and any(
        device_id < 0 for device_id in args.egl_device_ids
    ):
        parser.error("--egl-device-ids must be non-negative")
    if args.worker_count <= 0:
        parser.error("--worker-count must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.worker_count:
        parser.error("--worker-index must be in [0, --worker-count)")
    if args.ffmpeg_threads is not None and args.ffmpeg_threads <= 0:
        parser.error("--ffmpeg-threads must be positive")
    return args


def main():
    args = parse_args()
    if args.workers > 1 and args.worker_index is None:
        return launch_parallel_workers(args)

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(manifest_path)
    if args.sample_ids:
        requested = set(args.sample_ids)
        known = {record["id"] for record in records}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown --sample-ids: {', '.join(unknown)}")
        records = [record for record in records if record["id"] in requested]
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    if args.worker_index is not None:
        records = records[args.worker_index :: args.worker_count]
    if not records:
        worker_label = (
            f" for worker {args.worker_index}" if args.worker_index is not None else ""
        )
        print(f"No manifest entries selected{worker_label}")
        return

    if args.frame_temp_dir is not None:
        frame_temp_dir = Path(args.frame_temp_dir).expanduser().resolve()
        frame_temp_dir.mkdir(parents=True, exist_ok=True)
        args.frame_temp_dir = str(frame_temp_dir)

    C.smplx_models = str(Path(args.models).expanduser())
    C.auto_set_floor = False
    if args.device == "auto":
        compute_device = "cuda:0" if torch.cuda.is_available() else str(C.device)
    else:
        compute_device = args.device
    print(f"Loaded {len(records)} entries from {manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"SMPL-X compute device: {compute_device}")
    print(f"Body models: {C.smplx_models}")
    if args.worker_index is not None:
        print(
            f"Worker {args.worker_index}/{args.worker_count}: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"GLCONTEXT_DEVICE_INDEX={os.environ.get('GLCONTEXT_DEVICE_INDEX')}"
        )

    ffmpeg = find_ffmpeg()
    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender=args.gender,
        num_betas=10,
        num_expression_coeffs=10,
        device=compute_device,
    )
    viewer = create_batch_renderer(args)

    failures = []
    rendered = 0
    skipped = 0
    start_time = time.time()
    failure_log = (
        output_dir / f"render_failures.worker_{args.worker_index:02d}.json"
        if args.worker_index is not None
        else output_dir / "render_failures.json"
    )
    try:
        for index, record in enumerate(records, 1):
            elapsed = time.time() - start_time
            output_path = output_dir / f"{safe_output_stem(record['id'])}.mp4"
            print(
                f"\n=== [{index}/{len(records)}] {record['id']} "
                f"(elapsed {elapsed:.1f}s) ==="
            )
            if output_path.exists() and not args.overwrite:
                print(f"Exists, skipping: {output_path}")
                skipped += 1
                continue

            input_path = resolve_motion_path(record, args.motion_root)
            try:
                if not input_path.is_file():
                    raise FileNotFoundError(input_path)
                print(f"Input: {input_path}")
                print(f"Output: {output_path}")
                render_one(
                    record,
                    input_path,
                    output_path,
                    smpl_layer,
                    viewer,
                    args,
                    ffmpeg,
                )
                rendered += 1
            except Exception as exc:
                traceback.print_exc()
                failures.append(
                    {
                        "id": record["id"],
                        "motion": record["motion"],
                        "resolved_motion": str(input_path),
                        "error": str(exc),
                    }
                )
                write_failure_log(failure_log, failures)
                # Release tensors left by a failed sample immediately instead
                # of waiting for the regular cleanup interval.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if args.gc_interval and index % args.gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    finally:
        viewer.close()

    if not failures:
        failure_log.unlink(missing_ok=True)
    total = time.time() - start_time
    print(
        f"\nDone in {total:.1f}s: rendered={rendered}, "
        f"skipped={skipped}, failed={len(failures)}"
    )
    if failures:
        print(f"Failure details: {failure_log}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
