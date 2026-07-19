# My Own Motion Rendering Script for Visualization

thanks to motionfix and codex 🥺

## packages

```
python=3.12
ffmpeg
aitviewer
```


## Info

`modiff_render` for motionfix dataset render

`mesh_render` for a single SMPL-X JSON motion in motionx++

## MotionFix preview

```bash
python modiff_render.py --mode both --samples 8 --video-crf 18 --output motionfix_retina.mp4

# Three videos with the same fixed camera/floor and unchanged red/green colors.
python modiff_render.py --mode video --samples 8 --video-crf 18 \
  --video-variants all --output motionfix.mp4
# Writes: motionfix_overlap.mp4, motionfix_source.mp4, motionfix_target.mp4
```

## Single SMPL-X JSON motion

`Ways_to_Catch_A_Cold_clip1.json` stores one `smplx_params` record per annotation.
The renderer converts its camera coordinates to Y-up, grounds the complete
sequence, estimates the body's front, and fits one fixed camera to every frame.

```bash
# Interactive preview
python mesh_render.py Ways_to_Catch_A_Cold_clip1.json

# 1440x1440 Retina MP4, 8x MSAA and CRF 18 by default
python mesh_render.py Ways_to_Catch_A_Cold_clip1.json \
  --mode video \
  --output Ways_to_Catch_A_Cold_clip1.mp4

# I prefer this one.
# Export first, then browse the same scene interactively
python mesh_render.py Ways_to_Catch_A_Cold_clip1.json \
  --mode both \
  --output Ways_to_Catch_A_Cold_clip1.mp4

# If another file is already Y-up, or the auto view faces backward
python mesh_render.py other.json --input-coordinates y_up --camera-yaw 180

# Apply the JSON's jaw/expression animation instead of the neutral face
python mesh_render.py other.json --use-face
```

### Single SMPL-X JSON on a headless NVIDIA server

Use EGL when the server has no `$DISPLAY`. In headless mode, `--width` and
`--height` are the actual output dimensions and only `frame`/`video` modes are
available.

```bash
python mesh_render.py /path/to/motion.json \
  --mode video \
  --headless \
  --models /path/to/body_models \
  --output rendered_motion.mp4 \
  --width 1440 --height 1440 \
  --samples 8 \
  --video-crf 18

# Export one PNG instead.
python mesh_render.py /path/to/motion.json \
  --mode frame --frame 0 --headless \
  --models /path/to/body_models \
  --output rendered_motion.png \
  --width 1440 --height 1440
```

## Batch render all MotionFix samples

The batch renderer creates one AITViewer/OpenGL context for the complete run,
then replaces and releases the source/target meshes for each sample. Completed
MP4s are skipped, so running the same command again resumes the batch.

```bash
# Small smoke test first
python render_motionfix_all.py \
  --sample-ids 005454 001236 001363 \
  --output-dir motionfix_renders \
  --video-crf 18

# Render all 6,730 samples as overlap/source/target, 1440x1440 on Retina
caffeinate -dimsu python render_motionfix_all.py \
  --output-dir motionfix_renders \
  --samples 8 \
  --video-crf 18

# Resume after interruption: run the identical command again.
# Use --overwrite only when every existing output should be replaced.
```

Do not launch multiple copies on macOS: each process loads the 5 GB dataset and
creates its own Qt/OpenGL context. A full single-process run is expected to take
many hours; use the resumable output directory.

### Headless NVIDIA server (EGL)

Use the headless path when the server has no `$DISPLAY`. `--width` and
`--height` are the real output dimensions in this mode, so request 1440 directly
instead of relying on the macOS Retina pixel ratio.

```bash
# First verify one sample and three output variants.
python render_motionfix_all.py \
  --headless \
  --motionfix-pth /path/to/motionfix.pth.tar \
  --body-models /path/to/body_models \
  --output-dir motionfix_renders \
  --sample-ids 005454 \
  --width 1440 --height 1440 \
  --samples 8 \
  --video-crf 18

# Remove --sample-ids to render the entire dataset. Running the same command
# again resumes by skipping completed source/target/overlap MP4s.
```

The script prints both the CUDA compute device and the OpenGL vendor/renderer at
startup. On the H200 server the OpenGL renderer should contain `NVIDIA H200`;
the script rejects known software renderers such as `llvmpipe` in headless mode.

## Batch render MotionX++ SMPL-X JSON files

`render_mesh_all.py` reads the `id` and `motion` fields from
`motionx++_test_render.json`, then writes one `<id>.mp4` per entry. It reuses one
SMPL-X layer and one OpenGL context, skips completed outputs, writes MP4s
atomically, and records individual failures without stopping the batch.

```bash
# H200/EGL smoke test: render the first two entries at 1440x1440.
python render_mesh_all.py motionx++_test_render.json \
  --headless \
  --models /path/to/body_models \
  --output-dir motionxpp_renders \
  --limit 2 \
  --width 1440 --height 1440 \
  --samples 8 \
  --video-crf 18

# Full batch: remove --limit. Re-running resumes completed outputs.
python render_mesh_all.py motionx++_test_render.json \
  --headless \
  --models /path/to/body_models \
  --output-dir motionxpp_renders \
  --width 1440 --height 1440 \
  --samples 8 \
  --video-crf 18
```

The manifest already contains absolute server paths. If the dataset is moved,
pass `--motion-root /new/path/to/global_motion`; category subdirectories such as
`animation`, `idea400`, and `haa500` are preserved. Failures are written to
`<output-dir>/render_failures.json`.
