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
