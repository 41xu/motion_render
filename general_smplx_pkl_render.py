"""Render common single-person SMPL-X pickle layouts.

This is the format-flexible entry point. PromptHMR's multi-person
``people[track][smplx_world]`` layout is still supported by delegating to
``pkl_render.py``; flat MoSh++/SOMA files such as ``yoga.pkl`` are read here.
"""

from collections.abc import Mapping

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

import pkl_render as renderer


PROMPTHMR_LOADER = renderer.load_people
POSE_KEYS = ("fullpose", "full_pose", "full_pose_axis_angle", "poses", "pose")
TRANS_KEYS = ("trans", "transl", "translation", "translations")
BETA_KEYS = ("betas", "beta", "shape", "shapes")
MOTION_KEYS = ("smplx", "smplx_params", "params", "motion", "body")


def _pick(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _numpy(value, name):
    try:
        return np.asarray(renderer.numpy_data(value))
    except Exception as exc:
        raise ValueError(f"could not convert '{name}' to a NumPy array") from exc


def _motion_mapping(payload):
    candidates = [payload]
    candidates.extend(
        payload[key]
        for key in MOTION_KEYS
        if key in payload and isinstance(payload[key], Mapping)
    )
    for candidate in candidates:
        if _pick(candidate, POSE_KEYS) is not None:
            return candidate
        if _pick(candidate, ("body_pose", "poses_body")) is not None:
            return candidate
    raise ValueError(
        "no SMPL-X pose found; expected one of "
        f"{POSE_KEYS}, or separate root/body/hand pose fields"
    )


def _normalise_full_pose(value):
    pose = _numpy(value, "full pose").astype(np.float32)

    # Remove optional singleton person/batch dimensions.
    while pose.ndim > 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim == 3 and pose.shape == (55, 3, 3):
        pose = pose[None]

    if pose.ndim == 4 and pose.shape[-3:] == (55, 3, 3):
        matrices = pose.reshape(-1, 3, 3).copy()
        invalid = np.abs(np.linalg.det(matrices)) < 1e-6
        matrices[invalid] = np.eye(3, dtype=np.float32)
        return (
            Rotation.from_matrix(matrices)
            .as_rotvec()
            .reshape(len(pose), 165)
            .astype(np.float32)
        )

    while pose.ndim > 3 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim == 3 and pose.shape[-2:] == (55, 3):
        pose = pose.reshape(len(pose), 165)
    elif pose.ndim == 1 and pose.size == 165:
        pose = pose.reshape(1, 165)

    if pose.ndim != 2 or pose.shape[1] != 165:
        raise ValueError(
            "full SMPL-X pose must have shape (F, 165), (F, 55, 3), or "
            f"(F, 55, 3, 3); found {pose.shape}"
        )
    return pose


def _component(value, width, name):
    array = _numpy(value, name).astype(np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 1:
        if array.size % width:
            raise ValueError(f"'{name}' has incompatible shape {array.shape}")
        array = array.reshape(-1, width)
    elif array.ndim >= 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"'{name}' must contain {width} values per frame; found {array.shape}"
        )
    return array


def _broadcast_component(array, frames, name):
    if len(array) == frames:
        return array
    if len(array) == 1:
        return np.repeat(array, frames, axis=0)
    raise ValueError(f"'{name}' has {len(array)} frames, expected {frames}")


def _pose_from_components(motion):
    specs = (
        (("global_orient", "root_orient", "poses_root"), 3, "global orientation"),
        (("body_pose", "poses_body"), 63, "body pose"),
        (("jaw_pose", "poses_jaw"), 3, "jaw pose"),
        (("leye_pose", "left_eye_pose"), 3, "left eye pose"),
        (("reye_pose", "right_eye_pose"), 3, "right eye pose"),
        (("left_hand_pose", "poses_left_hand"), 45, "left hand pose"),
        (("right_hand_pose", "poses_right_hand"), 45, "right hand pose"),
    )
    arrays = []
    for keys, width, name in specs:
        value = _pick(motion, keys)
        arrays.append(None if value is None else _component(value, width, name))

    body = arrays[1]
    if body is None:
        raise ValueError("separate-component layout has no 63-value 'body_pose'")
    frames = max(len(array) for array in arrays if array is not None)

    completed = []
    for array, (_, width, name) in zip(arrays, specs):
        if array is None:
            completed.append(np.zeros((frames, width), dtype=np.float32))
        else:
            completed.append(_broadcast_component(array, frames, name))
    return np.concatenate(completed, axis=1)


def _load_pose(motion):
    value = _pick(motion, POSE_KEYS)
    if value is not None:
        try:
            return _normalise_full_pose(value)
        except ValueError:
            # Some exports call the 63-value body component simply "pose".
            if _pick(motion, ("body_pose", "poses_body")) is None:
                raise
    return _pose_from_components(motion)


def _load_translation(motion, frames):
    value = _pick(motion, TRANS_KEYS)
    if value is None:
        return np.zeros((frames, 3), dtype=np.float32)
    trans = _component(value, 3, "translation")
    return _broadcast_component(trans, frames, "translation")


def _load_betas(motion, payload):
    value = _pick(motion, BETA_KEYS)
    if value is None and motion is not payload:
        value = _pick(payload, BETA_KEYS)
    if value is None:
        return np.zeros(10, dtype=np.float32)

    betas = _numpy(value, "betas").astype(np.float32)
    if betas.ndim == 1:
        stable = betas
    else:
        betas = betas.reshape(-1, betas.shape[-1])
        stable = np.median(betas, axis=0)
    if stable.size < 10:
        stable = np.pad(stable, (0, 10 - stable.size))
    return stable[:10].astype(np.float32)


def load_people(path, source_key, track):
    payload = joblib.load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a dictionary, found {type(payload).__name__}")

    # Preserve support for PromptHMR files while keeping pkl_render.py as their
    # dedicated entry point.
    if payload.get("people"):
        return PROMPTHMR_LOADER(path, source_key, track)

    if track not in (None, 0):
        raise ValueError("flat single-person SMPL-X files only have track 0")

    motion = _motion_mapping(payload)
    pose = _load_pose(motion)
    trans = _load_translation(motion, len(pose))
    betas = _load_betas(motion, payload)
    return (
        [{"track_id": 0, "pose": pose, "trans": trans, "betas": betas}],
        len(pose),
    )


def main():
    renderer.main(
        load_people_fn=load_people,
        description=(
            "Render common SMPL-X pickle layouts, including flat MoSh++/SOMA "
            "fullpose/trans/betas files and PromptHMR results"
        ),
    )


if __name__ == "__main__":
    main()
