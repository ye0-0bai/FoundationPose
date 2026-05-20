"""Shared helpers for delayed FoundationPose tracking registration."""

import numpy as np


MIN_REGISTRATION_PIXELS = 4
MIN_VALID_DEPTH_M = 0.001


def registration_inputs_are_valid(mask, depth):
    """Return True when registration has enough masked pixels with valid depth."""

    valid = (np.asarray(mask) > 0) & (np.asarray(depth) >= MIN_VALID_DEPTH_M)
    return int(valid.sum()) >= MIN_REGISTRATION_PIXELS


def invalid_pose():
    """Create the all-zero pose sentinel used for frames before registration."""

    return np.zeros((4, 4), dtype=np.float64)


def is_invalid_pose(pose):
    """Return True for the all-zero invalid-pose sentinel."""

    return not np.asarray(pose).any()
