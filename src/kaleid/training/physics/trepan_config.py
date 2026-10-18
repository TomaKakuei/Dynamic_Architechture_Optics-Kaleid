"""Prescription, coordinates and sampling for the eight-body Trepan system."""

import numpy as np
from .assets import load_spec

SPEC = load_spec("trepan_prescription")
GROUPS = SPEC["groups"]
SCALES = np.array([0.1, 0.1, 0.1, 2.5 / 60, 2.5 / 60])
FIELDS = np.asarray(SPEC["fields"])


def disk(grid):
    y, x = np.mgrid[-1 : 1 : complex(grid), -1 : 1 : complex(grid)]
    mask = x * x + y * y <= 1
    return np.c_[x[mask], y[mask]]


def rotation(q):
    tx, ty = np.deg2rad(q)
    cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
    return np.array([[cy, sy * sx, sy * cx], [0, cx, -sx], [-sy, cy * sx, cy * cx]])
