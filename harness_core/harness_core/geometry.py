"""Small rigid-body math helpers (numpy only). Quaternions are (w, x, y, z)."""

from __future__ import annotations

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotvec_from_mat(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (log map) of a rotation matrix, robust near 0 and pi."""
    R = np.asarray(R, dtype=float)
    cos_a = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-7:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - angle < 1e-4:
        # near pi: use the diagonal to find the axis
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        i = int(np.argmax(axis))
        if axis[i] > 1e-9:
            for j in range(3):
                if j != i:
                    axis[j] = np.copysign(axis[j], R[i, j] + R[j, i])
        axis /= np.linalg.norm(axis)
        return axis * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (angle / (2.0 * np.sin(angle)))


def mat_from_rotvec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def tool_down_rotation(yaw: float) -> np.ndarray:
    """Tool frame pointing straight down (z_tool = -z_world) with its x axis at `yaw`.

    Columns are the tool axes expressed in the world frame.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def tool_yaw(R: np.ndarray) -> float:
    """Yaw of the tool x axis projected on the world xy plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def polyline_arclength(points: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_point_at(points: np.ndarray, s_query: float):
    """Point and unit tangent at arc length `s_query` along a polyline."""
    s = polyline_arclength(points)
    s_query = float(np.clip(s_query, 0.0, s[-1]))
    i = int(np.searchsorted(s, s_query, side="right") - 1)
    i = min(max(i, 0), len(points) - 2)
    seg_len = max(s[i + 1] - s[i], 1e-12)
    t = (s_query - s[i]) / seg_len
    p = points[i] + t * (points[i + 1] - points[i])
    tangent = (points[i + 1] - points[i]) / seg_len
    return p, tangent


def closest_point_on_polyline(points: np.ndarray, q: np.ndarray):
    """Closest point to q on a polyline: (point, arc length, distance, segment index)."""
    s = polyline_arclength(points)
    best = (None, 0.0, np.inf, 0)
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-16 else float(np.clip((q - a) @ ab / L2, 0.0, 1.0))
        p = a + t * ab
        dist = float(np.linalg.norm(q - p))
        if dist < best[2]:
            best = (p, s[i] + t * np.sqrt(L2), dist, i)
    return best
