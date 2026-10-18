"""Conservative 3D glass-envelope and whole straight-command-path clearance.

Surfaces retain catalog signs; no gap edits, gap projection or state clipping.
Unknown lens shoulders/housings are explicitly outside this model.
"""

from .trepan_config import SPEC, GROUPS, SCALES, FIELDS, disk, rotation, np

MARGIN = 0.001
SURF = {s["sid"]: s for s in SPEC["surface"]}


def sag(r, x):
    x = np.asarray(x)
    return (
        np.zeros_like(x)
        if not np.isfinite(r)
        else x * x / (r + np.copysign(np.sqrt(r * r - x * x), r))
    )


class Geometry:
    def __init__(self):
        self.pivots = np.array(
            [[0, 0, (SURF[a]["z"] + SURF[b]["z"]) / 2] for a, b in GROUPS]
        )
        self.outer = np.array(
            [
                max(
                    s["sd"]
                    for s in SPEC["surface"]
                    if a <= s["sid"] <= b and np.isfinite(s["sd"])
                )
                for a, b in GROUPS
            ]
        )
        self.faces = []
        self.extent = []
        for gi, (a, b) in enumerate(GROUPS):
            faces = []
            height = 0
            for sid in (a, b):
                s = SURF[sid]
                cap = min(s["sd"], self.outer[gi])
                dz = s["z"] - self.pivots[gi, 2]
                faces.append((s["r"], cap, self.outer[gi], dz))
                height = max(height, abs(dz), abs(dz + float(sag(s["r"], cap))))
            self.faces.append(faces)
            self.extent.append(np.hypot(self.outer[gi], height))
        self.extent = np.array(self.extent)

    def poses(self, q):
        v = np.asarray(q).reshape(8, 5) * SCALES
        return self.pivots + v[:, :3], np.array([rotation(x[3:]) for x in v])

    def support(self, center, rot, face):
        r, cap, outer, dz = face
        h = np.linalg.norm(rot[2, :2])
        a = rot[2, 2]
        rad = [0.0, cap]
        if np.isfinite(r):
            stationary = h * abs(r) / np.hypot(a, h)
            if stationary < cap:
                rad.append(stationary)
        rad = np.array(rad)
        z = a * (dz + sag(r, rad))
        ann = a * (dz + float(sag(r, cap)))
        vals = np.r_[z - h * rad, z + h * rad, ann - h * outer, ann + h * outer]
        return center[2] + np.array([vals.min(), vals.max()])

    def bounds(self, q):
        q = np.asarray(q)
        v = q.reshape(8, 5) * SCALES
        if not np.isfinite(q).all() or abs(v[:, 3:]).max() > 1:
            return np.full(29, -1e6)
        centers, rots = self.poses(q)
        front = []
        back = []
        for i in range(8):
            front.append(self.support(centers[i], rots[i], self.faces[i][0])[0])
            back.append(self.support(centers[i], rots[i], self.faces[i][1])[1])
        bounds = np.array(front[1:]) - back[:-1]
        # S80/S81 nested positive spheres: globally valid lower-hemisphere bound.
        i, j = 5, 6
        sl, sr = SURF[80], SURF[81]
        rl, rr = sl["r"], sr["r"]
        dr = rl - rr
        cl = centers[i] + rots[i] @ np.array([0, 0, sl["z"] - self.pivots[i, 2] + rl])
        cr = centers[j] + rots[j] @ np.array([0, 0, sr["z"] - self.pivots[j, 2] + rr])
        dc = cr - cl
        rho = np.linalg.norm(dc[:2])
        bounds[5] = dc[2] + np.sqrt(max(dr * dr - rho * rho, 0)) if rho < dr else -1e6
        left_vertex = centers[i] + rots[i] @ np.array(
            [0, 0, sl["z"] - self.pivots[i, 2]]
        )
        local = rots[i].T @ (centers[j] - left_vertex)
        rel = rots[i].T @ rots[j]
        domain = (
            sl["sd"]
            - np.linalg.norm(local[:2])
            - self.outer[j]
            - self.extent[j] * np.linalg.norm(rel[:2, 2])
        )
        other = [front[j] - back[i] for i in range(8) for j in range(i + 2, 8)]
        return np.r_[bounds, domain, other]

    def state(self, q):
        return bool(np.min(self.bounds(q)) >= MARGIN)

    def path(self, end, start=None):
        start = np.zeros(40) if start is None else np.asarray(start)
        end = np.asarray(end)
        d = (end - start).reshape(8, 5) * SCALES
        ang = np.sum(abs(np.deg2rad(d[:, 3:])), axis=1)
        speed = abs(d[:, 2]) + self.extent * ang
        lips = list(speed[:-1] + speed[1:])
        # Domain-specific conservative sphere-center speed bound; covers endpoints
        # produced by +/-1 commands on +/-1 samples, without treating this as travel.
        cv = []
        for gi, sid in ((5, 80), (6, 81)):
            distance = abs(SURF[sid]["z"] + SURF[sid]["r"] - self.pivots[gi, 2])
            cv.append(np.linalg.norm(d[gi, :3]) + distance * ang[gi])
        if max(abs(start).max(), abs(end).max()) > 2.000001:
            return False
        lips[5] = 2 * sum(
            cv
        )  # rho<1 mm, dr>41 mm within the registered endpoint domain
        lips.append(
            np.linalg.norm(d[6, :3] - d[5, :3])
            + (50 + self.extent[6]) * (ang[5] + ang[6])
        )
        lips.extend(speed[i] + speed[j] for i in range(8) for j in range(i + 2, 8))
        lips = np.asarray(lips)
        stack = [(0.0, 1.0, 0)]
        while stack:
            a, b, depth = stack.pop()
            m = (a + b) / 2
            values = self.bounds(start + m * (end - start))
            if values.min() < MARGIN:
                return False
            if np.all(values - lips * (b - a) / 2 >= MARGIN):
                continue
            if depth >= 12:
                return False
            stack.extend([(a, m, depth + 1), (m, b, depth + 1)])
        return self.state(start) and self.state(end)
