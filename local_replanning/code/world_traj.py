import numpy as np

from .graph_search import graph_search
from ..util.occupancy_map import OccupancyMap


class WorldTraj(object):
    def __init__(self, world, start, goal, local=False):
        self.world = world
        self.resolution = np.array([0.25, 0.25, 0.25], dtype=float)
        self.margin = 0.55

        # Do not change this local-map interface.  The EC simulator expects the
        # map to be updated around the current vehicle position.
        self.local = local
        self.local_occ_map = OccupancyMap(world, self.resolution, self.margin)

        self.planning_horizon = 7.5
        self.recovery_horizon = 3.5
        self.stopping_distance = 0.5
        self.step_t = 0.02
        self.no_replan_thresh = 2.0
        self.collision_check_horizon = 150.0
        self.min_replan_period = 0.0
        self.tracking_replan_thresh = 4.0
        self.late_recovery_time = 60.0

        self.exec_traj = True
        self.replan_num = 0
        self.t_check_traj = 0.25

        self.global_goal = np.asarray(goal, dtype=float).reshape(3)
        self.start = np.asarray(start, dtype=float).reshape(3)
        self.yaw_target = self._camera_forward_yaw(self.start, self.global_goal)
        self.traj_start_time = 0.0
        self.last_replan_time = -np.inf
        self.traj_duration = 0.0
        self.local_goal = self.global_goal.copy()
        self.start_velocity = np.zeros(3)
        self._slow_traj = False
        self.path = np.vstack([self.start, self.global_goal])
        self.points = self.path.copy()
        self.seg_T = np.array([1.0])
        self.t_breaks = np.array([0.0, 1.0])
        self.coeffs = [self._quintic_from_boundary(
            self.start, np.zeros(3), np.zeros(3),
            self.global_goal, np.zeros(3), np.zeros(3), 1.0)]

        self.local_occ_map.update(self.start)
        self.plan_traj(self.start, self.crop_local_goal(self.start))

    def check_traj_collsion(self, cur_t):
        check_t = float(cur_t)
        stop_t = min(self.traj_start_time + self.traj_duration,
                     check_t + self.collision_check_horizon)
        while check_t < stop_t:
            check_t += self.step_t
            check_pt = self.get_traj_pos(check_t)
            if self.local_occ_map.is_occupied_metric(check_pt):
                return check_t
        return -1

    def get_traj_pos(self, t):
        return self.update(t)["x"]

    def replan(self, cur_state, t):
        cur_pos = np.asarray(cur_state["x"], dtype=float).reshape(3)
        cur_vel = np.asarray(cur_state.get("v", np.zeros(3)), dtype=float).reshape(3)
        t = float(t)

        self.local_occ_map.update(cur_pos)
        self.replan_num += 1

        collision_t = self.check_traj_collsion(t)
        collision_soon = collision_t >= 0 and collision_t <= t + 1.5
        tracking_error = np.linalg.norm(cur_pos - self.get_traj_pos(t))
        late_recovery = t >= self.late_recovery_time
        tracking_thresh = 2.5 if late_recovery else self.tracking_replan_thresh
        tracking_lost = tracking_error >= tracking_thresh
        moved_far = np.linalg.norm(cur_pos - self.start) >= self.no_replan_thresh
        near_segment_end = t >= self.traj_start_time + 0.75 * self.traj_duration
        near_local_goal = np.linalg.norm(cur_pos - self.local_goal) <= 1.0

        min_checks = 4 if late_recovery else 12
        if self.replan_num < min_checks and not (collision_soon or tracking_lost or moved_far or near_segment_end or near_local_goal):
            return
        urgent_collision = collision_t >= 0 and collision_t <= t + 0.75
        if (t - self.last_replan_time) < self.min_replan_period and not (urgent_collision or tracking_lost):
            return
        self.replan_num = 0

        if np.linalg.norm(cur_pos - self.global_goal) < self.stopping_distance:
            return

        old_start_velocity = self.start_velocity.copy()
        old_slow_traj = self._slow_traj
        if tracking_lost or late_recovery:
            self.start_velocity = self._limit_vector(cur_vel, 0.55)
        else:
            self.start_velocity = self._limit_vector(cur_vel, 1.2)

        self._slow_traj = bool(tracking_lost or late_recovery)
        horizon = self.recovery_horizon if self._slow_traj else self.planning_horizon
        if self.plan_traj(cur_pos, self.crop_local_goal(cur_pos, horizon=horizon)):
            self.start = cur_pos.copy()
            self.traj_start_time = t
            self.last_replan_time = t
            self.exec_traj = True
        else:
            self.start_velocity = old_start_velocity
            self._slow_traj = old_slow_traj
            if tracking_lost:
                self._set_brake_traj(cur_pos, cur_vel, t)

    def crop_local_goal(self, start, horizon=None):
        start = np.asarray(start, dtype=float).reshape(3)
        to_goal = self.global_goal - start
        dist = float(np.linalg.norm(to_goal))
        if dist < 1e-9:
            return self.global_goal.copy()

        if horizon is None:
            horizon = self.planning_horizon
        step = min(float(horizon), dist)
        direction = to_goal / dist
        goal = start + step * direction

        if step >= dist or not self.local_occ_map.is_occupied_metric(goal):
            return goal

        # If the straight-line cropped goal lies in an obstacle, back off along
        # the same ray.  Graph search also has a relaxed goal condition, so this
        # is only a first-pass local-goal improvement.
        for ratio in np.linspace(0.9, 0.25, 14):
            cand = start + step * ratio * direction
            if not self.local_occ_map.is_occupied_metric(cand):
                return cand
        return start + min(1.0, dist) * direction

    def plan_traj(self, start, goal):
        start = np.asarray(start, dtype=float).reshape(3)
        goal = np.asarray(goal, dtype=float).reshape(3)
        self.local_occ_map.update(start)

        path, _ = graph_search(self.local_occ_map, start, goal, astar=True)
        if path is None or len(path) < 2:
            if self._segment_is_free(start, goal, step=0.10):
                path = np.vstack([start, goal])
            else:
                return False

        self.path = self._remove_near_duplicates(path)
        self.points = self._shortcut_path(self.path, window=8)
        self.points[0] = start
        self.local_goal = self.points[-1].copy()

        seg_T = self._allocate_times(self.points)
        if len(seg_T) == 0:
            seg_T = np.array([0.5])
            self.points = np.vstack([start, goal])

        self.seg_T = seg_T
        self.t_breaks = np.concatenate([[0.0], np.cumsum(self.seg_T)])
        self.traj_duration = float(self.t_breaks[-1])

        v_wp, a_wp = self._compute_waypoint_derivatives(self.points, self.t_breaks)
        self.coeffs = []
        for k, T in enumerate(self.seg_T):
            self.coeffs.append(self._quintic_from_boundary(
                self.points[k], v_wp[k], a_wp[k],
                self.points[k + 1], v_wp[k + 1], a_wp[k + 1], float(T)))
        return True

    def _set_brake_traj(self, pos, vel, t, duration=1.2):
        pos = np.asarray(pos, dtype=float).reshape(3)
        v0 = self._limit_vector(np.asarray(vel, dtype=float).reshape(3), 1.4)
        zeros = np.zeros(3)
        end = pos + 0.5 * v0 * float(duration)

        for scale in (1.0, 0.65, 0.35, 0.0):
            cand = pos + scale * (end - pos)
            if self._segment_is_free(pos, cand, step=0.06):
                end = cand
                break

        self.start = pos.copy()
        self.start_velocity = v0.copy()
        self._slow_traj = True
        self.path = np.vstack([pos, end])
        self.points = self.path.copy()
        self.local_goal = end.copy()
        self.seg_T = np.array([float(duration)])
        self.t_breaks = np.array([0.0, float(duration)])
        self.traj_duration = float(duration)
        self.traj_start_time = float(t)
        self.last_replan_time = float(t)
        self.coeffs = [self._quintic_from_boundary(pos, v0, zeros, end, zeros, zeros, float(duration))]

    def update(self, t):
        if not np.isfinite(t):
            x = self.global_goal.copy()
            zeros = np.zeros(3)
            return {"x": x, "x_dot": zeros, "x_ddot": zeros, "x_dddot": zeros,
                    "x_ddddot": zeros, "yaw": self.yaw_target, "yaw_dot": 0.0}

        tau_global = float(t) - float(self.traj_start_time)
        if tau_global <= 0.0:
            k = 0
            tau = 0.0
        elif tau_global >= self.traj_duration:
            k = len(self.seg_T) - 1
            tau = float(self.seg_T[k])
        else:
            k = int(np.searchsorted(self.t_breaks, tau_global, side="right") - 1)
            k = int(np.clip(k, 0, len(self.seg_T) - 1))
            tau = float(tau_global - self.t_breaks[k])

        x, x_dot, x_ddot, x_dddot, x_ddddot = self._eval_quintic(self.coeffs[k], tau)
        return {"x": x, "x_dot": x_dot, "x_ddot": x_ddot, "x_dddot": x_dddot,
                "x_ddddot": x_ddddot, "yaw": self.yaw_target, "yaw_dot": 0.0}

    def _allocate_times(self, pts):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return np.zeros(0)

        seg_T = []
        turns = self._turn_severity(pts)
        for k in range(len(pts) - 1):
            dist = float(np.linalg.norm(pts[k + 1] - pts[k]))
            v_nom = 1.05 if self._slow_traj else 1.62
            T = max(0.30, dist / v_nom)
            if k == len(pts) - 2:
                T = max(T, 1.60 if self._slow_traj else 1.25)
            turn_k = 0.90 if self._slow_traj else 0.55
            T *= 1.0 + turn_k * max(turns[k], turns[k + 1])
            seg_T.append(T)
        return np.asarray(seg_T, dtype=float)

    def _compute_waypoint_derivatives(self, pts, t_breaks):
        n = len(pts)
        v = np.zeros((n, 3))
        a = np.zeros((n, 3))
        v[0] = self.start_velocity
        v[-1] = 0.0

        for i in range(1, n - 1):
            dt = float(t_breaks[i + 1] - t_breaks[i - 1])
            if dt <= 1e-9:
                continue
            vi = (pts[i + 1] - pts[i - 1]) / dt
            a0 = pts[i] - pts[i - 1]
            a1 = pts[i + 1] - pts[i]
            n0 = float(np.linalg.norm(a0))
            n1 = float(np.linalg.norm(a1))
            if n0 > 1e-9 and n1 > 1e-9:
                cosang = float(np.clip(np.dot(a0, a1) / (n0 * n1), -1.0, 1.0))
                if cosang < -0.25:
                    vi *= 0.0
                elif cosang < 0.35:
                    vi *= 0.35
                else:
                    vi *= 0.65
            max_v = 1.3 if self._slow_traj else 2.4
            v[i] = self._limit_vector(vi, max_v)
        return v, a

    def _shortcut_path(self, pts, window=10):
        pts = np.asarray(pts, dtype=float)
        if len(pts) <= 2:
            return pts.copy()
        out = [pts[0]]
        i = 0
        last = len(pts) - 1
        while i < last:
            j_best = i + 1
            for j in range(min(last, i + int(window)), i, -1):
                if self._segment_is_free(pts[i], pts[j], step=0.06):
                    j_best = j
                    break
            out.append(pts[j_best])
            i = j_best
        return self._remove_near_duplicates(np.vstack(out))

    @staticmethod
    def _path_length(pts):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _segment_is_free(self, p0, p1, step=0.08):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        n = max(1, int(np.ceil(L / max(step, 1e-6))))
        for i in range(n + 1):
            p = p0 + (i / n) * d
            if self.local_occ_map.is_occupied_metric(p):
                return False
        return True

    @staticmethod
    def _camera_forward_yaw(start, goal):
        h = np.asarray(goal[:2] - start[:2], dtype=float)
        if float(np.linalg.norm(h)) < 1e-9:
            return 0.0
        return float(np.arctan2(h[1], h[0]))

    @staticmethod
    def _remove_near_duplicates(pts, eps=1e-8):
        pts = np.asarray(pts, dtype=float)
        out = [pts[0]]
        for i in range(1, len(pts)):
            if float(np.linalg.norm(pts[i] - out[-1])) > eps:
                out.append(pts[i])
        return np.vstack(out)

    @staticmethod
    def _turn_severity(pts):
        sev = np.zeros(len(pts))
        for i in range(1, len(pts) - 1):
            a = pts[i] - pts[i - 1]
            b = pts[i + 1] - pts[i]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-9 or nb <= 1e-9:
                continue
            cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            sev[i] = 1.0 - cosang
        return sev

    @staticmethod
    def _limit_vector(v, max_norm):
        n = float(np.linalg.norm(v))
        if n <= max_norm or n <= 1e-12:
            return v
        return v * (float(max_norm) / n)

    @staticmethod
    def _basis(d, t):
        if d == 0:
            return np.array([1.0, t, t ** 2, t ** 3, t ** 4, t ** 5])
        if d == 1:
            return np.array([0.0, 1.0, 2.0 * t, 3.0 * t ** 2, 4.0 * t ** 3, 5.0 * t ** 4])
        if d == 2:
            return np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t ** 2, 20.0 * t ** 3])
        if d == 3:
            return np.array([0.0, 0.0, 0.0, 6.0, 24.0 * t, 60.0 * t ** 2])
        return np.array([0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * t])

    def _quintic_from_boundary(self, p0, v0, a0, p1, v1, a1, T):
        T = max(float(T), 1e-6)
        A = np.vstack([
            self._basis(0, 0.0),
            self._basis(1, 0.0),
            self._basis(2, 0.0),
            self._basis(0, T),
            self._basis(1, T),
            self._basis(2, T),
        ])
        coeff = np.zeros((3, 6))
        for axis in range(3):
            b = np.array([p0[axis], v0[axis], a0[axis], p1[axis], v1[axis], a1[axis]], dtype=float)
            coeff[axis, :] = np.linalg.solve(A, b)
        return coeff

    def _eval_quintic(self, coeff, tau):
        x = coeff @ self._basis(0, tau)
        x_dot = coeff @ self._basis(1, tau)
        x_ddot = coeff @ self._basis(2, tau)
        x_dddot = coeff @ self._basis(3, tau)
        x_ddddot = coeff @ self._basis(4, tau)
        return x, x_dot, x_ddot, x_dddot, x_ddddot
