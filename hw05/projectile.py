"""Explicit-Euler solution of the hw05 projectile assignment.

Run: python hw05/projectile.py
Dependencies: numpy, scipy, matplotlib.
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

TARGET = np.array([1000.0, 100.0])
# A strict open hit region has an infimum, not an attained minimum.
# Use a 1 mm interior margin to report a reproducibly feasible shot.
HIT_RADIUS = 2.0
SEARCH_RADIUS = HIT_RADIUS - 0.001


class Projectile:
    """Common parameters and the single shared forward-Euler integrator."""

    def __init__(self, mass=10.0, g=9.8, dt=0.01):
        if mass <= 0 or g <= 0 or dt <= 0:
            raise ValueError("mass, g and dt must be positive")
        self.mass, self.g, self.dt = mass, g, dt

    def acceleration(self, y, vx, vy):
        return 0.0, -self.g

    def simulate(self, speed, angle_deg, stop_x=None, max_time=300.0):
        """Return rows (t, x, y, vx, vy), including the first ground crossing.

        Hits are checked at Euler nodes, not on interpolated line segments.
        stop_x is an optional optimization-only early stopping position.
        """
        theta = math.radians(angle_deg)
        x = y = 0.0
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        rows = [(0.0, x, y, vx, vy)]
        for step in range(1, math.ceil(max_time / self.dt) + 1):
            ax, ay = self.acceleration(y, vx, vy)
            # All right-hand sides use the OLD state (explicit Euler).
            x, y, vx, vy = (x + vx * self.dt, y + vy * self.dt,
                            vx + ax * self.dt, vy + ay * self.dt)
            rows.append((step * self.dt, x, y, vx, vy))
            if y < 0 or (stop_x is not None and x > stop_x):
                return np.asarray(rows)
        raise RuntimeError("Trajectory did not terminate within max_time")


class AdiabaticDragProjectile(Projectile):
    """Textbook adiabatic atmosphere and altitude-dependent quadratic drag."""

    def __init__(self, mass=10.0, g=9.8, dt=0.01, b2_over_m=4e-5,
                 a=6.5e-3, alpha=2.5, t0=300.0, rho0=1.225):
        super().__init__(mass, g, dt)
        self.b2 = b2_over_m * mass
        self.a, self.alpha, self.t0, self.rho0 = a, alpha, t0, rho0

    def density_ratio(self, y):
        return max(1.0 - self.a * y / self.t0, 0.0) ** self.alpha

    def density(self, y):
        # rho0 is arbitrary for this assignment: only rho/rho0 enters drag.
        return self.rho0 * self.density_ratio(y)

    def acceleration(self, y, vx, vy):
        coefficient = self.b2 / self.mass * self.density_ratio(y)
        drag = coefficient * math.hypot(vx, vy)
        return -drag * vx, -self.g - drag * vy


class ConstantDensityProjectile(AdiabaticDragProjectile):
    """Control model used only for the density comparison."""

    def density_ratio(self, y):
        return 1.0


def closest_node(projectile, parameters):
    speed, angle = parameters
    trajectory = projectile.simulate(speed, angle, stop_x=TARGET[0] + 3.0)
    distances2 = np.sum((trajectory[:, 1:3] - TARGET) ** 2, axis=1)
    index = int(np.argmin(distances2))
    return trajectory, index, float(distances2[index])


def optimize_launch(projectile, radius=SEARCH_RADIUS):
    """Bounded multistart constrained search, then polish nearby Euler nodes.

    Coarse starts span the full angular interval. Minimizing distance to the
    discrete trajectory is piecewise smooth; fixed-node polishing checks the
    neighboring time branches to avoid a nearest-node switching artifact.
    This is a numerical search, not a formal global-optimality proof.
    """
    bounds = [(100.0, 1000.0), (10.0, 80.0)]
    def constraint(p):
        return radius ** 2 - closest_node(projectile, p)[2]

    candidates = []
    for angle in np.linspace(10, 80, 15):
        # Vacuum target estimate is an inexpensive starting point, not a solver.
        theta = math.radians(angle)
        denominator = 2 * math.cos(theta) ** 2 * (1000 * math.tan(theta) - 100)
        estimate = math.sqrt(projectile.g * 1000 ** 2 / denominator)
        result = minimize(lambda p: p[0], [estimate * 1.03, angle],
                          method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": constraint}],
                          options={"ftol": 1e-10, "maxiter": 150})
        if result.success and constraint(result.x) >= -1e-7:
            candidates.append(result.x)
    if not candidates:
        raise RuntimeError("No feasible optimum found")
    best = min(candidates, key=lambda p: (p[0], p[1]))
    trajectory, index, _ = closest_node(projectile, best)
    # Integrate through the requested node using the SAME inherited integrator.
    for node in range(max(1, index - 5), index + 6):
        def fixed_constraint(p):
            path = projectile.simulate(*p)
            if node >= len(path):
                return -1e12
            return radius ** 2 - float(np.sum((path[node, 1:3] - TARGET) ** 2))
        result = minimize(lambda p: p[0], best, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": fixed_constraint}],
                          options={"ftol": 1e-10, "maxiter": 100})
        if result.success and fixed_constraint(result.x) >= -1e-7:
            candidates.append(result.x)
    # Treat speeds within numerical solver precision as ties; prefer lower angle.
    minimum = min(p[0] for p in candidates)
    return min((p for p in candidates if p[0] <= minimum + 1e-8),
               key=lambda p: p[1])


def main():
    directory = Path(__file__).resolve().parent
    projectile = AdiabaticDragProjectile()
    parameters = optimize_launch(projectile)
    trajectory = projectile.simulate(*parameters)
    distances = np.linalg.norm(trajectory[:, 1:3] - TARGET, axis=1)
    closest = int(np.argmin(distances))
    hits = np.flatnonzero(distances < HIT_RADIUS)
    if not len(hits):
        raise RuntimeError("Optimized shot failed the strict hit check")
    first_hit = int(hits[0])
    constant_parameters = optimize_launch(ConstantDensityProjectile())
    vacuum_parameters = optimize_launch(Projectile())
    result = {
        "angle_deg": float(parameters[1]), "speed_m_s": float(parameters[0]),
        "first_hit_time_s": float(trajectory[first_hit, 0]),
        "closest_time_s": float(trajectory[closest, 0]),
        "closest_distance_m": float(distances[closest]),
        "closest_position_m": trajectory[closest, 1:3].tolist(),
        "maximum_height_m": float(trajectory[:, 2].max()),
        "maximum_height_before_hit_m": float(trajectory[:first_hit + 1, 2].max()),
        "first_below_ground_time_s": float(trajectory[-1, 0]),
        "density_ratio_at_apex": projectile.density_ratio(trajectory[:, 2].max()),
        "constant_density_speed_m_s": float(constant_parameters[0]),
        "constant_density_angle_deg": float(constant_parameters[1]),
        "vacuum_speed_m_s": float(vacuum_parameters[0]),
        "vacuum_angle_deg": float(vacuum_parameters[1]),
        "dt_s": projectile.dt, "search_radius_m": SEARCH_RADIUS,
    }
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savetxt(directory / "trajectory.csv", trajectory, delimiter=",",
               header="t_s,x_m,y_m,vx_m_s,vy_m_s", comments="")
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis in (ax, zoom):
        axis.plot(trajectory[:, 1], trajectory[:, 2], label="Adiabatic drag (Euler)")
        axis.scatter(*TARGET, marker="x", color="red", label="Target", zorder=5)
        axis.add_patch(plt.Circle(TARGET, HIT_RADIUS, fill=False, color="red", linestyle="--"))
        axis.scatter(*trajectory[closest, 1:3], s=25, color="green", label="Closest Euler node", zorder=6)
        axis.set(xlabel="x (m)", ylabel="y (m)")
        axis.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_title(f"v0 = {parameters[0]:.6f} m/s, angle = {parameters[1]:.6f} deg")
    ax.legend(fontsize=8)
    zoom.set(xlim=(995, 1005), ylim=(95, 105), title="Target region (2 m tolerance)")
    zoom.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(directory / "trajectory.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
