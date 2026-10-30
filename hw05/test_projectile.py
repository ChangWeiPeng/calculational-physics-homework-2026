"""Run with: python -m unittest discover -s hw05 -v"""
import json
from pathlib import Path
import unittest

import numpy as np

from projectile import AdiabaticDragProjectile, Projectile, TARGET


class ProjectileTests(unittest.TestCase):
    def test_inheritance(self):
        self.assertIsInstance(AdiabaticDragProjectile(), Projectile)
        self.assertIs(AdiabaticDragProjectile.simulate, Projectile.simulate)
        self.assertIsNot(AdiabaticDragProjectile.acceleration, Projectile.acceleration)

    def test_density(self):
        p = AdiabaticDragProjectile()
        self.assertEqual(p.density_ratio(0), 1)
        self.assertAlmostEqual(p.density_ratio(1000), (1 - .0065 * 1000 / 300) ** 2.5)
        self.assertLess(p.density(1000), p.density(0))
        self.assertEqual(p.density_ratio(300 / .0065), 0)
        self.assertEqual(p.density_ratio(50000), 0)

    def test_acceleration(self):
        p = AdiabaticDragProjectile()
        ax, ay = p.acceleration(0, 30, 40)
        self.assertAlmostEqual(ax, -4e-5 * 50 * 30)
        self.assertAlmostEqual(ay, -9.8 - 4e-5 * 50 * 40)
        self.assertEqual(p.acceleration(50000, 30, 40), (0, -9.8))

    def test_explicit_euler(self):
        p = AdiabaticDragProjectile()
        path = p.simulate(110, 45)
        old, new = path[0], path[1]
        ax, ay = p.acceleration(old[2], old[3], old[4])
        np.testing.assert_allclose(new[1:], [old[1] + p.dt * old[3],
                                           old[2] + p.dt * old[4],
                                           old[3] + p.dt * ax,
                                           old[4] + p.dt * ay])

    def test_vacuum_euler_formula(self):
        p = Projectile()
        path = p.simulate(110, 45)
        n = np.arange(len(path))
        np.testing.assert_allclose(path[:, 1], path[0, 3] * n * p.dt, atol=1e-9)
        np.testing.assert_allclose(path[:, 2], path[0, 4] * n * p.dt
                                   - .5 * p.g * p.dt ** 2 * n * (n - 1), atol=1e-9)

    def test_saved_solution_hits_target(self):
        result = json.loads(Path(__file__).with_name("results.json").read_text())
        self.assertTrue(100 <= result["speed_m_s"] <= 1000)
        self.assertTrue(10 <= result["angle_deg"] <= 80)
        path = AdiabaticDragProjectile().simulate(result["speed_m_s"], result["angle_deg"])
        distance = np.linalg.norm(path[:, 1:3] - TARGET, axis=1)
        self.assertLess(float(distance.min()), 2)
        self.assertAlmostEqual(float(distance.min()), result["closest_distance_m"], places=8)
        self.assertLess(path[-1, 2], 0)


if __name__ == "__main__":
    unittest.main()
