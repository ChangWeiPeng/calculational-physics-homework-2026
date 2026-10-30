import unittest
import numpy as np
from pendulum import wrap, step, poincare, box_counts, PERIOD


class TestPendulum(unittest.TestCase):
    def test_wrap(self):
        for angle in [-100., -np.pi, 0., np.pi, 100.]:
            self.assertTrue(-np.pi <= wrap(angle) < np.pi)
            self.assertAlmostEqual(np.sin(angle), np.sin(wrap(angle)), places=13)

    def test_cromer_order(self):
        theta, omega, t, dt = .2, .3, .7, .04
        expected = omega + (-np.sin(theta)-.5*omega+1.2*np.sin(2*t/3))*dt
        a, b = step(theta, omega, t, dt)
        self.assertAlmostEqual(b, expected)
        self.assertAlmostEqual(a, theta+expected*dt)

    def test_sampling_against_independent_reference(self):
        # Start near the branch cut and cross it before the first sample.
        dt = .04
        theta, omega = 3.13, 2.
        times, angles, velocities = [0.], [theta], [omega]
        for i in range(int(np.ceil(3*PERIOD/dt))):
            omega += (-np.sin(theta)-.5*omega+1.2*np.sin(2*i*dt/3))*dt
            theta += omega*dt  # independent reference stays unwrapped throughout
            times.append((i+1)*dt); angles.append(theta); velocities.append(omega)
        targets = np.arange(1, 4)*PERIOD
        expected = np.column_stack(((np.interp(targets, times, angles)+np.pi)%(2*np.pi)-np.pi,
                                    np.interp(targets, times, velocities)))
        np.testing.assert_allclose(poincare(3.13, 2., dt, 3, 0), expected, atol=1e-11)

    def test_discard_and_bounds(self):
        full = poincare(.2, 0., .04, 12, 0)
        short = poincare(.2, 0., .04, 2, 10)
        np.testing.assert_allclose(short, full[10:], atol=1e-12)
        self.assertTrue(np.all(np.abs(full[:, 0]) <= np.pi))
        self.assertTrue(np.all(np.isfinite(full)))

    def test_boxes(self):
        points = np.array([[.1,.1], [.2,.2], [1.1,.1], [1.1,1.1]])
        np.testing.assert_array_equal(box_counts(points, [2., 1., .1], np.zeros(2)), [1,3,4])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            poincare(0., 0., 0., 10)


if __name__ == '__main__':
    unittest.main()
