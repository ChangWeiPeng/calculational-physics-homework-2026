"""Run with python -m unittest discover -s hw04 -v."""
import unittest
import numpy as np
from decay_oop import Decay


class TestDecay(unittest.TestCase):
    def test_reference_values(self):
        d = Decay(100, 1, 0.2, 5)
        np.testing.assert_allclose(d.half_life, 0.6931, rtol=0.005)
        np.testing.assert_allclose(d.numerical[-1], 0.3778, rtol=0.005)
        np.testing.assert_allclose(d.exact[-1], 0.6738, rtol=0.005)
        np.testing.assert_allclose(d.max_abs_error, 4.0199, rtol=0.005)
        self.assertAlmostEqual(d.t[np.argmax(abs(d.error))], 1)
        self.assertEqual(d.error[0], 0)
        self.assertTrue(np.all(d.error[1:] < 0))
        d.dt = 0.05
        np.testing.assert_allclose(d.max_abs_error, 0.9394, rtol=0.005)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Decay(100, -1, 0.2, 5)
        d = Decay(100, 1, 0.2, 5)
        for name in ('N0', 'tau', 'dt'):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    setattr(d, name, value)
        for value in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                d.t_max = value

    def test_grid_and_recurrence(self):
        for dt, end in ((0.2, 5), (0.3, 1), (0.1, 0.3), (2, 1), (0.2, 0)):
            d = Decay(100, 1, dt, end)
            np.testing.assert_allclose(d.t, np.arange(d.t.size) * dt)
            self.assertLessEqual(d.t[-1], end + 1e-14)
            self.assertGreater(d.t[-1] + dt, end)
            np.testing.assert_allclose(d.numerical, 100 * (1 - dt) ** np.arange(d.t.size))
            np.testing.assert_allclose(d.error, d.numerical - d.exact)
        self.assertEqual(Decay.euler_step(100, 0.2, 1), 80)

    def test_uncached_properties(self):
        d = Decay(100, 1, 0.2, 5)
        a = d.numerical
        a[0] = -1
        self.assertEqual(d.numerical[0], 100)
        for name, value in (("N0", 200), ("tau", 2), ("dt", 0.05), ("t_max", 3)):
            setattr(d, name, value)
            fresh = Decay(d.N0, d.tau, d.dt, d.t_max)
            np.testing.assert_allclose(d.numerical, fresh.numerical)
            np.testing.assert_allclose(d.exact, fresh.exact)
            self.assertEqual(d.max_abs_error, fresh.max_abs_error)

    def test_half_life_factory(self):
        d = Decay.from_half_life(100, np.log(2), 0.2, 5)
        self.assertAlmostEqual(d.tau, 1)
        rounded = Decay.from_half_life(100, 0.6931, 0.2, 5)
        np.testing.assert_allclose(rounded.max_abs_error, d.max_abs_error, rtol=0.0001)
        class SubDecay(Decay):
            pass
        self.assertIsInstance(SubDecay.from_half_life(100, 1, 0.2, 5), SubDecay)

    def test_convergence_and_stability(self):
        dts = np.array([0.5, 0.25, 0.2, 0.1, 0.05, 0.025])
        errors = np.array([Decay(100, 1, dt, 5).max_abs_error for dt in dts])
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        self.assertLess(abs(slope - 1), 0.1)
        np.testing.assert_allclose(errors[0] / errors[1], 2.29, rtol=0.005)
        np.testing.assert_allclose(errors[-2] / errors[-1], 2.02, rtol=0.005)
        unstable = Decay(100, 1, 2.5, 25).numerical
        self.assertTrue(np.all(unstable[1:] * unstable[:-1] < 0))
        self.assertGreater(abs(unstable[-1]), 1000)
        np.testing.assert_allclose(abs(Decay(100, 1, 2, 10).numerical), 100)


if __name__ == '__main__':
    unittest.main()
