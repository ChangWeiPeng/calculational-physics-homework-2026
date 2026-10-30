"""Forced damped pendulum: python pendulum.py (all outputs in results/)."""
from pathlib import Path
import json
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PERIOD = 3 * np.pi
OUT = Path(__file__).resolve().parent / 'results'

@njit(cache=True)
def wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

@njit(cache=True)
def step(theta, omega, t, dt):
    new_omega = omega + (-np.sin(theta) - 0.5 * omega + 1.2 * np.sin(2 * t / 3)) * dt
    # Return the unwrapped endpoint so interpolation cannot cross the branch cut.
    return theta + new_omega * dt, new_omega

@njit(cache=True)
def poincare(theta0, omega0, dt, count, discard=300):
    """Fixed dt Euler-Cromer; sample (discard+1 ... discard+count)*T.

    Interpolate the unwrapped step, then wrap the sampled angle. Physical
    integration ends at the first grid point at or after the last sample.
    """
    if dt <= 0 or count < 1 or discard < 0:
        raise ValueError('Invalid integration parameters')
    points = np.empty((count, 2))
    theta, omega = wrap(theta0), omega0
    i, k = 0, 0
    while k < count:
        t = i * dt
        raw_theta, new_omega = step(theta, omega, t, dt)
        end = (i + 1) * dt
        target = (discard + k + 1) * PERIOD
        while target <= end and k < count:
            alpha = (target - t) / dt
            points[k, 0] = wrap(theta + alpha * (raw_theta - theta))
            points[k, 1] = omega + alpha * (new_omega - omega)
            k += 1
            target = (discard + k + 1) * PERIOD
        theta, omega = wrap(raw_theta), new_omega
        i += 1
    return points


def select(points, window):
    x0, x1, y0, y1 = window
    return points[(points[:, 0] > x0) & (points[:, 0] < x1) &
                  (points[:, 1] > y0) & (points[:, 1] < y1)]


def box_counts(points, epsilons, origin):
    return np.array([len(np.unique(np.floor((points - origin) / e).astype(np.int64), axis=0))
                     for e in epsilons])


def scatter(ax, points, title, window=None):
    ax.scatter(points[:, 0], points[:, 1], s=0.3, linewidths=0, rasterized=True)
    ax.set(xlabel=r'$\theta$ (rad)', ylabel=r'$\omega$ (rad / time)', title=title)
    if window:
        ax.set_xlim(window[:2]); ax.set_ylim(window[2:])
    else:
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-2.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.15)


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=240, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    initials = [(0.2, 0.), (0.8, 0.), (-1., 0.5)]
    data = {}
    runs = []
    for label, ic, dt, n in [*(('ic' + str(i), ic, .04, 10000) for i, ic in enumerate(initials)),
                              ('dt02', initials[0], .02, 10000),
                              ('dt01', initials[0], .01, 50000)]:
        print(f'{label}: dt={dt}, samples={n}', flush=True)
        p = poincare(*ic, dt, n)
        data[label] = p
        np.savetxt(OUT / (label + '.csv'), p, delimiter=',', header='theta,omega', comments='')
        runs.append(dict(label=label, initial=ic, dt=dt, samples=n, discard_periods=300,
                         last_sample_time=(300+n)*PERIOD,
                         integration_end_time=np.ceil((300+n)*PERIOD/dt)*dt))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        scatter(ax, data['ic'+str(i)], f'IC = {initials[i]}, dt = 0.04')
    save(fig, '01_initial_conditions')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, dt in zip(axes, ['ic0', 'dt02', 'dt01'], [.04, .02, .01]):
        scatter(ax, data[key][:10000], f'dt = {dt}, N = 10000')
    save(fig, '02_time_steps')
    window = (2.0, 2.7, -1.4, -0.7)
    # Choose a populated nested window with fixed, explicit coordinates.
    subwindow = (2.14, 2.38, -1.32, -1.12)
    fine = data['dt01']
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout='constrained')
    for ax, title, w in zip(axes, ['Global', 'Zoom 1', 'Zoom 2'], [None, window, subwindow]):
        scatter(ax, fine, title, w)
    save(fig, '03_magnification')
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, key, dt in zip(axes, ['ic0', 'dt02', 'dt01'], [.04, .02, .01]):
        scatter(ax, data[key][:10000], f'dt = {dt}', window)
    save(fig, '04_local_time_steps')
    local = select(fine, window)
    eps = .4 / 2 ** np.arange(9)
    counts = box_counts(local, eps, np.array([window[0], window[2]]))
    half_counts = box_counts(select(fine[:25000], window), eps, np.array([window[0], window[2]]))
    # Intermediate scales, excluding coarse geometry and fine sample saturation.
    fit_indices = np.arange(2, 6)
    x, y = np.log(1 / eps), np.log(counts)
    slope, intercept = np.polyfit(x[fit_indices], y[fit_indices], 1)
    half_slope = np.polyfit(x[fit_indices], np.log(half_counts[fit_indices]), 1)[0]
    residual = y[fit_indices] - (slope*x[fit_indices] + intercept)
    r2 = 1 - np.sum(residual**2) / np.sum((y[fit_indices]-y[fit_indices].mean())**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, 'o-', label=f'50000 global points ({len(local)} local)')
    ax.plot(x, np.log(half_counts), 's--', label='25000 global points')
    ax.plot(x[fit_indices], slope*x[fit_indices]+intercept, 'k-', label=f'Fit: D = {slope:.3f}')
    ax.set(xlabel=r'$\ln(1/\epsilon)$', ylabel=r'$\ln N(\epsilon)$', title='Local box counting (finite-sample estimate)')
    ax.legend(); ax.grid(alpha=.2)
    save(fig, '05_box_counting')
    np.savetxt(OUT / 'box_counts.csv', np.column_stack([eps, counts, half_counts]),
               delimiter=',', header='epsilon,N_50000,N_25000', comments='')
    summary = dict(parameters=dict(g=9.8, length=9.8, q=.5, drive=1.2, drive_frequency=2/3),
                   runs=runs, window=window, subwindow=subwindow, local_count=len(local),
                   nested_count=len(select(fine, subwindow)), epsilons=eps.tolist(),
                   counts=counts.tolist(), half_counts=half_counts.tolist(), fit_indices=fit_indices.tolist(),
                   dimension=float(slope), half_dimension=float(half_slope), r_squared=float(r2))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
