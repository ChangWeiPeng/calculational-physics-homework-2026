"""Exercise 1.7: uncached properties and forward Euler radioactive decay."""

from pathlib import Path
import numpy as np


class Decay:
    """Uniform full Euler steps; omit a fractional final step if necessary."""

    def __init__(self, N0, tau, dt, t_max):
        self.N0 = N0
        self.tau = tau
        self.dt = dt
        self.t_max = t_max

    @staticmethod
    def _positive(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Parameter must be finite and positive")
        return value

    @property
    def N0(self):
        return self._N0

    @N0.setter
    def N0(self, value):
        self._N0 = self._positive(value)

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = self._positive(value)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = self._positive(value)

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("t_max must be finite and nonnegative")
        self._t_max = value

    @property
    def t(self):
        ratio = self.t_max / self.dt
        nearest = round(ratio)
        # Recover endpoints lost solely to floating-point division roundoff.
        if abs(ratio - nearest) <= 8 * np.finfo(float).eps * max(1, ratio):
            ratio = nearest
        return np.arange(int(np.floor(ratio)) + 1, dtype=float) * self.dt

    @staticmethod
    def euler_step(n, dt, tau):
        return n - n * dt / tau

    @property
    def numerical(self):
        values = np.empty(self.t.size)
        values[0] = self.N0
        for k in range(1, values.size):
            values[k] = self.euler_step(values[k - 1], self.dt, self.tau)
        return values

    @property
    def exact(self):
        return self.N0 * np.exp(-self.t / self.tau)

    @property
    def error(self):
        return self.numerical - self.exact

    @property
    def max_abs_error(self):
        return float(np.max(np.abs(self.error)))

    @property
    def half_life(self):
        return float(self.tau * np.log(2))

    @classmethod
    def from_half_life(cls, N0, t_half, dt, t_max):
        return cls(N0, cls._positive(t_half) / np.log(2), dt, t_max)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker

    candidates = ["Microsoft YaHei", "SimHei", "Source Han Sans SC",
                  "Noto Sans CJK SC", "PingFang SC", "Heiti SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in candidates if name in installed]
    if not fonts:
        raise RuntimeError("Install a Chinese font, e.g. Noto Sans CJK SC, to render labels")
    plt.rcParams["font.sans-serif"] = fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    d = Decay(100, 1, 0.2, 5)
    dts = np.array([0.5, 0.25, 0.2, 0.1, 0.05, 0.025])
    errors = np.array([Decay(100, 1, dt, 5).max_abs_error for dt in dts])
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    axes[0].plot(d.t, d.exact, label="解析解")
    axes[0].scatter(d.t, d.numerical, s=20, label="Euler 数值解", color="tab:orange")
    axes[0].set(xlabel="时间 t/s", ylabel="剩余核数 N（个）", title="放射性衰变：Δt = 0.2 s")
    axes[0].legend()
    axes[1].plot(d.t, d.error, "o-", markersize=3)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set(xlabel="时间 t/s", ylabel="误差 Nnum − Nexact（个）",
                title=f"最大绝对误差 = {d.max_abs_error:.4f}")
    axes[2].loglog(dts, errors, "o-", label="Euler 最大绝对误差")
    axes[2].loglog(dts, errors[-1] * dts / dts[-1], "--", label="斜率 1 参考线")
    axes[2].set(xlabel="时间步长 Δt/s", ylabel="最大绝对误差（个）",
                title=f"log-log 拟合斜率 = {slope:.4f}")
    axes[2].set_xticks([0.025, 0.05, 0.1, 0.25, 0.5])
    axes[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axes[2].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    output = Path(__file__).with_name("decay_oop.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)

    lines = [f"half_life = {d.half_life:.8f} s",
             f"N(5): Euler = {d.numerical[-1]:.8f}, exact = {d.exact[-1]:.8f}",
             f"dt = 0.2: max_abs_error = {d.max_abs_error:.8f}"]
    d.dt = 0.05
    lines.append(f"after dt = 0.05: max_abs_error = {d.max_abs_error:.8f}")
    lines.extend(f"dt = {dt:g}: max_abs_error = {err:.8f}" for dt, err in zip(dts, errors))
    lines.extend([f"error ratio 0.5/0.25 = {errors[0]/errors[1]:.8f}",
                  f"error ratio 0.05/0.025 = {errors[-2]/errors[-1]:.8f}",
                  f"log-log slope = {slope:.8f}",
                  f"unstable dt = 2.5: N(25) = {Decay(100, 1, 2.5, 25).numerical[-1]:.8f}"])
    text = "\n".join(lines) + "\n"
    Path(__file__).with_name("results.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
