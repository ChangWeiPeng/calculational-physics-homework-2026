import numpy as np
import matplotlib.pyplot as plt

N0 = 1
lam = 0.5

t = np.linspace(0, 10, 200)

N = N0 * np.exp(-lam * t)

plt.figure(figsize=(8, 5))
plt.plot(t, N, label=r"$N(t)=N_0e^{-\lambda t}$", linewidth=2)

plt.xlabel("Time $t$")
plt.ylabel("N(t)")

plt.title("Exponential Decay")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("exponential_decay.png", dpi=300, bbox_inches="tight")

plt.show()