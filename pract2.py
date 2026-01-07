import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Part 1: Coin Toss Simulation
# ------------------------------------
n = 1000
coin = np.random.choice(["H", "T"], n)
heads = np.sum(coin == "H")
tails = np.sum(coin == "T")

plt.bar(["Heads", "Tails"], [heads, tails])
plt.title("Coin Toss Frequency (1000 trials)")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()


# ------------------------------------
# First Year of MTech- R25-MCO-PCC-105: Laboratory Practice I
# Experiment No: 9
# ------------------------------------

# ------------------------------------
# Part 2: Dice Roll Simulation
# ------------------------------------
dice = np.random.randint(1, 7, n)
values, counts = np.unique(dice, return_counts=True)
probabilities = counts / n

plt.bar(values, counts)
plt.title("Dice Roll Distribution (1000 trials)")
plt.xlabel("Outcome")
plt.ylabel("Frequency")
plt.show()

print("Dice Probabilities:", dict(zip(values, probabilities)))


# ------------------------------------
# Part 3: Normal Distribution
# ------------------------------------
mu = 0
sigma = 1
normal_data = np.random.normal(mu, sigma, n)

plt.hist(normal_data, bins=30, density=True)
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# ------------------------------------
# Part 4: Poisson Distribution
# ------------------------------------
lam = 4
poisson_data = np.random.poisson(lam, n)

plt.hist(poisson_data, bins=range(0, 15), align='left', rwidth=0.8)
plt.title("Poisson Distribution (λ=4)")
plt.xlabel("k")
plt.ylabel("Frequency")
plt.show()
