import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy

n = 100
x = np.linspace(0, 1, n)
y = x.reshape(-1, 1)
grid = y-x
grid = grid*(grid > 0)

lam = 5
growth_matrix = lam*np.exp(-1*lam*grid)*(grid > 0)

fig, ax = plt.subplots()
ax.imshow(growth_matrix)
ax.set_ylim(-0.5, n-0.5)
ax.set_xticks([0, n*0.25, n*0.50, n*0.75, n], [0, 0.25, 0.50, 0.75, 1], fontsize=14)
ax.set_yticks([0, n*0.25, n*0.50, n*0.75, n], [0, 0.25, 0.50, 0.75, 1], fontsize=14)
ax.set_xlabel('Age at time $t$', fontsize=14)
ax.set_ylabel('Age at time $t + 1$', fontsize=14)
plt.show()


pop = scipy.stats.norm.pdf(x, loc=0.3, scale=0.1)

fig, ax = plt.subplots(figsize=(3, 1))

ax.plot(x, pop)
ax.set_xticks([0, 0.25, 0.50, 0.75, 1], [0, 0.25, 0.50, 0.75, 1], fontsize=14)
ax.set_yticks([])

plt.show()

pop = growth_matrix @ pop

fig, ax = plt.subplots(figsize=(1,3))

ax.plot(pop, x)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1], [0, 0.25, 0.50, 0.75, 1], fontsize=14)
ax.set_xticks([])

plt.show()