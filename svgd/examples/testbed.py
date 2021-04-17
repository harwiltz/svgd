from svgd import svgd

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax.scipy.stats.multivariate_normal import pdf
from matplotlib.animation import FuncAnimation

def randloc():
    return 4 * np.random.uniform() - 2

def randscale():
    return 3 * np.random.uniform() + 0.2

loc1 = jnp.array([randloc(), randloc()]).astype(jnp.float32)
loc2 = jnp.array([randloc(), randloc()]).astype(jnp.float32)
scale1 = jnp.diag(jnp.array([randscale(), randscale()]).astype(jnp.float32))
scale2 = jnp.diag(jnp.array([randscale(), randscale()]).astype(jnp.float32))
ratios = np.random.uniform(size=2)
coeffs = ratios / ratios.sum()
contour_xs = jnp.linspace(-3, 3, 75)
contour_ys = jnp.linspace(-3, 3, 75)
density = lambda x: coeffs[0] * pdf(x, loc1, scale1) + coeffs[1] * pdf(x, loc2, scale2)
print(f"Loc1: {loc1}\nScale1: {scale1}")
print(f"Loc2: {loc2}\nScale2: {scale2}")
print(coeffs)

zs = np.zeros((len(contour_xs), len(contour_ys)))
for (i, x) in enumerate(contour_xs):
    points = jnp.array([[x, y] for y in contour_ys])
    zs[i]= density(points)

def update(model):
    plt.clf()
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    model.update(density)
    show(model)

def main():
    plt.style.use('seaborn-ticks')

    model = svgd.svgd(101, lr=0.5)

    fig = plt.figure()

    ani = FuncAnimation(fig, lambda i: update(model), interval=300)
    plt.show()

def show(model):
    plt.contour(contour_xs, contour_ys, zs.T, levels=20, alpha=0.4)
    plt.scatter(model.particles[:, 0], model.particles[:, 1], alpha=0.7)

if __name__ == "__main__":
    main()
