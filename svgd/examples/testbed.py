from svgd import svgd

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax.scipy.stats.multivariate_normal import pdf

# locs = [jnp.array(x).astype(jnp.float32) for x in[[-4, -1], [2, 0], [0.5, 2.5]]]
# scales = [jnp.diag(jnp.array(x)).astype(jnp.float32)
#           for x in [[1, 1], [0.2, 2], [1.5, 0.8]]]
# densities = [(lambda x: pdf(x, l, s)) for (l, s) in zip(locs, scales)]
# ratios = np.random.uniform(size=(len(densities,)))
# coeffs = ratios / ratios.sum()
# density = lambda x: sum([c * d(x) for (c, d) in zip(coeffs, densities)])

loc1 = jnp.array([-2, -1]).astype(jnp.float32)
loc2 = jnp.array([0, 1]).astype(jnp.float32)
scale1 = jnp.diag(jnp.array([1, 1]).astype(jnp.float32))
scale2 = jnp.diag(jnp.array([1.5, 0.7]).astype(jnp.float32))
ratios = np.random.uniform(size=2)
coeffs = ratios / ratios.sum()
contour_xs = jnp.linspace(-4, 3, 50)
contour_ys = jnp.linspace(-2, 2, 50)
density = lambda x: coeffs[0] * pdf(x, loc1, scale1) + coeffs[1] * pdf(x, loc2, scale2)
print(coeffs)

zs = np.zeros((len(contour_xs), len(contour_ys)))
grid = np.meshgrid(contour_xs, contour_ys)
for (i, x) in enumerate(contour_xs):
    print(i)
    for (j, y) in enumerate(contour_ys):
        zs[i][j] = density(np.array([x, y]))

def main():
    plt.style.use('seaborn')

    model = svgd.svgd(11)

    while True:
        show(model)
        n_updates = int(input("How many updates should be done? "))
        for _ in range(n_updates):
            model.update(density)

def show(model):
    plt.contour(contour_xs, contour_ys, zs, levels=20, alpha=0.4)
    plt.scatter(model.particles[:, 0], model.particles[:, 1], alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
