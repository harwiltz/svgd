import jax
import jax.experimental.optimizers as optim
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import Optional, Callable, Tuple, List, Union

Particle = Union[float, List[float]]
Kernel = Callable[[Particle, Particle], float]
ParticleInitializer = Callable[[int], List[Particle]]
Probability = float

@jax.jit
def rbf(scale: float, x: Particle, y: Particle) -> float:
    return jnp.exp(-(x - y).dot(x - y)/ (2 * scale))

# @jax.jit
def gaussian_init_fun(key, loc: Particle, scale: float, n_atoms: int) -> List[Particle]:
    loc = jnp.array(loc)
    scale = jnp.eye(loc.shape[0]) * scale
    keys = [key]
    for _ in range(n_atoms - 1):
        key, sub = jax.random.split(key)
        keys.append(sub)
    return jnp.array([jax.random.multivariate_normal(k, loc, scale) for k in keys])

default_rbf_kernel = partial(rbf, 2)
default_gaussian_init_fun = partial(gaussian_init_fun,
                                    jax.random.PRNGKey(0),
                                    [0, 0],
                                    1)

# TODO Make subclass of jax.experimental.optimizers.Optimizer
class svgd():
    def __init__(self,
                 n_atoms: int,
                 lr: Optional[float] = 1e-1,
                 kernel: Optional[Kernel] = default_rbf_kernel,
                 init_fun: Optional[ParticleInitializer] = default_gaussian_init_fun):
        self.particles = init_fun(n_atoms)
        self.n_atoms = n_atoms
        self.kernel = kernel
        self.lr = lr

    def update(self, target: Callable[[Particle], Probability]):
        grads = svgd_grads(self.particles, self.kernel, target)
        self.particles = self.particles + self.lr * grads

# @jax.jit
def svgd_grads(particles: List[Particle],
               kernel: Kernel,
               target: Callable[[Particle], Probability]) -> List[Particle]:
    # [N, d]
    logprobgrads = jax.vmap(jax.grad(lambda x: jnp.log(target(x))))(particles)
    kernel_matrix_fn = jax.vmap(jax.vmap(kernel, in_axes=(0, None)), in_axes=(None, 0))
    # [N, N]
    kernel_matrix = kernel_matrix_fn(particles, particles)
    # [N, d]
    kernel_grads = jax.vmap(jax.grad(lambda x, y: jax.vmap(kernel, in_axes=(None, 0))(x, y).sum()), in_axes=(0, None))(particles, particles)
    grad_matrix = (kernel_matrix @ logprobgrads) - kernel_grads
    return grad_matrix / len(particles)
