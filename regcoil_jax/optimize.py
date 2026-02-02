from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdamState:
    step: jnp.ndarray
    m: jnp.ndarray
    v: jnp.ndarray

    def tree_flatten(self):
        return (self.step, self.m, self.v), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        step, m, v = children
        return cls(step=step, m=m, v=v)


def adam_init(x: jnp.ndarray) -> AdamState:
    x = jnp.asarray(x)
    return AdamState(step=jnp.asarray(0, dtype=jnp.int32), m=jnp.zeros_like(x), v=jnp.zeros_like(x))


def adam_step(
    *,
    x: jnp.ndarray,
    state: AdamState,
    grad: jnp.ndarray,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, AdamState]:
    """One Adam update (JAX-friendly)."""
    step = state.step + jnp.asarray(1, dtype=jnp.int32)
    m = b1 * state.m + (1.0 - b1) * grad
    v = b2 * state.v + (1.0 - b2) * (grad * grad)
    mhat = m / (1.0 - (b1**step))
    vhat = v / (1.0 - (b2**step))
    x_new = x - lr * mhat / (jnp.sqrt(vhat) + eps)
    return x_new, AdamState(step=step, m=m, v=v)


@dataclass(frozen=True)
class OptimizeResult:
    x: Any
    loss_history: Any


def minimize_adam(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    steps: int,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    jit: bool = True,
) -> OptimizeResult:
    """Minimize a scalar objective f(x) with Adam, returning x and loss history."""
    x0 = jnp.asarray(x0)
    steps = int(steps)

    grad_f = jax.grad(lambda x: jnp.asarray(f(x)))

    def one_step(carry, _):
        x, state = carry
        g = grad_f(x)
        x, state = adam_step(x=x, state=state, grad=g, lr=lr, b1=b1, b2=b2, eps=eps)
        loss = f(x)
        return (x, state), loss

    state0 = adam_init(x0)
    def run(x_init):
        return jax.lax.scan(one_step, (x_init, adam_init(x_init)), xs=None, length=steps)

    if jit:
        run_compiled = jax.jit(run)
        (x_final, _), loss_hist = run_compiled(x0)
    else:
        (x_final, _), loss_hist = run(x0)

    return OptimizeResult(x=x_final, loss_history=loss_hist)
