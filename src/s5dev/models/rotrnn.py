import jax
from flax import linen as nn
from jax import numpy as jnp

parallel_scan = jax.lax.associative_scan


class RotRNN(nn.Module):
    lru_dim: int = 64  # devisible by heads
    hidden_dim: int = 128  # devisible by heads
    nheads: int = 64  # apply model in parallel
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = 6.28
    bidirectional: bool = False
    step_rescale: float = 0.0

    def theta_init(self, rng_key, N, max_phase):
        return jnp.log(jax.random.uniform(rng_key, shape=(N, 1), maxval=max_phase))

    def nu_log_init(self, rng_key, H, r_min=0, r_max=1):
        """
        r_min, r_max in (0, 1)
        """
        subkey, rng_key = jax.random.split(rng_key)
        u1 = jax.random.uniform(subkey, shape=(H,))
        # double exponential
        nu_log = jnp.log(-jnp.log(r_max)) + u1 * (
            jnp.log(-jnp.log(r_min)) - jnp.log(-jnp.log(r_max))
        )
        return nu_log

    def mat_init(self, rng_key, lru_dim, hidden_dim):
        subkey, rng_key = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkey, shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            hidden_dim + lru_dim
        )
        return B

    def ortho_mat_init(self, rng_key, lru_dim, hidden_dim):
        subkey, rng_key = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkey, shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            hidden_dim + lru_dim
        )
        return B

    @staticmethod
    def mix_sequence(gamma, R, Us, reverse=False):
        """
        N - per head dimension
        Args:
            gammas: jax.Array(T,)
            As: jax.Array(T,N,N)
            Us: jax.array(T,B,N)
        Returns:
            out: jax.array(T,B,N)
        """

        def binf(a, b):
            gamma_i, thetas_i, acc_i = a
            gamma_j, thetas_j, acc_j = b
            # R_j@acc_i + acc_j
            # get [-x2, x1, -x4, x3,...]
            rotate_half_mat_i = jnp.stack(
                [-acc_i[..., 1::2], acc_i[..., 0::2]], axis=-1
            )
            shapes = list(rotate_half_mat_i.shape)[:-1]
            shapes[-1] *= 2
            rotate_half_mat_i = rotate_half_mat_i.reshape(shapes)
            # duplicate theta [o1, o1, o2, o2,...]
            shapes = list(thetas_j.shape)
            shapes[-1] *= 2
            theta = jnp.repeat(thetas_j[..., None], repeats=2, axis=-1).reshape(
                tuple(shapes)
            )
            sin = jnp.sin(theta)[..., None, :]  # add mock batch dimension
            cos = jnp.cos(theta)[..., None, :]  # add mock batch dimension
            acc = gamma_j[..., None, None] * (cos * acc_i + sin * rotate_half_mat_i)

            return (gamma_i * gamma_j, thetas_i + thetas_j, acc + acc_j)

        T = Us.shape[0]
        gammas = jnp.repeat(gamma[None, ...], repeats=T, axis=0)
        R = jnp.repeat(R[None, ...], repeats=T, axis=0)
        _, _, res = parallel_scan(binf, (gammas, R, Us), reverse=reverse)
        return res

    @nn.compact
    def __call__(self, input_sequence):
        x = input_sequence[:, 0, :, 0]  # Remove singleton dimensions
        x = x.transpose(0, 2, 1)  # Now, (bsz, L, H)
        # add dummy batch dimension for code
        # x = input_sequence[None, ...]
        # print("Input: ", x.shape)
        # naming shortcut
        H, N = self.nheads, self.lru_dim // self.nheads
        assert N % 2 == 0, "N should be even"
        batch_sz, T, D = x.shape
        # log might not be necessary for theta
        thetas = self.param(
            "theta_log", self.theta_init, self.lru_dim // 2, self.max_phase
        ).reshape(H, N // 2)
        P = self.param("P", self.ortho_mat_init, self.lru_dim, N).reshape(H, N, N)
        B = self.param("B", self.mat_init, self.lru_dim, self.hidden_dim).reshape(
            H, N, self.hidden_dim
        )
        C = self.param("C", self.mat_init, self.hidden_dim, self.lru_dim)
        if self.bidirectional:
            C2 = self.param("C2", self.mat_init, self.hidden_dim, self.lru_dim)

        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.hidden_dim
        )
        gamma_log = self.param(
            "gamma_log", self.nu_log_init, H, r_min=self.r_min, r_max=self.r_max
        )

        # do not forget the double exponential
        gamma_log = -jnp.exp(gamma_log)
        gamma = jnp.exp(gamma_log)

        trace_per_head = jnp.trace(jnp.einsum("HDd,HAd->HDA", B, B), axis1=-2, axis2=-1)
        norm = jnp.sqrt((1 - gamma**2) / trace_per_head)  #  H / H elementwise -> H
        B_norm = jnp.einsum("H,HnD->HnD", norm, B)
        P = jax.scipy.linalg.expm(P - P.transpose(0, 2, 1))
        # apply P.T to Bx_t
        Us = jnp.einsum("HnD,BTD->HTBn", B_norm, x)
        Us = jnp.einsum("HnN,HTBn->HTBN", P.transpose(0, 2, 1), Us)
        # mix per head
        mix_head_fn = jax.vmap(self.mix_sequence, in_axes=(0, 0, 0, None), out_axes=0)
        thetas = jnp.exp(thetas)

        y = mix_head_fn(gamma, thetas, Us, False)  # H T B N
        # multiply P back to \tilde{x}_t
        y = jnp.einsum("HNn,HTBN->HTBn", P, y)

        if self.bidirectional:
            backward = mix_head_fn(gamma, thetas, Us, True)  # H T B N
            # multiply P back to \tilde{x}_t
            backward = jnp.einsum("HNn,HTBN->HTBn", P, backward)
            y = jnp.concatenate([y, backward], axis=-1)
            C = jnp.concatenate([C, C2], axis=-1)

        y = y.transpose(2, 1, 0, 3)  # H T B N -> B T H N
        y = jnp.einsum("Dn,BTn->BTD", C, y.reshape(batch_sz, T, -1)) + D * x
        # squeeze batch dimension
        return y[0]
