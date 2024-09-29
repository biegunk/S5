import jax
from flax import linen as nn
from jax import numpy as jnp

parallel_scan = jax.lax.associative_scan


class LRU(nn.Module):
    """
    Complex number implementation as in the original LRU paper
    """

    lru_dim: int  # N: state dimension
    hidden_dim: int  # H: model dimension
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = jnp.pi / 10
    bidirectional: bool = False
    step_rescale: float = 1.0
    activation: str = "gelu"

    @staticmethod
    def theta_init(rng_key, lru_dim, max_phase=6.28):
        # print("Max_phase: ", max_phase)
        subkeys = jax.random.split(rng_key, num=3)
        u1 = jax.random.uniform(subkeys[0], shape=(lru_dim,))
        theta_log = jnp.log(max_phase * u1)
        return theta_log

    @staticmethod
    def nu_init(rng_key, lru_dim, r_min=0, r_max=1):
        # print("R_min: ", r_min, "r_max: ",  r_max)
        subkeys = jax.random.split(rng_key, num=3)
        u1 = jax.random.uniform(subkeys[0], shape=(lru_dim,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        return nu_log

    @staticmethod
    def b_init(rng_key, lru_dim, hidden_dim):
        subkeys = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkeys[0], shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            2 * hidden_dim
        )
        return B

    @staticmethod
    def c_init(rng_key, lru_dim, hidden_dim):
        subkeys = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        C = jax.random.normal(subkeys[0], shape=(hidden_dim, lru_dim)) / jnp.sqrt(
            lru_dim
        )
        return C

    @staticmethod
    def binary_operator_diag(element_i, element_j):
        """Binary operator for parallel scan of linear recurrence."""
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j

    def mix_sequence(self, Lambda_elements, Bu_elements, reverse=False):
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(
            self.binary_operator_diag, elements, reverse=reverse
        )  # all x_k  # LBN
        # inner_states = inner_states.transpose(1, 0, 2)  # LBN -> BLN
        return inner_states

    @nn.compact
    def __call__(self, input_sequence):
        """
        Args:
            x: jnp.array(TD)
        """
        theta_log = self.param(
            "theta_log",
            self.theta_init,
            self.lru_dim,
            self.max_phase,
        )
        nu_log = self.param(
            "nu_log",
            self.nu_init,
            self.lru_dim,
            self.r_min,
            self.r_max,
        )
        B_re = self.param("B_re", self.b_init, self.lru_dim, self.hidden_dim)
        B_im = self.param("B_im", self.b_init, self.lru_dim, self.hidden_dim)
        C_re = self.param("C_re", self.c_init, self.lru_dim, self.hidden_dim)
        C_im = self.param("C_im", self.c_init, self.lru_dim, self.hidden_dim)
        if self.bidirectional:
            C_re2 = self.param("C_re2", self.c_init, self.lru_dim, self.hidden_dim)
            C_im2 = self.param("C_im2", self.c_init, self.lru_dim, self.hidden_dim)

        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.hidden_dim
        )

        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(Lambda) ** 2))
        B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
        C = C_re + 1j * C_im

        x = input_sequence[:, 0, :, 0]  # Remove singleton dimensions
        x = x.transpose(0, 2, 1)  # Now, (bsz, L, H)

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[1], axis=0)
        # Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Lambda_elements = Lambda_elements.transpose(1, 0, 2)  # BLN -> LBN

        Bu_elements = jax.vmap(lambda u: u @ B_norm.T)(x)  # LBN
        # Bu_elements = jnp.einsum("LH,NH->LN", x, B_norm)

        # print(Lambda_elements.shape, Bu_elements.shape)
        inner_states = self.mix_sequence(Lambda_elements, Bu_elements, reverse=False)
        if self.bidirectional:
            C2 = C_re2 + 1j * C_im2
            C = jnp.concatenate([C, C2], axis=-1)
            backward = self.mix_sequence(Lambda_elements, Bu_elements, reverse=True)
            inner_states = jnp.concatenate(
                [inner_states, backward], axis=-1
            )  # BLN -> BL2N

        # y = jnp.einsum("HN,LN->LH", C, inner_states).real + jnp.einsum("H,LH->LH", D, x)
        y = jax.vmap(lambda x, u: (x @ C.T).real + D * u)(inner_states, x).transpose(
            1, 0, 2
        )  # LBH -> BLH

        # Define activations
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.hidden_dim)
            self.out2 = nn.Dense(self.hidden_dim)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.hidden_dim)

        if self.activation in ["full_glu"]:
            y = nn.activation.gelu(y, approximate=False)
            y = self.out1(y) * jax.nn.sigmoid(self.out2(y))
        elif self.activation in ["half_glu1"]:
            y = nn.activation.gelu(y, approximate=False)
            y = y * jax.nn.sigmoid(self.out2(y))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.activation.gelu(y, approximate=False)
            y = y * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            y = nn.activation.gelu(y, approximate=False)
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation)
            )

        y = jnp.expand_dims(y.transpose(0, 2, 1), (1, 3))  # Now, (bsz,1,H,1,L)

        return y
