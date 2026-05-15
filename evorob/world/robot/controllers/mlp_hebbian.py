import numpy as np

from evorob.world.robot.controllers.base import Controller


class HebbianNumpyNetwork:

    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.size_l1 = n_input * n_hidden
        self.size_l2 = n_hidden * n_output
        self.total_weights = self.size_l1 + self.size_l2

        self.lr = 0.01

        self.A = np.zeros(self.total_weights)
        self.B = np.zeros(self.total_weights)
        self.C = np.zeros(self.total_weights)
        self.D = np.zeros(self.total_weights)

        rng = np.random.default_rng(42)
        self.rescale_weights = 0.1
        self.lin1_init = rng.uniform(-self.rescale_weights, self.rescale_weights, (n_hidden, n_input)).astype(np.float32)
        self.output_init = rng.uniform(-self.rescale_weights, self.rescale_weights, (n_output, n_hidden)).astype(np.float32)

        self.lin1: np.ndarray
        self.output: np.ndarray

    def set_hebbian_rules(self, abcd: np.ndarray) -> None:
        abcd = np.array(abcd, dtype=np.float32).reshape(4, self.total_weights)
        self.A = abcd[0, :]
        self.B = abcd[1, :]
        self.C = abcd[2, :]
        self.D = abcd[3, :]

        self.A1 = self.A[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.B1 = self.B[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.C1 = self.C[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.D1 = self.D[:self.size_l1].reshape(self.n_hidden, self.n_input)

        self.A2 = self.A[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.B2 = self.B[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.C2 = self.C[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.D2 = self.D[self.size_l1:].reshape(self.n_output, self.n_hidden)

    def reset_weights(self, batch_size=1):
        self.lin1 = np.tile(self.lin1_init, (batch_size, 1, 1))
        self.output = np.tile(self.output_init, (batch_size, 1, 1))

    def forward(self, state: np.ndarray):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        # Keep all arithmetic in float32; MuJoCo observations arrive as float64.
        state = state.astype(np.float32, copy=False)

        # Forward pass — matmul is faster than einsum for batched matrix-vector products.
        hid_l    = np.tanh(np.matmul(self.lin1, state[..., None]).squeeze(-1))
        output_l = np.tanh(np.matmul(self.output, hid_l[..., None]).squeeze(-1))

        # Hebbian weight updates
        outer_1 = np.einsum('bh,bi->bhi', hid_l, state)
        self.lin1 += self.lr * (
            self.A1 * outer_1 +
            self.B1 * state[:, None, :] +
            self.C1 * hid_l[:, :, None] +
            self.D1
        )
        np.clip(self.lin1, -5.0, 5.0, out=self.lin1)

        outer_2 = np.einsum('bo,bh->boh', output_l, hid_l)
        self.output += self.lr * (
            self.A2 * outer_2 +
            self.B2 * hid_l[:, None, :] +
            self.C2 * output_l[:, :, None] +
            self.D2
        )
        np.clip(self.output, -5.0, 5.0, out=self.output)

        return output_l


class HebbianController(Controller):

    def __init__(self, input_size, output_size, hidden_size,):
        self.controller_type = "Hebbian"
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size
        self.model = HebbianNumpyNetwork(input_size, hidden_size, output_size)
        self.n_params = (self.model.size_l1 + self.model.size_l2) * 4

    def geno2pheno(self, genotype: np.ndarray) -> None:
        self.model.set_hebbian_rules(genotype)

    def reset_controller(self, batch_size=1) -> None:
        self.model.reset_weights(batch_size)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = self.model.forward(state)
        return action
