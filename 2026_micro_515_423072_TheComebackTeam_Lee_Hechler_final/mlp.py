import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16,
                 hidden_size_2: int = 0):
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size
        self.n_hidden_2 = hidden_size_2

        self.input_to_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.n_params_i2h = input_size * hidden_size

        if hidden_size_2:
            self.hidden_to_hidden2 = np.random.uniform(-1, 1, (hidden_size_2, hidden_size))
            self.hidden2_to_output = np.random.uniform(-1, 1, (output_size, hidden_size_2))
            self.n_params_h2h2 = hidden_size * hidden_size_2
            self.n_params_h2o = hidden_size_2 * output_size
        else:
            self.hidden_to_hidden2 = None
            self.hidden2_to_output = None
            self.n_params_h2h2 = 0
            self.n_params_h2o = hidden_size * output_size
            self.hidden_to_output = np.random.uniform(-1, 1, (output_size, hidden_size))

        self.n_params = self.get_num_params()

    def get_action(self, state):
        hidden = np.tanh(state @ self.input_to_hidden.T)
        if self.n_hidden_2:
            hidden2 = np.tanh(hidden @ self.hidden_to_hidden2.T)
            output = np.tanh(hidden2 @ self.hidden2_to_output.T)
        else:
            output = np.tanh(hidden @ self.hidden_to_output.T)
        return np.clip(output, -1, 1)

    def set_weights(self, encoding):
        end_i2h = self.n_params_i2h
        self.input_to_hidden = encoding[:end_i2h].reshape(self.n_hidden, self.n_input)
        if self.n_hidden_2:
            end_h2h2 = end_i2h + self.n_params_h2h2
            self.hidden_to_hidden2 = encoding[end_i2h:end_h2h2].reshape(
                self.n_hidden_2, self.n_hidden)
            self.hidden2_to_output = encoding[end_h2h2:].reshape(
                self.n_output, self.n_hidden_2)
        else:
            self.hidden_to_output = encoding[end_i2h:].reshape(self.n_output, self.n_hidden)

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):
        return self.n_params_i2h + self.n_params_h2h2 + self.n_params_h2o

    def reset_controller(self, batch_size=1) -> None:
        pass
