import numpy as np


class SimplePerceptron(object):

    def __init__(self, eta: float, w0: np.ndarray):
        self.eta: float = eta
        self.w: np.ndarray = w0

    def train(self, data: np.ndarray):

        # out for this neuron
        y = np.dot(data, self.w)

        # calculate the delta w (oja algorithm)
        delta_w = self.eta * y * (data - y * self.w)

        # update w
        self.w += delta_w

    def get_w(self) -> np.ndarray:
        return self.w

    def get_normalized_w(self) -> np.ndarray:
        return np.divide(self.w, np.sqrt(np.dot(self.w, self.w)))

    def __str__(self) -> str:
        return f"SP=(w={self.w})"

    def __repr__(self) -> str:
        return f"SP=(w={self.w})"
