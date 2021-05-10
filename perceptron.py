import numpy as np


class SimplePerceptron(object):

    def __init__(self, w0: np.ndarray, accum_func):
        self.w: np.ndarray = w0
        self.acc_w: np.ndarray = np.zeros(len(self.w))
        self.accum_func = accum_func

    def train(self, data: np.ndarray, eta: float):



        # update accumulative w
        self.acc_w += self.accum_func(data, self.w, eta)

    def update_w(self):
        self.w += self.acc_w
        self.acc_w = np.zeros(len(self.w))

    def get_w(self) -> np.ndarray:
        return self.w

    def get_normalized_w(self) -> np.ndarray:
        return np.divide(self.w, np.sqrt(np.dot(self.w, self.w)))

    def __str__(self) -> str:
        return f"SP=(w={self.w})"

    def __repr__(self) -> str:
        return f"SP=(w={self.w})"
