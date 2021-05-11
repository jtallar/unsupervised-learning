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
        return np.divide(self.w, np.linalg.norm(self.w, 2))

    def __str__(self) -> str:
        return f"SP=(w={self.w})"

    def __repr__(self) -> str:
        return f"SP=(w={self.w})"


class HopfieldPerceptron(object):

    def __init__(self, patterns: np.ndarray, query: np.ndarray):
        self.w = np.dot(patterns, patterns.T) / patterns.shape[0]
        np.fill_diagonal(self.w, 0)
        self.s = [query]

    # Finish if last two elements are equal
    def is_over(self) -> bool:
        return len(self.s) >= 2 and np.array_equal(self.s[-1], self.s[-2])

    def iterate(self) -> np.ndarray:
        self.s.append(np.sign(np.dot(self.w, self.s[-1])))
        return self.s[-1]
