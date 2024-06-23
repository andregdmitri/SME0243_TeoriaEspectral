import networkx as nx
import numpy as np


def laplacian(g: nx.Graph):
    return nx.laplacian_matrix(g).toarray()


class GFT:
    def __init__(self, graph: nx.Graph):

        self.L = laplacian(graph)
        self.eigenvalues, self.U = np.linalg.eigh(self.L)
        sorted_indices = np.argsort(self.eigenvalues)

        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.U = self.U[:, sorted_indices]
        self.n = self.U.shape[0]

    def gft(self, signal, components=None):
        gft_signal = np.zeros([self.n, 1])

        for i in range(0, self.n):
            gft_signal[i, 0] = np.dot((signal).T, self.U[:, i])

        if components is not None:
            gft_signal = gft_signal[:components]

        return gft_signal

    def igft(self, gft_signal: np.array):
        igft_signal = np.zeros([self.n, 1])

        for i in range(0, gft_signal.shape[0]):
            igft_signal[:, 0] += gft_signal[i] * self.U[:, i]

        return igft_signal[:, 0]
