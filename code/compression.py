# %%
import pickle

import gft
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def signal(g: nx.Graph):
    return np.array([atrib["signal"] for node, atrib in g.nodes(data=True)])


class Compressor:
    def __init__(self, graph: nx.Graph, components: int):
        self.engine = gft.GFT(graph)
        self.components = components
        self.pos = np.array(list(nx.spring_layout(graph, seed=1).values()))

        self.edges = []
        nodes = list(graph.nodes())
        for edge in graph.edges():
            self.edges.append((nodes.index(edge[0]), nodes.index(edge[1])))
        self.edges = np.array(self.edges)

    def recons(self, graph):
        sig = signal(graph)
        rec_sig = self.engine.igft(self.engine.gft(sig, self.components))
        return rec_sig

    def error_rate(self, all_graphs):
        errors = []
        signals = []

        for g in all_graphs:
            sig = signal(g)
            signals.append(sig)
            rec_sig = self.engine.igft(self.engine.gft(sig, self.components))
            error = np.mean(np.abs(sig - rec_sig))
            errors.append(error)

        return np.mean(errors) / np.std(signals)

    def compression_rate(self, all_graphs):
        nodes = all_graphs[0].number_of_nodes()
        siglen = len(all_graphs)

        bytes_orig = nodes * siglen * 4
        bytes_comp = (nodes * self.components + self.components * siglen) * 4

        return 1 - bytes_comp / bytes_orig

    def plot(self, graph: nx.Graph, sig: np.array):

        for edge in self.edges:
            plt.plot(
                [self.pos[edge[0]][0], self.pos[edge[1]][0]],
                [self.pos[edge[0]][1], self.pos[edge[1]][1]],
                color="black",
                alpha=0.5,
            )

        plt.scatter(
            self.pos[:, 0],
            self.pos[:, 1],
            c=sig,
            s=200,
            cmap="cividis",
        )

    def plot_component(self, graph: nx.Graph, component: int):
        sig = self.engine.U[:, component]
        self.plot(graph, sig)

    def plot_recons(self, graph):
        sig = signal(graph)
        rec_sig = self.engine.igft(self.engine.gft(sig, self.components))
        self.plot(graph, rec_sig)


def metrics(all_graphs):
    error_rates = []
    compression_rates = []

    for i in range(21):
        compressor = Compressor(all_graphs[0], i)
        error_rates.append(compressor.error_rate(all_graphs))
        compression_rates.append(compressor.compression_rate(all_graphs))

    plt.plot(compression_rates[::-1], error_rates[::-1])
    plt.xlabel("Compression Rate (%)")
    plt.ylabel("Error Rate")


def plot_recons(graph):
    plt.figure(figsize=(10, 5))

    compressor = Compressor(graph, 15)

    plt.subplot(1, 2, 1)
    compressor.plot(graph, signal(graph))

    plt.subplot(1, 2, 2)
    compressor.plot_recons(graph)


def history(all_graphs, components, node):
    compressor = Compressor(all_graphs[0], components)

    all_recons = []

    for graph in all_graphs:
        recons = compressor.recons(graph)
        all_recons.append(recons)

    all_recons = np.array(all_recons)

    return all_recons[:, node]


def lineplot(all_graphs, node, components=[10, 15, 20]):
    plt.figure(figsize=(12, 5))
    sigs = history(all_graphs, 20, node)

    for component in components:
        plt.plot(
            history(all_graphs, component, node),
            label=f"{component} components",
            linewidth=1,
        )
        plt.legend()


with open("../grafos/graph_list.pickle", "rb") as f:
    all_graphs = pickle.load(f)

metrics(all_graphs)
plt.savefig("../imagens/metrics.png")

plot_recons(all_graphs[30])
plt.savefig("../imagens/recons.png")

lineplot(all_graphs, 5)
plt.savefig("../imagens/lineplot.png")
