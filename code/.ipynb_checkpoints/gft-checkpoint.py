import numpy as np
import networkx as nx

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

    def signal_gft(self, signal: np.array):

        gft_signal =  np.zeros([self.n,1])

        for i in range(0, self.n):
          gft_signal[i,0] = np.dot((signal).T,self.U[:,i])
        
        return  gft_signal
    
    def signal_igft(self, signal: np.array , gft_signal: np.array, components_low_filter:int = 2):

        if components_low_filter > self.n:
          raise Exception("components to filter should be less than number of nodes")

        igft_signal =  np.zeros([self.n,1])

        for i in range(0, components_low_filter):
          igft_signal[:,0] += gft_signal[i]*self.U[:,i]
        
        return  igft_signal

    def low_filter_graph_signal(self, graph_with_signal: nx.Graph , components_low_filter:int = 2):
       G= graph_with_signal

       signal = np.array([atrib["signal"] for node, atrib in G.nodes(data=True)])

       signal_gft = self.signal_gft(signal)

       filtered_signal = self.signal_igft(signal, signal_gft, components_low_filter=3)

       G_filtered = G.copy()
       for idx, (node, atrib) in enumerate(G_filtered.nodes(data=True)):
          G_filtered.nodes[node]["signal"] = filtered_signal[idx]

       return filtered_signal, G_filtered