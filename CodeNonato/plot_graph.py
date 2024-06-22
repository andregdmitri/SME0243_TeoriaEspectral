import numpy as np
import matplotlib.pyplot as plt

def plot_graph(A, p, pn, save_path='../imagens'):
    n, n = A.shape
    plt.figure(pn)
    plt.scatter(p[:, 0], p[:, 1])
    for i in range(0, n-1):
        for j in range(i+1, n):
            if (A[i, j] != 0):
                plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="blue")
    if save_path:
        plt.savefig(f'{save_path}/figure_{pn}.png')

def plot_graph_s(A, p, cm, pn, save_path='../imagens'):
    n, n = A.shape
    plt.figure(pn)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if (A[i, j] != 0):
                plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="black", linewidth=0.5)
    sc1 = plt.scatter(p[:, 0], p[:, 1], c=cm, s=50, edgecolor='none', alpha=1.0, cmap='terrain')
    plt.colorbar(sc1)
    if save_path:
        plt.savefig(f'{save_path}/figure_{pn}.png')

def plot_graph_h(A, p, cm, save_path='../imagens'):
    cm_max = np.amax(cm)
    cm_min = np.amin(cm)
    xmax = np.amax(p[:, 0])
    xmin = np.amin(p[:, 0])
    ymax = np.amax(p[:, 1])
    ymin = np.amin(p[:, 1])
    dx = xmax - xmin
    dy = ymax - ymin
    s = np.sqrt(dx*dx + dy*dy)
    n, n = A.shape
    plt.figure(1)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if (A[i, j] != 0):
                plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="black", linewidth=0.5)
    plt.scatter(p[:, 0], p[:, 1], s=0.8, edgecolor='none')
    for i in range(0, n):
        hcm = s * ((cm[i] - cm_min) / (cm_max - cm_min)) - s / 2
        plt.plot([p[i, 0], p[i, 0]], [p[i, 1], (p[i, 1] + hcm)], color="blue")
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/figure_{1}.png')

def plot_specf(x, f, pn, save_path='../imagens'):
    plt.figure(pn)
    plt.plot(x, f, color='black')
    zr = np.zeros((x.shape[0], 1))
    plt.scatter(x, zr, s=0.5)
    if save_path:
        plt.savefig(f'{save_path}/figure_{pn}.png')

def plot_spec(s, pn, save_path='../imagens'):
    plt.figure(pn)
    plt.plot(s, s, '*', color='black')
    if save_path:
        plt.savefig(f'{save_path}/figure_{pn}.png')
