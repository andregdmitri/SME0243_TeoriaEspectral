import sys
import numpy as np
import plot_graph as pg
import matplotlib.pyplot as plt
from GFT import GFT
from functools import reduce

##########
# How to run the examples:
# python3 signal_synth.py rgd.adj rgd.xy
# python3 signal_synth.py pert400+.adj pert400+.xy
#########

A = np.loadtxt(sys.argv[1])
p = np.loadtxt(sys.argv[2])

D = np.diag(np.sum(A,axis=0))
L = D - A


###########
# Synthetized function
###########
#gft = GFT(L)
#S = gft.get_spec()
#fs = np.exp(-(S**2)/0.1)
##mp = np.ceil(np.amax(S)/2)
##fs = np.exp(-(S**2)/0.3) + np.exp(-(S-mp)**2)
##fs = np.ones((S.shape[0],1))
#gft.set_kernel(fs)
#gs = gft.synthetize()
#
#pg.plot_specf(S,fs,1)
#pg.plot_graph_s(A,p,gs,2)


###########
# Uncertainty principle
###########
#n = p.shape[0]
#fs = np.zeros((n,1))
#pc = 455
#for i in range(0,n):
#    fs[i] = np.exp(-(np.linalg.norm(p[i,:]-p[pc,:])**2)/0.5)
#
#gft = GFT(L)
#gft.set_signal(fs)
#hfs = gft.gft()
#S = gft.get_spec()
#pg.plot_specf(S,hfs,1)
#pg.plot_graph_s(A,p,fs,2)


###########
# Graph Fourier of a Gaussian
###########
#n = p.shape[0]
#fs = np.zeros((n,1))
#U = np.array([[1,0],[0,1]])
#U = np.matrix(np.sqrt(2)*U)
#D = np.matrix(np.array([[5,0],[0,50]]))
#Di = D.I
#V = reduce(np.dot,[U,Di,U.transpose()])
#pc = 200
#for i in range(0,n):
#    x = p[i,:]-p[pc,:]
#    e = reduce(np.dot,[x.transpose(),V,x])
#    fs[i] = np.exp(-e)
#
#gft = GFT(L)
#gft.set_signal(fs)
#fst = gft.gft()
#ifs = gft.igft()
#S = gft.get_spec()
#pg.plot_specf(S,fst,1)
#pg.plot_graph_s(A,p,fs,2)
#pg.plot_graph_s(A,p,fs_trans,3)


###########
# Translation
###########
gft = GFT(L)
S = gft.get_spec()
fs = np.exp(-(S**2)/0.1)
gft.set_gft_as_kernel(fs)
fs_trans = gft.translation(155)
fs_trans2 = gft.translation(555)
pg.plot_specf(S,fs,1)
pg.plot_graph_s(A,p,fs_trans,2)
pg.plot_graph_s(A,p,fs_trans2,4)

ifs = gft.igft()
pg.plot_graph_s(A,p,ifs,3)

#gft2 = GFT(L)
#S2 = gft2.get_spec()
#gft2.set_signal(ifs)
#gfifs = gft2.gft()
#pg.plot_specf(S,gfifs,4)

plt.show()
