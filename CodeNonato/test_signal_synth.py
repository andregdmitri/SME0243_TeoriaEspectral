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
# Translation
###########
gft = GFT(L)

gft.set_signal(p)
S = gft.get_spec()
fs = np.exp(-(S**2)/0.1)

gft.gft()
#gft.set_gft_as_kernel(fs)
#fs_trans = gft.translation(155)
#fs_trans2 = gft.translation(555)

pg.plot_specf(S,fs,1)
pg.plot_graph_s(A=A,p=p,cm=None, pn=2)#,fs_trans,2)
pg.plot_graph_s(A=A,p=p,cm=None, pn=4)#,fs_trans2,4)

ifs = gft.igft()
pg.plot_graph_s(A,p,ifs,3)

# pg.savefig('')


# plt.savefig(f'../imagens/figure_5.png')
plt.show()

# python3 signal_synth.py rgd.adj rgd.xy