import sys
import numpy as np
from functools import reduce

class GFT():
    def __init__(self, L):

       self.S,self.U = np.linalg.eigh(L)
       self.n = self.U.shape[0]
       self.f = np.empty((0))
       self.gft_f = np.zeros([self.n,1])
       self.igft_f = np.zeros([self.n,1])
       self.kernel = np.empty((0))
       self.gft_cpt = False

#######
    def set_signal(self, f):
        self.f = f

#######
    def set_kernel(self, k):
        self.kernel = k

#######
    def synthetize(self):
        if ((self.kernel).size == 0):
            sys.exit("Empty kernel: use GFT.set_kernel")

        sf = np.zeros((self.n,1))
        for i in range(0,self.n):
            sf[:,0] += self.kernel[i]*self.U[:,i]

        return(sf)

#######
    def gft(self):
        if ((self.f).size == 0):
            sys.exit("Empty function: use GFT.set_signal")

        for i in range(0,self.n):
                self.gft_f[i,0] = np.dot((self.f).transpose(),self.U[:,i])

        self.gft_cpt = True
        return(self.gft_f)

#######
    def igft(self):
        if (self.gft_cpt == False):
            sys.exit("Empty Spectrum: use GFT.gft")

        for i in range(0,self.n):
            self.igft_f[:,0] += self.gft_f[i]*self.U[:,i]

        return self.igft_f

#######
    def get_spec(self):
        return(self.S)

#######
    def set_gft_as_kernel(self,k):
        self.gft_f[:,0] = k[:]
        self.gft_cpt = True

#######
    def translation(self,j):

        if (self.gft_cpt == False):
            if ((self.f).size == 0):
                sys.exit("Empty function: use GFT.set_signal")
            self.gft(self.f)

        ft = np.zeros((self.n,1))
        for i in range(0,self.n):
            ft[:,0] = ft[:,0] + self.gft_f[i,0]*self.U[j,i]*self.U[:,i]

        ft[:,0] = np.sqrt(self.n)*ft[:,0]
        return ft


