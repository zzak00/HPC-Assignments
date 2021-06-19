"""

@author: zzak00
"""

from mpi4py import MPI
import numpy as np
from numpy.core.function_base import linspace

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

xmin=0
xmax=3*np.pi/2
n=20
h=(xmax-xmin)/n
start=xmin + h*n/SIZE*RANK
stop=start+h*n/SIZE
def parallelTrap(a,b):
    integrale=0
    i=0
    while (a+i*h)!=b:
        integrale+=np.cos(a+i*h)+np.cos(a+(i+1)*h)
        i+=1
    return h/2*integrale

integrale=parallelTrap(start,stop)

sum_red=COMM.reduce(integrale,op=MPI.SUM,root=0)

if RANK==0:
    print("integrale =", sum_red)
