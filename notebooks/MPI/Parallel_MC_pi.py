"""

@author: zzak00
"""

from mpi4py import MPI
import numpy as np
import random 
import timeit
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def serial_compute_points(INTERVAL):
    random.seed(RANK)  
    circle_points= 0
    # Total Random numbers generated= possible x 
    # values* possible y values 
    for i in range(INTERVAL**2): 
        # Randomly generated x and y values from a 
        # uniform distribution 
        # Rannge of x and y values is -1 to 1 
        r=RANK
        rand_x= random.uniform(r%2-1,r%2) 
        rand_y= random.uniform(r%2,r%2-1) 
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1
    return circle_points

N=4000
INTERVAL= int(N/SIZE)
circle_points = serial_compute_points(INTERVAL)
area=circle_points/ INTERVAL**2 
sum_red=COMM.reduce(area,op=MPI.SUM,root=0)

if RANK==0:
    pi=4/SIZE*sum_red
    print("pi =", pi)
