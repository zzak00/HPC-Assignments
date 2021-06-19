from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
periods = tuple([False,False])
reorder = True
dims = [2,2]
cart2d=comm.Create_cart(dims= dims ,periods = periods , reorder = reorder)
coord2d = cart2d.Get_coords( rank )
left , right = cart2d . Shift ( direction = 0 , disp=1 )
low , high= cart2d . Shift ( direction = 1 , disp=1 )
print ( " I'm rank" , rank ,"my 2dcoords are " , coord2d, "my (left,right) neighbours are",(left,right ),
"my(low,high) neighbours are " , ( low , high ) )
