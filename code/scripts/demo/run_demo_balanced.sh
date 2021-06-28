./bin/serial 1000000 NONE
mpirun -np 1 ./bin/mpi 1000000 1
mpirun -np 14 ./bin/mpi 1000000 2
mpirun -np 28 ./bin/mpi 1000000 3
