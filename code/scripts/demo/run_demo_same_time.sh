./bin/serial 1000000 NONE
mpirun -np 1 ./bin/mpi 1000000 NONE
mpirun -np 14 ./bin/mpi 1000000 NONE
mpirun -np 28 ./bin/mpi 1000000 NONE
