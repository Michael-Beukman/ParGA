for rosen in {0,1}; do
    #annealing, exp 1, TSP
    ./bin/serial 1 1 $rosen
done


for rosen in {0,1}; do
    #GA, exp 1, TSP and Rosen
    ./bin/serial 0 1 $rosen
done
