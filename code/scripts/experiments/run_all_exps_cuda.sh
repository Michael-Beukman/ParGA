for rosen in {0,1}; do
    #annealing, exp 1, TSP and Rosen
    ./bin/cuda 1 1 $rosen
done

echo "DOING GA NOW"

for rosen in {0,1}; do
    #ga, exp 1, TSP and Rosen
    ./bin/cuda 0 1 $rosen
done
