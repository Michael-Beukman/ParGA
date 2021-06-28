for proc in {1,2,4,8,14,28}; do
    echo "PROC = $proc"
    for rosen in {0,1}; do
        
        #annealing, exp 1 
        ./run.sh $proc 1 1 $rosen
        
        #annealing, exp 3,
        ./run.sh $proc 1 3 $rosen

        # annealing exp 4
        ./run.sh $proc 1 4 $rosen
    done

    for rosen in {0,1}; do
        #GA, exp 1, TSP and Rosen
        ./run.sh $proc 0 1 $rosen

        #GA, exp 2, TSP and Rosen.
        ./run.sh $proc 0 2 $rosen
    done
done