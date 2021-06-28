for proc in {28, 56, 112, 224}; do
    #for proc in {224,}; do
    echo "PROC = $proc"
    for rosen in {0,1}; do
        
        #annealing, exp 1 
        ./run.sh $proc 1 1 $rosen
        
        #annealing, exp 3,
        ./run.sh $proc 1 3 $rosen

        # annealing exp 4
        ./run.sh $proc 1 4 $rosen
    done

    # Don't to multiple nodes GA, very slow.
done
