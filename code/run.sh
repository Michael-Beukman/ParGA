# This runs the entire demo
make clean
DFLAGS=-DDEMO make all
echo "DEMO - Same Time"
./scripts/demo/run_demo_same_time.sh

echo "DEMO - Faster"
./scripts/demo/run_demo_faster.sh

echo "DEMO - Balanced"
./scripts/demo/run_demo_balanced.sh

echo "DEMO - CUDA"
./scripts/demo/run_demo_cuda.sh
