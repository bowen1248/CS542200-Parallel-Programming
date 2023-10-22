#!/bin/bmpiash
module load tbb/latest compiler-rt/latest oclfpga/latest compiler/latest 

# mpicc -O3 ./hw1.cc -o ./hw1
mpicc -O3 -lm -L /opt/ipm/lib ./hw1.cc -o ./hw1
srun -p judge -N2 -n2 ./hw1 4 ../testcases/01.in ./01.out
# srun -p judge -N4 -n28 ./hw1 15 ../testcases/02.in ./02.out
# srun -p judge -N2 -n24 ./hw1 11183 ../testcases/03.in ./03.out
srun -p judge -N2 -n10 ./hw1 65536 ../testcases/03.in ./03.out
