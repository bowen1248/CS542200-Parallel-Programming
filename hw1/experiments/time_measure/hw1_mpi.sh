#!/bin/bash
#SBATCH -ptest
#SBATCH -N 3
#SBATCH -n 12

module load mpi/latest
# export IPM_REPORT=full
# export IPM_REPORT_MEM=yes
# export IPM_LOG=full
# export LD_PRELOAD=/opt/ipm/lib/libipm.so
# export IPM_HPM=“PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_REF_CYC,PAPI_SP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP”
# mpirun ./hw1 65536 ../testcases/03.in ./03.out
# mpicc -n 12 ./hw1 536869888 ../testcases/40.in ./40.out
# IPM_REPORT=full IPM_REPORT_MEM=yes IPM_LOG=full LD_PRELOAD=/opt/ipm/lib/libipm.so
make all
srun -pjudge -N2 -n10 ./hw1 65536 ../../testcases/03.in ./03.out > timing3.txt
srun -pjudge -N3 -n12 ./hw1 536831999 ../../testcases/38.in ./38.out > timing38.txt
srun -pjudge -N3 -n12 ./hw1 536869888 ../../testcases/40.in ./40.out > timing40.txt