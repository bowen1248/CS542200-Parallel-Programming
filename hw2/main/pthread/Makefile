CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -march=native
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS) 
TARGETS = hw2a

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)

# ifeq (test, $(firstword $(MAKECMDGOALS)))
# TESTCASE := $(word 2, $(MAKECMDGOALS))
# $(eval $(TESTCASE):;@:)
# endif

.PHONY: test
test: $(TARGETS)
	@echo test
	# @srun -n1 -c1 hw2a out.png 2602 -3 0.2 -3 0.2 979 2355
	@srun -n1 -c6 hw2a out.png 54564 -0.34499 -0.34501 -0.61249 -0.61251 800 800
	@hw2-diff ../../testcases/slow02.png out.png
