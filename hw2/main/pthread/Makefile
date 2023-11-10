CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS) -std=c++17
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
	@srun -n1 -c1 hw2a out.png 2602 -3 0.2 -3 0.2 979 2355
	@hw2-diff ../../testcases/fast01.png out.png
