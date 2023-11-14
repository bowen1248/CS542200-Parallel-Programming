CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -march=native
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS) 
TARGETS = hw2b

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
	@srun -N3 -c4 hw2b out.png 10000 -0.2931209325179713 -0.2741427339562606 -0.6337125743279389 -0.6429654881215695 7680 4320
	@hw2-diff ../../testcases/strict35.png out.png
