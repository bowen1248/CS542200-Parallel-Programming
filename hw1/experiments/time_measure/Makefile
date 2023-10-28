CC = mpicc
CXX = mpicxx
cxx = icpx
CXXFLAGS = -O3 -flto -march=native -mtune=native
CFLAGS = -O3
# CXXFLAGS = -O3 -lm -L /opt/ipm/lib -lipm
# CFLAGS = -O3 -lm -L /opt/ipm/lib -lipm
TARGETS = hw1_seq

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
