CC = mpicc
CXX = mpicxx
# CXXFLAGS = -O3
# CFLAGS = -O3
CXXFLAGS = -O3 -lm -L /opt/ipm/lib -lipm
CFLAGS = -O3 -lm -L /opt/ipm/lib -lipm
TARGETS = hw1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
