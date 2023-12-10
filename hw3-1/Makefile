CC = gcc
CXX = g++
CXXFLAGS = -O3 -fopenmp
CFLAGS = -O3 -lm -fopenmp
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

ifeq (test, $(firstword $(MAKECMDGOALS)))
  TESTCASE := $(word 2, $(MAKECMDGOALS))
  $(eval $(TESTCASE):;@:)
endif

.PHONY: test
test: $(TARGETS)
	@echo test on $(TESTCASE)
	./$(TARGETS) ./cases/$(TESTCASE) ./output/$(TESTCASE).out
	./verify ./cases/$(TESTCASE) ./cases/$(TESTCASE).out ./output/$(TESTCASE).out