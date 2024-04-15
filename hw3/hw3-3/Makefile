NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm -Xcompiler -fopenmp
EXES     := hw3-3

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

ifeq (test, $(firstword $(MAKECMDGOALS)))
  TESTCASE := $(word 2, $(MAKECMDGOALS))
  $(eval $(TESTCASE):;@:)
endif

.PHONY: test
test: $(EXES)
	@echo test on $(TESTCASE)
	./$(EXES) /home/pp23/share/hw3-3/cases/$(TESTCASE) ./output/$(TESTCASE).out
	./verify /home/pp23/share/hw3-3/cases/$(TESTCASE) /home/pp23/share/hw3-3/cases/$(TESTCASE).out ./output/$(TESTCASE).out

# /home/pp23/share/hw3-2/
# /tmp/dataset-nthu-pp23/pp23/share/hw3-2/