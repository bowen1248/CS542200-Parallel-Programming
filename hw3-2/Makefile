NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm
EXES     := hw3-2

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

ifeq (test, $(firstword $(MAKECMDGOALS)))
  TESTCASE := $(word 2, $(MAKECMDGOALS))
  $(eval $(TESTCASE):;@:)
endif

.PHONY: test
test: $(EXES)
	@echo test on $(TESTCASE)
	./$(EXES) /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/$(TESTCASE) ./output/$(TESTCASE).out
	./verify /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/$(TESTCASE) /tmp/dataset-nthu-pp23/pp23/share/hw3-2/cases/$(TESTCASE).out ./output/$(TESTCASE).out
