CXX = g++ 
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = nvcc 
NVCXXFLAGS = -g --ptxas-options=-v -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(wildcard $(OBJDIR)/*.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/reduce

bin:
	mkdir -p bin


bin/reduce: $(OBJS) $(CUOBJS) 
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			    @echo "Compiled "$<" successfully!"


.PHONY:	test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
		mkdir -p cuobj
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

clean: 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*

#########################
# Submit
##########################
submit_64:
	mkdir -p result
	condor_submit reduce64.cmd

submit_4194304:
	mkdir -p result
	condor_submit reduce4194304.cmd

submit_8388608:
	mkdir -p result
	condor_submit reduce8388608.cmd

submit_16777216:
	mkdir -p result
	condor_submit reduce16777216.cmd
