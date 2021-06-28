# Parts of this makefile is based on: https://stackoverflow.com/questions/3968656/how-to-compile-mpi-and-non-mpi-version-of-the-same-program-with-automake
# The vast majority was based off of this: https://stackoverflow.com/a/27794283, which in turn references http://scottmcpeak.com/autodepend/autodepend.html.
# I used that as a starting point and simply duplicated necessary parts to accommodate compiling a CUDA and MPI Binary.
# This was also used, but later discarded: https://stackoverflow.com/questions/58575795/makefile-rule-for-depends-on-header-file-if-present.

# The idea behind this makefile is that it automatically generates .d dependency files (using something like g++ -MM), 
# and uses this to determine when to recompile which file => only if it or its dependencies changed. This results in faster, incremental builds.

#Compiler and Linker
CC          := gcc
# CXX			:= g++
CXX			:= g++
# MPICXX 		:= mpicxx.mpich
# On batch, own compiled MPI.
MPICXX 		:= mpicxx#.mpich
NVCC		:= nvcc
# NVCC		:=/usr/local/cuda-10.0/bin/nvcc

#The Target Binary Program
TARGET      := serial

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
INCDIR      := src
BUILDDIR    := obj
TARGETDIR   := bin
RESDIR      := res
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

#Flags, Libraries and Includes
# Could use to profile
PROF_FLAGS  := #-pg

# Define flags, any of the following are possible.
# -DLOG_ALL: 				If set, write intermediate results to a text file, for more data. The CSVs get written regardless.
# -DDEMO:	 				If set, run the demo instead of the experiments
# -DPOPULATION_VERBOSE 		Control verbosity of serial population
# -DMPI_VERBOSE 			Control verbosity of mpi code
# -DCUDA_VERBOSE			Control verbosity of CUDA code
DFLAGS      ?= 

# CXX flags => proper c++ standard, -O3, and native architecture
CXXFLAGS    := -std=c++14 -Wno-unused-command-line-argument -O3 $(DFLAGS) $(PROF_FLAGS) -march=native
# No linking
LIB         := 
INC         := -I./src/common/vendor/random/include -I$(INCDIR) -I/usr/local/include
INCDEP      := -I$(INCDIR)

# NVCC specific flags
NVCCINC		:= -I./src/CUDA/inc -I$(INCDIR) -I./src/common/vendor/random/include
NVCCFLAGS   :=  $(DFLAGS) -use_fast_math  -std=c++14
NVCC_XCOMPILER := -isystem /usr/local/cuda-10.0-alternative/include -MM

#---------------------------------------------------------------------------------
# Sources for normal is all .cpp files that isn't the main mpi file.
SOURCES     := $(shell find $(SRCDIR) -not -path "src/MPI/*" -type f -name *.$(SRCEXT))
# MPI sources is all .cpp files that isn't serial main file.
MPI_SOURCES     := $(shell find $(SRCDIR) -not -path "src/serial/main.cpp" -type f -name *.$(SRCEXT))

# Objects is simply all .cpp => .o
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
# But, to compile MPI objects separately, add in a '_mpi' suffix to not clash.
MPI_OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(MPI_SOURCES:.$(SRCEXT)=_mpi.$(OBJEXT)))

# All cuda sources
CUDA_SOURCES  := $(shell find $(SRCDIR) -type f -name *.cu)
CUDA_OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUDA_SOURCES:.cu=.$(OBJEXT)))
# Some specific host code that the CUDA program needs.
HOST_OBJS_FOR_CUDA := obj/common/problems/salesman/TSP.o obj/common/metrics/metrics.o obj/common/problems/salesman/SalesmanUtils.o obj/common/utils/random/Random.o obj/common/experiments/Experiment.o obj/common/problems/rosenbrock/rosenbrock.o obj/common/utils/utils.o

#Default Make, all 3.
all: serial mpi cuda

#Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))
-include $(MPI_OBJECTS:.$(OBJEXT)=.$(DEPEXT))
-include $(MPI_OBJECTS:.$(OBJEXT)=_mpi.$(DEPEXT))
-include $(CUDA_OBJECTS:.$(OBJEXT)=.$(DEPEXT))



serial: $(TARGETDIR)/$(TARGET)

mpi: bin/mpi

#Link Serial
$(TARGETDIR)/$(TARGET): $(OBJECTS)
	@printf "%-10s: linking   %-30s -> %-100s\n" $(CXX) "objects"  $(TARGETDIR)/$(TARGET)
    ifeq ($(DFLAGS),-DDEMO)
		@printf "%-10s: in demo mode\n" $(CXX)
    endif
	@printf "\n"
	@$(CXX) $(PROF_FLAGS) -o $(TARGETDIR)/$(TARGET) $^ $(LIB)

#Link MPI
bin/mpi: $(MPI_OBJECTS)
	@printf "%-10s: linking   %-30s -> %-100s\n" $(MPICXX) "objects"  bin/mpi
    ifeq ($(DFLAGS),-DDEMO)
		@printf "%-10s: in demo mode\n" $(MPICXX)
    endif
	@printf "\n"
	@$(MPICXX) -o bin/mpi $^ $(LIB)

#Link CUDA
cuda: bin/cuda
bin/cuda: $(CUDA_OBJECTS) $(HOST_OBJS_FOR_CUDA)
	@printf "%-10s: linking   %-30s -> %-100s\n" $(NVCC) "objects"  bin/cuda
    ifeq ($(DFLAGS),-DDEMO)
		@printf "%-10s: in demo mode\n" $(NVCC)
    endif
	@printf "\n"
	@$(NVCC) $(NVCCFLAGS) $(NVCCINC) $(CUDA_OBJECTS) $(HOST_OBJS_FOR_CUDA) -o bin/cuda

#Remake
remake: cleaner all

#Copy Resources from Resources Directory to Target Directory
resources: directories
	@cp $(RESDIR)/* $(TARGETDIR)/

#Make the Directories
directories:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#Clean only Objecst
cleaner:
	@$(RM) -rf $(BUILDDIR)/*

#Full Clean, Objects and Binaries
clean: cleaner
	@$(RM) -rf $(TARGETDIR)/*



#Compile Serial
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
	@printf "%-10s: compiling %-30s -> %-100s\n" $(CXX) $(shell basename $<)  $@
	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp
	
#Compile MPI
$(BUILDDIR)/%_mpi.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	@printf "%-10s: compiling %-30s -> %-100s\n" $(MPICXX) $(shell basename $<)  $@
	@$(MPICXX) $(CXXFLAGS) $(INC) -c -o $@ $<
	@$(MPICXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*_mpi.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*_mpi.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*_mpi.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

#Compile CUDA
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	@printf "%-10s: compiling %-30s -> %-100s\n" $(NVCC) $(shell basename $<)  $@
	@$(NVCC) -dc $(NVCCFLAGS) $(NVCCINC) -c -o $@ $<
	@$(NVCC) -E -Xcompiler "$(NVCC_XCOMPILER)" $(NVCCFLAGS) $(NVCCINC) $(SRCDIR)/$*.cu > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp


#Non-File Targets
.PHONY: all remake clean cleaner resources

