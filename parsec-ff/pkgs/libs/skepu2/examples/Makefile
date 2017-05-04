################################################################################
#                     ____  _        ____  _   _   ____                        #
#                    / ___|| | _____|  _ \| | | | |___ \                       #
#                    \___ \| |/ / _ \ |_) | | | |   __) |                      #
#                     ___) |   <  __/  __/| |_| |  / __/                       #
#                    |____/|_|\_\___|_|    \___/  |_____|                      #
#                                                                              #
#                          ~ SkePU 2 main Makefile ~                           #
################################################################################

# NOTE:
# This Makefile should be kept realtively platform-agnostic.
# Only sensible default values are defined here.
# Platform-specific include paths, user-local test programs etc.,
# should be specified in Makefile.in or on the command line.


# ---------------------------------------------------------------------------- #
# Default values which can be changed in Makefile.in

# Default backend set for SkePU precompiler.
BACKENDS = -openmp -opencl # -cuda

SKEPU_OPTIONS =
PRECOMPILER_OPTIONS =

# Backend-specific flags
BACKEND_FLAGS =

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -lOpenCL -I /usr/local/cuda/include/


include Makefile.in

# ---------------------------------------------------------------------------- #
# Compilers.

# Conditionally choose either g++ or nvcc for target compiler,
# depending on whether CUDA is in the list of backends
# This will also be used for non-preprocessed compilation of SkePU programs.
ifneq (,$(findstring cuda, $(BACKENDS)))
CXX = nvcc
endif

# Location of SkePU precompiler binary.
SKEPU = $(LLVM_BIN)/skepu-tool


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

OUT_DIR = bin
SEQ_DIR = $(OUT_DIR)/sequential

# All SkePU 2 exmple programs (used for 'make all-[parallel|sequential]' and 'make clean').
PROGS = map reduce mapreduce scan mapoverlap call
PROGS += mandelbrot mvmult mmmult cma ppmcc psnr taylor coulombic nbody median
PROGS += writeback dotproduct lambda 

TESTS = $(addprefix $(SEQ_DIR)/, $(PROGS))
PTESTS = $(addprefix $(OUT_DIR)/, $(PROGS))


# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(BACKENDS) $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- -std=c++11 -Wno-expansion-to-defined
SKEPU_FLAGS += -I $(CLANG_SRC)/lib/Headers
SKEPU_FLAGS += -I $(SKEPU_SRC)/include
SKEPU_FLAGS += $(CLANGTOOL_USER_FLAGS)

# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
NVCCPASS = -Xcompiler
FILETYPE = cu
else
FILETYPE = cpp
endif

# Activate backend flags for OpenMP backend
ifneq (,$(findstring openmp, $(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif

# Activate backend flags for OpenCL backend
ifneq (,$(findstring opencl, $(BACKENDS)))
BACKEND_FLAGS += $(OPENCL_FLAGS)
endif

# Flags for target compiler (preprocessed sources).
TARGET_FLAGS = -g -O3 -std=c++11 $(SKEPU_OPTIONS) -I $(SKEPU_SRC)/include -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS = -std=c++11 -O3 -I $(SKEPU_SRC)/include -Wno-attributes # -fno-rtti -fno-exceptions 


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

# Builds a single sequential test program.
$(SEQ_DIR)/% : %.cpp
	$(DBGR) $(CXX) $(SEQ_FLAGS) $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/median : median.cpp
	$(DBGR) $(CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(OUT_DIR)/%: %.cpp
	$(DBGR) $(SKEPU) -name $*_precompiled $<  -dir $(OUT_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) -o $@ $(TARGET_FLAGS)

# Precompiles and builds a single parallel test program.
$(OUT_DIR)/median: median.cpp
	$(DBGR) $(SKEPU) -name median_precompiled $<  -dir $(OUT_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp -o $@ $(TARGET_FLAGS)


# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	-$(RM) $(SEQ_DIR)/*.o $(TESTS)
	-$(RM) $(OUT_DIR)/*_precompiled.cpp $(OUT_DIR)/*_cl_source.inl $(OUT_DIR)/*.cu $(PTESTS)
