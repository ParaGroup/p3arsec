PREFIX := ${PARSECDIR}/pkgs/apps/swaptions/inst/${PARSECPLAT}
DEF =
INCLUDE =

EXEC = swaptions 
SECFILE = HJM_Securities.o

ifdef version
  ifeq "$(version)" "pthreads" 
    DEF := $(DEF) -DENABLE_THREADS
    CXXFLAGS := $(CXXFLAGS) -pthread
  endif
  ifeq "$(version)" "tbb"
    DEF := $(DEF) -DENABLE_THREADS -DTBB_VERSION
    LIBS := $(LIBS) -ltbb
  endif
  ifeq "$(version)" "ff"
    DEF := $(DEF) -DENABLE_THREADS -DFF_VERSION
  endif
  ifeq "$(version)" "skepu"
	# ./skepu-tool -openmp /home/daniele/Code/p3arsec/parsec-ff/pkgs/apps/swaptions/src/HJM_Securities_skepu.cpp -name=/tmp/HJM_Securities_skepu_omp -fnames="HJM_Swaption_Blocking __assert_fail" -- -std=c++11 -I../lib/Headers -I/home/daniele/Desktop/p3arsec_with_tools/pkgs/libs/skepu2/include -I/usr/include -I/home/daniele/Code/parsec-3.0-ff/pkgs/apps/swaptions/src
    SECFILE := HJM_Securities_skepu_omp.o
  endif
endif

OBJS= CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o \
	HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o  \
	$(SECFILE)

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(DEF) $(OBJS) $(INCLUDE) $(LIBS) -o $(EXEC)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.cpp -o $*.o

.c.o:
	$(CXX) $(CXXFLAGS) $(DEF) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(EXEC) $(PREFIX)/bin/$(EXEC)

