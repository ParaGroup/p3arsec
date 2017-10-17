# Makefile for dedup kernel

PREFIX=${PARSECDIR}/pkgs/kernels/dedup/inst/${PARSECPLAT}

TARGET=dedup

CFLAGS += -Wall -fno-strict-aliasing -D_XOPEN_SOURCE=600

ostype=$(findstring solaris, ${PARSECPLAT})

ifeq "$(ostype)" "solaris"
    CFLAGS += -std=gnu99
endif

LIBS += -lm

DEDUP_OBJ = hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o

# Uncomment the following to enable gzip compression
CFLAGS += -DENABLE_GZIP_COMPRESSION
LIBS += -lz

# Uncomment the following to enable bzip2 compression
#CFLAGS += -DENABLE_BZIP2_COMPRESSION
#LIBS += -lbz2

ifdef version
  ifeq "$(version)" "pthreads"
    CFLAGS += -DENABLE_PTHREADS -pthread
    DEDUP_OBJ += queue.o binheap.o tree.o
  else ifeq "$(version)" "ff"
    CFLAGS += -DENABLE_FF -DENABLE_FF_ONDEMAND
    DEDUP_OBJ += queue.o binheap.o tree.o encoder_ff_ofarm.o
  else ifeq "$(version)" "nornir"
    CFLAGS += -DENABLE_NORNIR_NATIVE
    DEDUP_OBJ += queue.o binheap.o tree.o encoder_nornir.o
  endif
endif


all: $(TARGET)

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cpp.o:
	$(CXX) -c $(CFLAGS) $(CXXFLAGS) $< -o $@

$(TARGET): $(DEDUP_OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(DEDUP_OBJ) $(LIBS)

clean:
	rm -f *~ *.o $(TARGET)

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)

