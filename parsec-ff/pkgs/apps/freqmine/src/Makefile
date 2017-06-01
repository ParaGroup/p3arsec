PREFIX=${PARSECDIR}/pkgs/apps/freqmine/inst/${PARSECPLAT}
TARGET = freqmine
CXXFLAGS += -Wno-deprecated

all:
	$(MAKE) -f Makefile.${version} all

clean:
	$(MAKE) -f Makefile.${version} clean

install:
	mkdir -p $(PREFIX)/bin
	cp -f $(TARGET) $(PREFIX)/bin/$(TARGET)

