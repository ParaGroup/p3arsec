#!/bin/bash

MEASURE=true
INPUT=true
ONLYINPUT=false
NORNIR=false
SKEPU=false
APPLICATIONS="blackscholes bodytrack facesim ferret fluidanimate freqmine raytrace swaptions vips"
KERNELS="canneal dedup streamcluster"
ALLAPS="$APPLICATIONS $KERNELS"
VERSIONS="gcc-pthreads gcc-openmp gcc-tbb gcc-ff gcc-serial gcc-skepu"

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
#    -e|--extension)
#    EXTENSION="$2"
#    shift # past argument (Needed only if the parameter has an argument: e.g. --foo bar).
#    ;;
    -m|--nomeasure)
    MEASURE=false
    ;;
    -f|--fast)
    INPUT=false
    ;;
    -i|--inputs)
    ONLYINPUT=true
    ;;
    -n|--nornir)
    NORNIR=true
    ;;
    -s|--skeputools)
    SKEPU=true
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done

if [ "$ONLYINPUT" = true ]; then
	# Only download and add the inputs
	wget http://parsec.cs.princeton.edu/download/3.0/parsec-3.0-input-sim.tar.gz
	tar -xvzf parsec-3.0-input-sim.tar.gz
	rm -f parsec-3.0-input-sim.tar.gz
	rsync --remove-source-files -a parsec-3.0/ ./
	rm -rf parsec-3.0

	wget http://parsec.cs.princeton.edu/download/3.0/parsec-3.0-input-native.tar.gz
	tar -xvzf parsec-3.0-input-native.tar.gz
	rm -f parsec-3.0-input-native.tar.gz
	rsync --remove-source-files -a parsec-3.0/ ./
	rm -rf parsec-3.0
else
	# Get PARSEC
	if [ "$INPUT" = true ]; then
		wget http://parsec.cs.princeton.edu/download/3.0/parsec-3.0.tar.gz
		tar -xvzf parsec-3.0.tar.gz
		rm -f parsec-3.0.tar.gz
	else
		wget http://parsec.cs.princeton.edu/download/3.0/parsec-3.0-core.tar.gz
		tar -xvzf parsec-3.0-core.tar.gz
		rm -f parsec-3.0-core.tar.gz
	fi

	# Copy its content in the current directory
	rsync --remove-source-files -a parsec-3.0/ ./
	rm -rf parsec-3.0

	# Overwrite files with parsec-ff ones.
	rsync --remove-source-files -a parsec-ff/ ./
	rm -rf parsec-ff

	# Overwrite files with parsec-hooks ones.
	if [ "$MEASURE" = true ]; then
		rsync --remove-source-files -a parsec-hooks/ ./
		rm -rf parsec-hooks
		# Download and install Mammut (for energy measurements)
		mkdir pkgs/libs/mammut
		git clone https://github.com/DanieleDeSensi/mammut.git pkgs/libs/mammut
		pushd pkgs/libs/mammut
		make 
		popd
		# Add "hooks" to 'build_deps'
		echo $ALLAPS
		for APP in $ALLAPS
		do
			for VER in $VERSIONS
			do
				if [[ $APPLICATIONS == *$APP* ]]
				then
					AFOLD="apps"
				else
					AFOLD="kernels"
				fi
				FILE="pkgs/"$AFOLD"/"$APP"/parsec/"$VER".bldconf"
				echo $APP" "$VER" "$FILE
				if [ -f $FILE ];
				then
					echo "build_deps=\"hooks \${build_deps}\"" >> $FILE
				fi
			done
		done
	fi

	rootdir=$(pwd)
	# Install Nornir
	if [ "$NORNIR" = true ]; then
		# TODO: Maybe better to integrate directly in FastFlow patterns?
		cd ./pkgs/libs && git clone https://github.com/DanieleDeSensi/nornir.git
		# TODO link already downloaded mammut.
		cd nornir && make
		cd $rootdir
		for CONFIG in $VERSIONS
		do
			if [ "$CONFIG" != "gcc-serial" ]
			then
				echo CXXFLAGS=\"\${CXXFLAGS} -DENABLE_NORNIR -I\${PARSECDIR}/pkgs/libs/nornir/src\" >> "./config/"$CONFIG".bldconf"
				echo CFLAGS=\"\${CFLAGS} -DENABLE_NORNIR -I\${PARSECDIR}/pkgs/libs/nornir/src\" >> "./config/"$CONFIG".bldconf"
				echo LIBS=\"\${LIBS} -lnornir\" >> "./config/"$CONFIG".bldconf"				
				echo LDFLAGS=\"\${LDFLAGS} -L\${PARSECDIR}/pkgs/libs/nornir/src\" >> "./config/"$CONFIG".bldconf"
			fi
		done
	fi

	# Install SkePU2
	if [ "$SKEPU" = true ]; then
		# Change LLVM_TARGETS_TO_BUILD value according to the specific architecture
		cd ./pkgs/libs/skepu2 && mkdir external
		cd external && git clone http://llvm.org/git/llvm.git && cd llvm && git checkout d3d1bf00  && cd tools 
		git clone http://llvm.org/git/clang.git && cd clang/ && git checkout 37b415dd  && git apply ../../../../clang_patch.patch
		ln -s $(pwd)/../../../../clang_precompiler $(pwd)/tools/skepu-tool
		cmake -DLLVM_TARGETS_TO_BUILD=X86 -G "Unix Makefiles" ../../ -DCMAKE_BUILD_TYPE=Release && make -j skepu-tool
		cd $rootdir
	fi

	# Repair documentation
	echo "Repairing documentation..."
	sh repairpod.sh &> /dev/null
    
    # Removing caches
    rm -rf pkgs/apps/bodytrack/src/autom4te.cache/

    # Removing VIPS old file (is needed otherwise will not compile the new version).
    rm -rf pkgs/apps/vips/src/libvips/iofuncs/threadpool.c

	# Clean
	mv README README_PARSEC
	echo "Download succeeded."
fi
