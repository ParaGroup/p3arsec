#!/bin/bash

MEASURE=true
INPUT=true
ONLYINPUT=false
APPLICATIONS="blackscholes ferret swaptions"
KERNELS="canneal dedup"
ALLAPS="$APPLICATIONS $KERNELS"
VERSIONS="gcc-pthreads gcc-openmp gcc-tbb gcc-ff"

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
					echo "build_deps=\"\${build_deps} hooks\"" >> $FILE
				fi
			done
		done
	fi

	# Repair documentation
	sh repairpod.sh

	# Clean
	mv README README_PARSEC
fi
