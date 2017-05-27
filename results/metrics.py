#!/usr/bin/python
#
# Performs different analysis on the source code of the different P3ARSEC (and parsec-ompss) versions,
# reporting different metrics.
#
# Dependencies:
# 1. coan (https://sourceforge.net/projects/coan2/)
# 2. astyle
# 3. cloc
# 4. git > 1.8.4 (for 'git diff')
# 5. lizard (https://github.com/terryyin/lizard) for cyclomatic complexity calculation ('sudo pip install lizard' should be enough)
#
# To compute locs relative to pthreads: ./metrics.py --locs -n pthreads
# To compute modified lines wrt serial: ./metrics.py --modified
#
# For other options: ./metrics.py --help

import os
import sys
import argparse
from subprocess import PIPE, Popen

# Let the following two variables point to the correct folders.

# ./install.sh must have been called on the P3ARSEC folder.
p3arsec_root = "/tmp/p3arsec/"  

# To download parsec-ompss: https://pm.bsc.es/gitlab/benchmarks/parsec-ompss
# If the following variable is empty, metrics will not be computed for OMPSS versions.
parsec_ompss_root = "/home/daniele/Code/parsec-ompss/"

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

def getFullPath(v, f):
    return "/tmp/" + v + "/src/" + f

# Returns the corresponding file in case of different filenames.
def getCorresponding(b, v, f):    
    # Blackscholes #
    if 'blackscholes' in b:
        if 'ompss' in v:
            if 'blackscholes-ompss.c' in f:
                return getFullPath('serial', 'blackscholes.c')
        if 'skepu' in v and '_skepu' in f:
            return getFullPath('serial', f.replace('_skepu.cpp', '.c'))
    # Bodytrack #
    elif 'bodytrack' in b:
        if 'ParticleFilter' in f:
            return getFullPath('serial', 'TrackingBenchmark/ParticleFilter.h')
        if 'TrackingModel' in f:
            if f.endswith('cpp'):
                return getFullPath('serial', 'TrackingBenchmark/TrackingModel.cpp')
            else:
                return getFullPath('serial', 'TrackingBenchmark/TrackingModel.h')
    # Facesim #
    elif 'facesim' in b:
        if 'ompss' in v and 'OMPSS_' in f and not 'OMPSS_COLLISION_PENALTY_FORCES.cpp' in f:
            return getFullPath('serial', f.replace('OMPSS_', ''))
    # Ferret #
    elif 'ferret' in b:
        if 'ferret-' in f:
            return getFullPath('serial', 'benchmark/ferret-serial.c')
    # Fluidanimate #
    elif 'fluidanimate' in b:
        if v in f:
            return getFullPath('serial', 'serial.cpp')
    # Freqmine #
    elif 'freqmine' in b:
        if 'ff' in v and 'fp_tree_ff.cpp' in f:
            return getFullPath('serial', 'fp_tree.cpp')
    # Raytrace #
    elif 'raytrace' in b:
        if 'skepu' in v and '_skepu' in f:
            return getFullPath('serial', f.replace('_skepu', ''))
        return ''
    # Swaptions #
    elif 'swaptions' in b:
        if 'skepu' in v and '_skepu' in f:
            return getFullPath('serial', f.replace('_skepu', ''))
        return ''                        
    # Vips #
    elif 'vips' in b:
        return ''
    # Canneal #
    elif 'canneal' in b:
        if 'ff' in v:
            if 'annealer_thread_ff.h' in f:
                return getFullPath('serial', 'annealer_thread.h')
            elif 'annealer_thread_ff.cpp' in f:
                return getFullPath('serial', 'annealer_thread.cpp')
    # Dedup #
    elif 'dedup' in b:
        if 'ff' in v:
            if 'encoder_ff_' in f:
                return getFullPath('serial', 'encoder.c')
    # Streamcluster #
    elif 'streamcluster' in b:
        if 'ompss' in v:
            if 'ompss_streamcluster.cpp' in f:
                return getFullPath('serial', 'streamcluster.cpp')
        if 'skepu' in v and '_skepu' in f:
            return getFullPath('serial', f.replace('_skepu', ''))
    else:
        print "FATAL ERROR: Unkown benchmark."
        sys.exit()

macros = {} # Must be defined if the corresponding version exists (even if empty)
files = {} # Files that differ for different implementations
locs = {}
modifiedlines = {}
cc = {}
benchmarktype = {}

benchmarks = ["blackscholes", "bodytrack", "canneal", "dedup", "facesim", "ferret", "fluidanimate", "freqmine", "raytrace", "streamcluster", "swaptions", "vips"]

versions = ["serial", "pthreads", "ff" , "tbb", "openmp", "skepu"]
if parsec_ompss_root:
    versions.append("ompss")


for b in benchmarks:
    macros[b] = {}
    files[b] = {}
    locs[b] = {}
    modifiedlines[b] = {}
    cc[b] = {}

################
# Blackscholes #
################
if 'blackscholes' in benchmarks:
    macros['blackscholes']['pthreads'] = "-DENABLE_THREADS"
    macros['blackscholes']['ff'] = "-DENABLE_FF"
    macros['blackscholes']['openmp'] = "-DENABLE_OPENMP"
    macros['blackscholes']['tbb'] = "-DENABLE_TBB"
    macros['blackscholes']['serial'] = ""
    macros['blackscholes']['ompss'] = ""
    macros['blackscholes']['skepu'] = ""
    files['blackscholes']['pthreads'] = ["blackscholes.c"]
    files['blackscholes']['ff'] = ["blackscholes.c"]
    files['blackscholes']['openmp'] = ["blackscholes.c"]
    files['blackscholes']['tbb'] = ["blackscholes.c"]
    files['blackscholes']['serial'] = ["blackscholes.c"]
    files['blackscholes']['ompss'] = ["blackscholes-ompss.c"]
    files['blackscholes']['skepu'] = ["blackscholes_skepu.cpp"]
    benchmarktype["blackscholes"] = "apps"

#############
# Bodytrack #
#############
if 'bodytrack' in benchmarks:
    macros['bodytrack']['pthreads'] = "-DUSE_THREADS=1 -DHAVE_LIBPTHREAD"
    macros['bodytrack']['ff'] = "-DUSE_FF=1"
    macros['bodytrack']['openmp'] = "-DUSE_OPENMP=1"
    macros['bodytrack']['tbb'] = "-DUSE_TBB=1"
    macros['bodytrack']['serial'] = ""
    macros['bodytrack']['ompss'] = "-DUSE_OMPSS=1"
    files['bodytrack']['pthreads'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilterPthread.h", "TrackingBenchmark/TrackingModelPthread.h", "TrackingBenchmark/TrackingModelPthread.cpp", "TrackingBenchmark/WorkPoolPthread.h", "TrackingBenchmark/AsyncIO.h", "TrackingBenchmark/threads/Thread.h",
    "TrackingBenchmark/threads/Mutex.cpp", "TrackingBenchmark/threads/Mutex.h", "TrackingBenchmark/threads/Barrier.cpp", "TrackingBenchmark/threads/WorkerGroup.cpp", "TrackingBenchmark/threads/SynchQueue.h", "TrackingBenchmark/threads/Condition.cpp", "TrackingBenchmark/threads/atomic/sparc/cpufunc.h",
    "TrackingBenchmark/threads/atomic/sparc/asi.h", "TrackingBenchmark/threads/atomic/sparc/atomic.h", "TrackingBenchmark/threads/atomic/atomic.h", "TrackingBenchmark/threads/atomic/powerpc/cpufunc.h", "TrackingBenchmark/threads/atomic/powerpc/atomic.h", "TrackingBenchmark/threads/atomic/i386/atomic.h", "TrackingBenchmark/threads/atomic/amd64/atomic.h", "TrackingBenchmark/threads/atomic/ia64/atomic.h", "TrackingBenchmark/threads/Thread.cpp", "TrackingBenchmark/threads/RWLock.h", "TrackingBenchmark/threads/Condition.h", "TrackingBenchmark/threads/ThreadGroup.cpp", "TrackingBenchmark/threads/WorkerGroup.h", "TrackingBenchmark/threads/TicketDispenser.h", "TrackingBenchmark/threads/LockTypes.h", "TrackingBenchmark/threads/RWLock.cpp", "TrackingBenchmark/threads/ThreadGroup.h", "TrackingBenchmark/threads/Barrier.h"]
    files['bodytrack']['ff'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilterFF.h", "TrackingBenchmark/TrackingModelFF.h", "TrackingBenchmark/TrackingModelFF.cpp"]
    files['bodytrack']['openmp'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilterOMP.h", "TrackingBenchmark/TrackingModelOMP.h", "TrackingBenchmark/TrackingModelOMP.cpp"]
    files['bodytrack']['tbb'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilterTBB.h", "TrackingBenchmark/TrackingModelTBB.h", "TrackingBenchmark/TrackingModelTBB.cpp"]
    files['bodytrack']['serial'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilter.h", "TrackingBenchmark/TrackingModel.h", "TrackingBenchmark/TrackingModel.cpp"]
    files['bodytrack']['ompss'] = ["TrackingBenchmark/main.cpp", "TrackingBenchmark/ParticleFilterOMPSS.h", "TrackingBenchmark/TrackingModelOMPSS.h", "TrackingBenchmark/TrackingModelOMPSS.cpp"]
    benchmarktype["bodytrack"] = "apps"

###########
# Canneal #
###########
if 'canneal' in benchmarks:
    macros['canneal']['pthreads'] = "-DENABLE_THREADS"
    macros['canneal']['ff'] = "-DENABLE_THREADS -DENABLE_FF"
    macros['canneal']['serial'] = ""
    macros['canneal']['ompss'] = "-DENABLE_OMPSS"
    files['canneal']['pthreads'] = ["main.cpp", "annealer_thread.h", "annealer_thread.cpp"]
    files['canneal']['ff'] = ["main.cpp", "annealer_thread_ff.h", "annealer_thread_ff.cpp"]
    files['canneal']['serial'] = ["main.cpp", "annealer_thread.h", "annealer_thread.cpp"]
    files['canneal']['ompss'] = ["main.cpp", "annealer_thread.h", "annealer_thread.cpp"]
    benchmarktype["canneal"] = "kernels"

#########
# Dedup #
#########
if 'dedup' in benchmarks:
    macros['dedup']['pthreads'] = "-DENABLE_PTHREADS"
    macros['dedup']['ff'] = "-DENABLE_FF -DENABLE_FF_ONDEMAND"
    macros['dedup']['serial'] = ""
    macros['dedup']['ompss'] = "-DENABLE_OMPSS -DENABLE_OMPSS_LOCKS"
    files['dedup']['pthreads'] = ["encoder.c", "queue.h", "binheap.h", "tree.h", "queue.c", "binheap.c", "tree.c"]
    files['dedup']['ff'] = ["encoder_ff_ofarm.cpp"]
    files['dedup']['serial'] = ["encoder.c"]
    files['dedup']['ompss'] = ["encoder.c"]
    benchmarktype["dedup"] = "kernels"

###########
# Facesim #
###########
if 'facesim' in benchmarks:
    macros['facesim']['pthreads'] = "-DENABLE_PTHREADS"
    macros['facesim']['ff'] = "-DENABLE_FF"
    macros['facesim']['serial'] = "-DNEW_SERIAL_IMPLEMENTATIOM"
    macros['facesim']['ompss'] = "-DENABLE_OMPSS -DUSE_TASKS"
    files['facesim']['pthreads'] = ["Benchmarks/facesim/FACE_EXAMPLE.h", "Public_Library/Arrays/ARRAY_PARALLEL_OPERATIONS.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp", "Public_Library/Forces_And_Torques/DIAGONALIZED_FINITE_VOLUME_3D.cpp", "Public_Library/Thread_Utilities/THREAD_POOL.cpp", "Public_Library/Thread_Utilities/THREAD_POOL.h", "Public_Library/Collisions_And_Interactions/COLLISION_PENALTY_FORCES.h", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT_3D.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp"]
    files['facesim']['ff'] = ["Benchmarks/facesim/FACE_EXAMPLE.h", "Public_Library/Arrays/ARRAY_PARALLEL_OPERATIONS.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp", "Public_Library/Forces_And_Torques/DIAGONALIZED_FINITE_VOLUME_3D.cpp", "Public_Library/Collisions_And_Interactions/COLLISION_PENALTY_FORCES.h", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT_3D.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp"]
    files['facesim']['serial'] = ["Benchmarks/facesim/FACE_EXAMPLE.h", "Public_Library/Arrays/ARRAY_PARALLEL_OPERATIONS.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp", "Public_Library/Forces_And_Torques/DIAGONALIZED_FINITE_VOLUME_3D.cpp", "Public_Library/Collisions_And_Interactions/COLLISION_PENALTY_FORCES.h", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT_3D.cpp", "Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp"]
    files['facesim']['ompss'] = ["Benchmarks/facesim/OMPSS_FACE_EXAMPLE.h", "Public_Library/Arrays/OMPSS_ARRAY_PARALLEL_OPERATIONS.cpp", "Public_Library/Deformable_Objects/OMPSS_DEFORMABLE_OBJECT.cpp", "Public_Library/Forces_And_Torques/OMPSS_DIAGONALIZED_FINITE_VOLUME_3D.cpp", "Public_Library/Collisions_And_Interactions/OMPSS_COLLISION_PENALTY_FORCES.cpp", "Public_Library/Collisions_And_Interactions/COLLISION_PENALTY_FORCES.h", "Public_Library/Deformable_Objects/OMPSS_DEFORMABLE_OBJECT_3D.cpp", "Public_Library/Deformable_Objects/OMPSS_DEFORMABLE_OBJECT.cpp"]
    benchmarktype["facesim"] = "apps"

##########
# Ferret #
##########
if 'ferret' in benchmarks:
    macros['ferret']['pthreads'] = ""
    macros['ferret']['ff'] = ""
    macros['ferret']['tbb'] = ""
    macros['ferret']['serial'] = ""
    macros['ferret']['ompss'] = ""
    files['ferret']['pthreads'] = ["benchmark/ferret-pthreads.c", "include/tpool.h", "src/tpool.c"]
    files['ferret']['ff'] = ["benchmark/ferret-ff-farm.cpp"]
    files['ferret']['tbb'] = ["benchmark/ferret-tbb.cpp", "benchmark/ferret-tbb.h"]
    files['ferret']['serial'] = ["benchmark/ferret-serial.c"]
    files['ferret']['ompss'] = ["benchmark/ferret-ompss.c"]
    benchmarktype["ferret"] = "apps"

################
# Fluidanimate #
################
if 'fluidanimate' in benchmarks:
    macros['fluidanimate']['pthreads'] = ""
    macros['fluidanimate']['ff'] = ""
    macros['fluidanimate']['tbb'] = ""
    macros['fluidanimate']['serial'] = ""
    macros['fluidanimate']['ompss'] = ""
    files['fluidanimate']['pthreads'] = ["pthreads.cpp", "parsec_barrier.hpp", "parsec_barrier.cpp"]
    files['fluidanimate']['ff'] = ["ff.cpp"]
    files['fluidanimate']['tbb'] = ["tbb.cpp"]
    files['fluidanimate']['serial'] = ["serial.cpp"]
    files['fluidanimate']['ompss'] = ["ompss-multideps-nobar.cpp"]
    benchmarktype["fluidanimate"] = "apps"

############
# Freqmine #
############
if 'freqmine' in benchmarks:
    macros['freqmine']['openmp'] = ""
    macros['freqmine']['ff'] = ""
    macros['freqmine']['serial'] = ""
    macros['freqmine']['ompss'] = "-D_OMPSS"
    files['freqmine']['openmp'] = ["fp_tree.cpp"]
    files['freqmine']['ff'] = ["fp_tree_ff.cpp"]
    files['freqmine']['serial'] = ["fp_tree.cpp"]
    files['freqmine']['ompss'] = ["fp_tree.cpp"]
    benchmarktype["freqmine"] = "apps"

############
# Raytrace #
############
if 'raytrace' in benchmarks:
    macros['raytrace']['pthreads'] = ""
    macros['raytrace']['ff'] = "-DFF_VERSION"
    macros['raytrace']['serial'] = ""
    macros['raytrace']['skepu'] = ""
    files['raytrace']['pthreads'] = ["LRT/render.cxx", "RTTL/common/RTThread.cxx", "RTTL/common/RTThread.hxx"]
    files['raytrace']['ff'] = ["LRT/render.cxx"]
    files['raytrace']['serial'] = ["LRT/render.cxx"]
    files['raytrace']['skepu'] = ["LRT/render_skepu.cxx"]
    benchmarktype["raytrace"] = "apps"

#############
# Swaptions #
#############
if 'swaptions' in benchmarks:
    macros['swaptions']['pthreads'] = "-DENABLE_THREADS"
    macros['swaptions']['ff'] = "-DENABLE_THREADS -DFF_VERSION"
    macros['swaptions']['tbb'] = "-DENABLE_THREADS -DTBB_VERSION"
    macros['swaptions']['serial'] = ""
    macros['swaptions']['ompss'] = "-DENABLE_OMPSS"
    macros['swaptions']['skepu'] = ""
    files['swaptions']['pthreads'] = ["HJM_Securities.cpp"]
    files['swaptions']['ff'] = ["HJM_Securities.cpp"]
    files['swaptions']['tbb'] = ["HJM_Securities.cpp"]
    files['swaptions']['serial'] = ["HJM_Securities.cpp"]
    files['swaptions']['ompss'] = ["HJM_Securities.cpp"]
    files['swaptions']['skepu'] = ["HJM_Securities_skepu.cpp"]
    benchmarktype["swaptions"] = "apps"

#################
# Streamcluster #
#################
if 'streamcluster' in benchmarks:
    macros['streamcluster']['pthreads'] = "-DENABLE_THREADS"
    macros['streamcluster']['ff'] = "-DFF_VERSION"
    macros['streamcluster']['tbb'] = "-DTBB_VERSION"
    macros['streamcluster']['serial'] = ""
    macros['streamcluster']['ompss'] = "-DENABLE_OMPSS"
    macros['streamcluster']['skepu'] = ""
    files['streamcluster']['pthreads'] = ["streamcluster.cpp", "parsec_barrier.hpp", "parsec_barrier.cpp"]
    files['streamcluster']['ff'] = ["streamcluster.cpp"]
    files['streamcluster']['tbb'] = ["streamcluster.cpp"]
    files['streamcluster']['serial'] = ["streamcluster.cpp"]
    files['streamcluster']['ompss'] = ["ompss_streamcluster.cpp"]
    files['streamcluster']['skepu'] = ["streamcluster_skepu.cpp"]
    benchmarktype["streamcluster"] = "kernels"

########
# Vips #
########
if 'vips' in benchmarks:
    macros['vips']['pthreads'] = "-DHAVE_THREADS"
    macros['vips']['ff'] = "-DHAVE_FF -DHAVE_THREADS"
    macros['vips']['serial'] = ""
    files['vips']['pthreads'] = ["libvips/iofuncs/threadpool.cc"]
    files['vips']['ff'] = ["libvips/iofuncs/threadpool.cc"]
    files['vips']['serial'] = ["libvips/iofuncs/threadpool.cc"]
    benchmarktype["vips"] = "apps"


# Parse arguments
parser = argparse.ArgumentParser(description='Computes metrics over source code.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--modified', help='Prints number of modified lines of code.', action='store_true', required=False, default=False)
parser.add_argument('-l', '--locs', help='Prints number of lines of code.', action='store_true', required=False, default=False)
parser.add_argument('-c', '--cyclomatic', help='Prints cyclomatic complexity.', action='store_true', required=False, default=False)
parser.add_argument('-v', '--verbose', help='Verbose mode.', action='store_true', required=False, default=False)
parser.add_argument('-b', '--benchmark', help='Computes metrics just for one benchmark.', required=False)
parser.add_argument('-k', '--keepfiles', help='Do not remove temporary files (just for debugging purposes).', action='store_true', required=False, default=False)
parser.add_argument('-n', '--normalize', help='Normalizes lines of code with respect to a specified version .', required=False)
args = parser.parse_args()

# Compute Metrics
for benchmark in benchmarks:
    if args.benchmark and args.benchmark not in benchmark:
        continue
    for v in versions:
        if not v in macros[benchmark]:
            continue
        # In the folder /tmp/[VERSION]/src/ there will be the files of the specific
        # version (with the not-used parts of code removed).
        if "ompss" in v:
            filepath = parsec_ompss_root + "/" + benchmark
        else:
            filepath = p3arsec_root + "/pkgs/" + benchmarktype[benchmark] + "/" + benchmark
        # Copy original files in appropriate folders (needed for computing different lines).
        os.system("mkdir -p /tmp/" + v + "/src")
        os.system("cp -r " + filepath + "/src/* /tmp/" + v + "/src/")
        for f in files[benchmark][v]:
            tmpdir = f.rsplit('/', 1)[0]
            if v not in locs[benchmark]:
                locs[benchmark][v] = 0 
            if v not in modifiedlines[benchmark]:
                modifiedlines[benchmark][v] = 0 
            if v not in cc[benchmark]:
                cc[benchmark][v] = 0 
            os.system("grep -v '^#include' " + filepath + "/src/" + f + " | grep -v '^# include' > " + getFullPath(v, f))
            os.system("astyle --style=banner --suffix=none " + getFullPath(v, f) + ">/dev/null")
            os.system("coan source --implicit -ge -gs " + macros[benchmark][v] + " " + getFullPath(v, f) + " | grep -v '^# '> " + getFullPath(v, f) + ".clean.c")
            os.system("mv " + getFullPath(v, f) + ".clean.c " + getFullPath(v, f))
            # Remove comments
            cmdline("cloc --csv --strip-comments=sc --original-dir --force-lang=\"C++\",hxx --quiet " + getFullPath(v, f))
            cmdline("mv " + getFullPath(v, f) + ".sc " + getFullPath(v, f))
            # Count LOCs
            locs[benchmark][v] += float(cmdline("cloc --csv --force-lang=\"C++\",hxx --quiet " + getFullPath(v, f) + " | tail -n 1 | cut -d ',' -f 5 | tr -d '\n'"))
            # Compute Cyclomatic complexity
            cc[benchmark][v] += float(cmdline("lizard -l cpp " + getFullPath(v, f) + " | tail -n 1 | sed 's/\s\s*/ /g' | cut -d' ' -f4"))
               
    # At this point we have all the cleaned file in the correct folders.
    for v in versions:
        if args.verbose:
            print "###################"
            print "#####"  + v + "####"
            print "###################"
        if not v in macros[benchmark]:
            continue
        for f in files[benchmark][v]:
            correspondingFile = getCorresponding(benchmark, v, f)
            if args.verbose and correspondingFile:
                print "Corresponding of: " + benchmark + " " + v + " " + f + ": " + str(correspondingFile)
            # Compute modified lines
            if v not in 'serial':
                if f in files[benchmark]['serial']:
                    difflines = float(cmdline("git diff --minimal --no-index --ignore-all-space --ignore-blank-lines " + getFullPath('serial', f) + " " + getFullPath(v, f) + " | diffstat | tail -n 1 | cut -d ',' -f 2 | cut -d ' ' -f 2"))
                    modifiedlines[benchmark][v] += difflines
                    if args.verbose:
                        print f + " differs with " + str(difflines) + " lines"
                # Different file name but 'same' code (e.g. blackscholes-ompss.c vs blackscholes.c)
                elif correspondingFile:
                    difflines = float(cmdline("git diff --minimal --no-index --ignore-all-space --ignore-blank-lines " + correspondingFile + " " + getFullPath(v, f) + " | diffstat | tail -n 1 | cut -d ',' -f 2 | cut -d ' ' -f 2"))
                    modifiedlines[benchmark][v] += difflines
                    if args.verbose:
                        print f + " differs with " + str(difflines) + " lines"
                else:
                    newlines = float(cmdline("cloc --csv --force-lang=\"C++\",hxx --quiet " + getFullPath(v, f) + " | tail -n 1 | cut -d ',' -f 5 | tr -d '\n'"))
                    modifiedlines[benchmark][v] += newlines
                    if args.verbose:
                        print f + " introduces " + str(newlines) + " lines"

    if not args.keepfiles:
        for v in versions:
            os.system("rm -rf /tmp/" + v)

# Normalization
if args.normalize:
    for benchmark in benchmarks:
        normalizer = args.normalize
        if args.benchmark and args.benchmark not in benchmark:
            continue
        if 'freqmine' in benchmark and 'pthreads' in normalizer:
            normalizer = "openmp"
        for v in versions:
            if not v in macros[benchmark]:
                continue
            if not normalizer in v:
                locs[benchmark][v] = (locs[benchmark][v] / locs[benchmark][normalizer]) 
        locs[benchmark][normalizer] = 1

#Print results
sys.stdout.write("#Bench\t")    
for v in versions:
    sys.stdout.write(v + "\t")
sys.stdout.write("\n")
for benchmark in benchmarks:
    if args.benchmark and args.benchmark not in benchmark:
        continue
    sys.stdout.write("\"" + benchmark + "\"\t")    
    for v in versions:
        if not v in macros[benchmark]:
            sys.stdout.write("-1\t")
        else:
            if args.locs:
                sys.stdout.write(str(locs[benchmark][v]) + "\t")
            elif args.modified:
                if 'serial' in v:
                    sys.stdout.write("0\t")
                else:
                    sys.stdout.write(str(modifiedlines[benchmark][v]) + "\t")
            elif args.cyclomatic:
                sys.stdout.write(str(cc[benchmark][v] / len(files[benchmark][v])) + "\t")
    
    sys.stdout.write("\n")
