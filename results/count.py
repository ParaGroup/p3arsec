#!/usr/bin/python
#
# Performs different analysis on the source code of the different P3ARSEC versions,
# reporting different metrics.
#
# Dependencies:
# 1. coan (https://sourceforge.net/projects/coan2/)
# 2. astyle
# 3. cloc

import os
import sys
from subprocess import PIPE, Popen

# Let the following two variables point to the correct folders.
p3arsec_root = "/tmp/p3arsec/pkgs/"
parsec_ompss_root = "/home/daniele/Code/parsec-ompss/"

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]


versions = ["pthreads", "ff", "openmp", "tbb", "serial", "ompss"]

macros = {} # Must be defined if the corresponding version exists (even if empty)
files = {} # Files that differ for different implementations
locs = {}
benchmarktype = {}

#benchmarks = ["blackscholes", "bodytrack", "facesim", "ferret", "fluidanimate", "freqmine", "raytrace", "swaptions", "vips", "canneal", "dedup", "streamcluster"]
benchmarks = ["ferret"]

for b in benchmarks:
    macros[b] = {}
    files[b] = {}
    locs[b] = {}

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
    files['blackscholes']['pthreads'] = ["blackscholes.c"]
    files['blackscholes']['ff'] = ["blackscholes.c"]
    files['blackscholes']['openmp'] = ["blackscholes.c"]
    files['blackscholes']['tbb'] = ["blackscholes.c"]
    files['blackscholes']['serial'] = ["blackscholes.c"]
    files['blackscholes']['ompss'] = ["blackscholes-ompss.c"]
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
    files['canneal']['ff'] = ["main.cpp"]
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
    files['raytrace']['pthreads'] = ["LRT/render.cxx", "RTTL/common/RTThread.cxx", "RTTL/common/RTThread.hxx"]
    files['raytrace']['ff'] = ["LRT/render.cxx"]
    files['raytrace']['serial'] = ["LRT/render.cxx"]
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
    files['swaptions']['pthreads'] = ["HJM_Securities.cpp"]
    files['swaptions']['ff'] = ["HJM_Securities.cpp"]
    files['swaptions']['tbb'] = ["HJM_Securities.cpp"]
    files['swaptions']['serial'] = ["HJM_Securities.cpp"]
    files['swaptions']['ompss'] = ["HJM_Securities.cpp"]
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
    files['streamcluster']['pthreads'] = ["streamcluster.cpp", "parsec_barrier.hpp", "parsec_barrier.cpp"]
    files['streamcluster']['ff'] = ["streamcluster.cpp"]
    files['streamcluster']['tbb'] = ["streamcluster.cpp"]
    files['streamcluster']['serial'] = ["streamcluster.cpp"]
    files['streamcluster']['ompss'] = ["ompss_streamcluster.cpp"]
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

# Compute LOCs
for benchmark in benchmarks:
    for v in versions:
        if not v in macros[benchmark]:
            continue
        for f in files[benchmark][v]:
            f = "/src/" + f
            if "ompss" in v:
                filepath = parsec_ompss_root + "/" + benchmark + f
            else:
                filepath = p3arsec_root + benchmarktype[benchmark] + "/" + benchmark + f
            tmpdir = f.rsplit('/', 1)[0]
            if v not in locs[benchmark]:
                locs[benchmark][v] = 0 
            os.system("mkdir -p /tmp/" + tmpdir)
            os.system("grep -v '^#include' " + filepath + " | grep -v '^# include' > /tmp/" + f)
            os.system("astyle --style=banner /tmp/" + f + ">/dev/null")
            os.system("coan source --implicit -ge -gs " + macros[benchmark][v] + " /tmp/" + f + " | grep -v '^# '> /tmp/" + f + ".clean.c") 
            #print cmdline("cloc --csv --quiet /tmp/" + f + ".clean.c | tail -n 1 | cut -d ',' -f 5 | tr -d '\n'")
            locs[benchmark][v] += int(cmdline("cloc --csv --quiet /tmp/" + f + ".clean.c | tail -n 1 | cut -d ',' -f 5 | tr -d '\n'"))

#Print results
for benchmark in benchmarks:
    print benchmark
    for v in versions:
        sys.stdout.write(v + "\t")
    sys.stdout.write("\n")
    for v in versions:
        if not v in macros[benchmark]:
            sys.stdout.write("N.A.\t")
        else:
            sys.stdout.write(str(locs[benchmark][v]) + "\t")
    sys.stdout.write("\n")
