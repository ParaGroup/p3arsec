[![Build Status](https://travis-ci.org/ParaGroup/p3arsec.svg?branch=master)](https://travis-ci.org/ParaGroup/p3arsec)

# Description
This repository contains parallel patterns implementations of some applications contained in the PARSEC benchmark.

The structure and modelling of some provided applications (Blackscholes, Canneal, Dedup, Ferret and Swaptions) is described in the paper:
**P<sup>3</sup>ARSEC: Towards Parallel Patterns Benchmarking** by Marco Danelutto, Tiziano De Matteis, Daniele De Sensi, Gabriele Mencagli and Massimo Torquati. The paper has been presented at the [32nd ACM Symposium on Applied Computing (SAC)](http://www.sigapp.org/sac/sac2017/).

All the applications (except x264) have been implemented by using the [FastFlow](http://calvados.di.unipi.it/) pattern-based parallel programming framework. Some benchmarks have been also implemented with the [SkePU2](https://www.ida.liu.se/labs/pelab/skepu/) framework. In the following table you can find more details about the pattern used for each benchmark and the file(s) containing the actual implementation, both for **FastFlow** and for **SkePU2**. The pattern descriptions reported here are an approximation and exact descriptions will come later. Some benchmarks are implemented by using different patterns (**bold** pattern is the one used by default). To run the benchmark a different pattern refer to the specific [section](#run-alternative-versions) of this document.

Application   | Used Pattern           | FastFlow Files                                                       | SkePU2 Files
------------- | -----------------------|----------------------------------------------------------------------|--------------
Blackscholes  | **Map**                | [File 1](parsec-ff/pkgs/apps/blackscholes/src/blackscholes.c)        | [File 1](parsec-ff/pkgs/apps/blackscholes/src/blackscholes_skepu.cpp)
Bodytrack     | **Maps**               | [File 1](parsec-ff/pkgs/apps/bodytrack/src/TrackingBenchmark/TrackingModelFF.cpp), [File 2](parsec-ff/pkgs/apps/bodytrack/src/TrackingBenchmark/ParticleFilterFF.h)
Canneal       | **Master-Worker**      | [File 1](parsec-ff/pkgs/kernels/canneal/src/main.cpp)
Dedup         | Pipeline of Farms      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_pipeoffarms.cpp)
"             | Farm                   | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farm.cpp)
"             | Farm of Pipelines      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farmofpipes.cpp)
"             | **Ordering Farm**      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_ofarm.cpp)
Facesim       | **Maps**               | [File 1](parsec-ff/pkgs/apps/facesim/src/Benchmarks/facesim/FACE_EXAMPLE.h), [File 2](parsec-ff/pkgs/apps/facesim/src/Public_Library/Forces_And_Torques/DIAGONALIZED_FINITE_VOLUME_3D.cpp), [File 3](parsec-ff/pkgs/apps/facesim/src/Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp), [File 4](parsec-ff/pkgs/apps/facesim/src/Public_Library/Arrays/ARRAY_PARALLEL_OPERATIONS.cpp)
Ferret        | Pipeline of Farms      | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-pipeoffarms.cpp)
"             | Farm of Pipelines      | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farmofpipes.cpp)
"             | Farm                   | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farm.cpp)
"             | **Farm (Optimized)     | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farm-optimized.cpp)
Fluidanimate  | **Maps**               | [File 1](parsec-ff/pkgs/apps/fluidanimate/src/ff.cpp)
Freqmine      | **Maps**               | [File 1](parsec-ff/pkgs/apps/freqmine/src/fp_tree_ff.cpp)
Raytrace      | **Map**                | [File 1](parsec-ff/pkgs/apps/raytrace/src/LRT/render.cxx)            | [File 1](parsec-ff/pkgs/apps/raytrace/src/LRT/render_skepu_omp.cxx)
Streamcluster | **Maps and MapReduce** | [File 1](parsec-ff/pkgs/kernels/streamcluster/src/streamcluster.cpp) | [File 1](parsec-ff/pkgs/kernels/streamcluster/src/streamcluster_skepu.cpp)
Swaptions     | **Map**                | [File 1](parsec-ff/pkgs/apps/swaptions/src/HJM_Securities.cpp)       | [File 1](parsec-ff/pkgs/apps/swaptions/src/HJM_Securities_skepu.cpp)
Vips          | **Farm**               | [File 1](parsec-ff/pkgs/apps/vips/src/libvips/iofuncs/threadpool.cc)
x264          | Not available.


These implementations have been engineered in order to be used with the standard PARSEC tools.
Accordingly, you can use and evaluate the parallel patterns implementations together with
the *Pthreads*, *OpenMP* and *TBB* versions already present in PARSEC.
After following this guide, more details can be found on [PARSEC Website](http://parsec.cs.princeton.edu/).

# Download
To download PARSEC and the parallel patterns implementation of some of its benchmark, run the
following commands:

```
git clone https://github.com/ParaGroup/p3arsec.git
```

```
cd p3arsec
```

Then, run:
```
./install.sh 
```

These commands could take few minutes to complete, since it will download the original PARSEC
implementations with all the input datasets (around 3GB) and all the needed dependencies.

You can specify the following parameters to the `./install.sh` command:
* `--nomeasure`: In this case the infrastructure for measuring execution time and energy 
consumption will not be installed. If this parameter **is not** specified, then you will
be able to measure execution time and energy consumption for all the benchmarks 
(both for those implemented as parallel patterns and for those already present in PARSEC).
* `--fast`: By specifying this parameter you will only download the PARSEC source code (112MB)
and some small test inputs. You can download the input datasets later by running `./install.sh --inputs`
* `--inputs`: This parameter will **only** download the PARSEC input files. It should be used
only if `./install.sh --fast` has already been run.
* `--skeputools`: This parameter compiles and install the [SkePU2](https://www.ida.liu.se/labs/pelab/skepu/) 
source to source compiler. This is not mandatory and you only need it if you want to modify the 
`*_skepu.cpp` files. This parameter is mainly intended for developers.


# Compile
To let PARSEC properly work, some dependencies needs to be installed. For Ubuntu systems, you can do it with the following command:
```
sudo apt-get install git build-essential m4 x11proto-xext-dev libglu1-mesa-dev libxi-dev libxmu-dev libtbb-dev libssl-dev
```
Similar packages can be found for other Linux distributions.

After that, you need to install the benchmarks you are interested in:

```
cd bin
```

The parallel patterns versions of the benchmarks have been integrated with the original PARSEC management system
(`./parsecmgmt`).
You can find the full documentation [here](http://parsec.cs.princeton.edu/doc/man/man1/parsecmgmt.1.html) or
in the *README_PARSEC* file which will appear in the directory after the previous commands have been run.

To compile the parallel patterns version of a specific benchmark, is sufficient to run the following command:
```
./parsecmgmt -a build -p [BenchmarkName] -c gcc-ff
```

If you also want to compile the other existing versions of the benchmark, just replace `gcc-ff` with one of the following:

* *gcc-skepu* for the SkePU2 parallel pattern-based implementation.
* *gcc-pthreads* for the Pthreads implementation.
* *gcc-openmp* for the OpenMP implementation.
* *gcc-tbb* for the Intel TBB implementation.

Note that not all these implementations are available for all the benchmarks. For more details on supported 
implementations, please refer to the original [PARSEC documentation](http://wiki.cs.princeton.edu/index.php/PARSEC) 
(and to the top table in this file for the SkePU2 versions).

**ATTENTION: If you plan to execute the benchmark with more than 1024 threads, you need to modify the following MACROS:**
* `MAX_THREADS` in `pkgs/apps/blackscholes/src/c.m4.pthreads` file.
* `MAX_NUM_THREADS` in `pkgs/libs/fastflow/ff/config.hpp` file.

# Run
Once you compiled a benchmark, you can run it with:
```
./parsecmgmt -a run -p [BenchmarkName] -c gcc-ff -n [ConcurrencyLevel]
```
Even in this case, you can run the other existing version by replacing `gcc-ff` with the name of the desired version.
By default, the program is run on a test input. PARSEC provides different input datasets: *test*, 
*simsmall*, *simmedium*, *simlarge*, *simdev* and *native*.
The *native* input set is the one resembling a real execution scenario, while the others should be used for testing/simulation
purposes. To specify the input set, is sufficient to specify it with the `-i` parameter. For example, to run
the parallel patterns implementation of the *Canneal* benchmark on the *native* input set:
```
./parsecmgmt -a run -p canneal -c gcc-ff -i native
```
All the datasets are present if you ran `./install.sh` (or `./install.sh --fast` plus `./install.sh --inputs`).

*ConcurrencyLevel* has the same meaning it has in the original PARSEC benchmarks. It represents the concurrency level 
and it is the *minimum* number of threads that will be activated by the application. 
Accordingly, we have the following values:

* *blackscholes*: n+1 threads.
* *canneal*: n+1 threads.
* *dedup*: n threads for each pipeline stage (3n + 3 threads). (For the *pipe of farms* version.)
* *ferret*:  n threads for each pipeline stage (4n + 4 threads). (For the *pipe of farms* version.)
* *swaptions*: n+1 threads.

Some parallel patterns implementations may not follow this rule. For example, the *ordered farm* implementation
of the *dedup* benchmark will activate n+2 threads.

## Measuring time and energy consumption
If you want to measure energy consumption of the benchmarks (and if you __**do not**__ specified the `--nomeasure` parameter in
the `./install.sh` script), 
please run the benchmarks with `sudo`. In this case, in the output of the program you will find something like:

```
sudo ./parsecmgmt -a run -p canneal -c gcc-ff -i native
...
roi.time|12.3
roi.joules|TYPE|456.7
...
```

Where *12.3* is the execution time and *456.7* is the energy consumption. Both values consider only the time and energy spent in the 
*Region Of Interest (ROI)*, i.e. the parallel part of the application, excluding initialisation and cleanup phases 
(e.g. loading a dataset from the disk to the main memory). This approach is commonly used in scientific literature to evaluate 
PARSEC behaviour.

Energy measurements are provided through the [Mammut library](http://danieledesensi.github.io/mammut/).
The meaning of the energy consumption value depends on the type of energy counters available
on the running architecture. TYPE can be one of the followings:

* `CPUS`: In this case, 4 values will be printed TAB separated (e.g. `roi.joules|CPUS|400	300	0	20`).
	* The first value represents the energy consumed by all the CPUs/Sockets on the machine.
	* The second value represents the energy consumed by __**only**__ the cores on the CPUs.
	* The third value represents the energy consumed by the DRAM controllers. This counter may not be available on
	  some architectures. In this case, 0 is printed.
	* The fourth value is architecture dependent. In general it represents the energy consumed by the integrated graphic card. 
	  This counter may not be available on some architectures. In this case, 0 is printed.

	This counter is available on newer Intel architectures (Silvermont, Broadwell, Haswell, Ivy Bridge, Sandy Bridge, Skylake,
        Xeon Phi KNL).
	If you need more detailed measurements (e.g. separating the consumption of individual sockets, please [contact us](mailto:d.desensi.software@gmail.com)).
* `PLUG`: In this case only one value will be printed, corresponding to the total energy consumption of the machine (measured at the
	  power plug level. This counter is available on:
	* Architectures using a [SmartPower](http://odroid.com/dokuwiki/doku.php?id=en:odroidsmartpower).
	* IBM Power8 machines. This support is still experimental. If you need to use it, please [contact us](mailto:d.desensi.software@gmail.com).
	
If energy counters are not present, only execution time will be printed.

# Run alternative versions
Some applications (e.g. *ferret* and *dedup*) have been implemented according to different pattern compositions.
To run versions different from the default one, you need first to remove the existing one (if present).
To do so, execute:

```
./parsecmgmt -a fullclean -c gcc-ff -p [BenchmarkName]
./parsecmgmt -a fulluninstall -c gcc-ff -p [BenchmarkName]
```

To compile and run the other versions, please refer to the following sections.
## Dedup
At line 33 of the [Makefile](p3arsec/pkgs/kernels/dedup/src/Makefile), replace `encoder_ff_pipeoffarms.o` with:

* `encoder_ff_farm.o` if you want to run the *farm* version.
* `encoder_ff_pipeoffarms.o` if you want to run the *pipeline of farms* version.
* `encoder_ff_farmofpipes.o` if you want to run the *farm of pipelines* version.
* `encoder_ff_ofarm.o` if you want to run the *ordered farm* version.

After that, build and run *dedup* as usual.

## Ferret
At line 78 of the [Makefile](p3arsec/pkgs/apps/swaptions/src/Makefile), replace `ferret-ff-pipeoffarms` with:

* `ferret-ff-farm` if you want to run the *farm* version.
* `ferret-ff-farm-optimized` if you want to run the *farm (optimized)* version.
* `ferret-ff-farmofpipes` if you want to run the *farm of pipelines* version.
* `ferret-ff-pipeoffarms` if you want to run the *pipelines of farms* version.

After that, build and run *ferret* as usual.

# Contributors
P<sup>3</sup>ARSEC has been developed by [Daniele De Sensi](mailto:d.desensi.software@gmail.com) and [Tiziano De Matteis](mailto:dematteis@di.unipi.it).
