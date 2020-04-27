[![Build Status](https://travis-ci.org/ParaGroup/p3arsec.svg?branch=master)](https://travis-ci.org/ParaGroup/p3arsec)
[![release](https://img.shields.io/github/release/paragroup/p3arsec.svg)](https://github.com/paragroup/p3arsec/releases/latest)
[![HitCount](http://hits.dwyl.io/paragroup/p3arsec.svg)](http://hits.dwyl.io/paragroup/p3arsec)


# P<sup>3</sup>ARSEC
This repository contains parallel patterns implementations of some applications contained in the PARSEC benchmark.

The structure and modelling of the applications is described in the paper:

[**Bringing Parallel Patterns out of the Corner: the P<sup>3</sup>ARSEC Benchmark Suite**</br>
Daniele De Sensi, Tiziano De Matteis, Massimo Torquati, Gabriele Mencagli, Marco Danelutto</br>
*ACM Transactions on Architecture and Code Optimization (TACO), October 2017* </br>](https://dl.acm.org/citation.cfm?id=3132710)

[Release v1.0](https://github.com/paragroup/p3arsec/releases/tag/v1.0) was used in the paper.

All the applications (except x264) have been implemented by using the [FastFlow](http://calvados.di.unipi.it/) pattern-based parallel programming framework.
Some benchmarks have been also implemented with the [SkePU2](https://www.ida.liu.se/labs/pelab/skepu/) framework and other with the [C++ Actor Framework (CAF)](https://github.com/actor-framework/actor-framework).
In the following table you can find more details about the pattern used for each benchmark and the file(s) containing the actual implementation, both for **FastFlow**, **SkePU2** and **CAF**.
The pattern descriptions reported here are an approximation and exact descriptions will come later.
Some benchmarks are implemented by using different patterns (**bold** pattern is the one used by default).
To run the benchmark a different pattern refer to the specific [section](#run-alternative-versions) of this document.

| Application   | Used Pattern           | FastFlow Files                                                                                                                                                                                                                                                                                                                                                                               | SkePU2 Files                                                               | CAF Files                                                                        |
| ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Blackscholes  | **Map**                | [File 1](parsec-ff/pkgs/apps/blackscholes/src/blackscholes.c)                                                                                                                                                                                                                                                                                                                                | [File 1](parsec-ff/pkgs/apps/blackscholes/src/blackscholes_skepu.cpp)      | [File 1](parsec-ff/pkgs/apps/blackscholes/src/blackscholes.c)                    |
| Bodytrack     | **Maps**               | [File 1](parsec-ff/pkgs/apps/bodytrack/src/TrackingBenchmark/TrackingModelFF.cpp), [File 2](parsec-ff/pkgs/apps/bodytrack/src/TrackingBenchmark/ParticleFilterFF.h)                                                                                                                                                                                                                          |                                                                            |                                                                                  |
| Canneal       | **Master-Worker**      | [File 1](parsec-ff/pkgs/kernels/canneal/src/main.cpp)                                                                                                                                                                                                                                                                                                                                        |                                                                            | [File 1](parsec-ff/pkgs/kernels/canneal/src/main.cpp)                            |
| Dedup         | Pipeline of Farms      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_pipeoffarms.cpp)                                                                                                                                                                                                                                                                                                                        |                                                                            |                                                                                  |
| "             | Farm                   | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farm.cpp)                                                                                                                                                                                                                                                                                                                               |                                                                            |                                                                                  |
| "             | Farm of Pipelines      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farmofpipes.cpp)                                                                                                                                                                                                                                                                                                                        |                                                                            |                                                                                  |
| "             | **Ordering Farm**      | [File 1](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_ofarm.cpp)                                                                                                                                                                                                                                                                                                                              |                                                                            |                                                                                  |
| Facesim       | **Maps**               | [File 1](parsec-ff/pkgs/apps/facesim/src/Benchmarks/facesim/FACE_EXAMPLE.h), [File 2](parsec-ff/pkgs/apps/facesim/src/Public_Library/Forces_And_Torques/DIAGONALIZED_FINITE_VOLUME_3D.cpp), [File 3](parsec-ff/pkgs/apps/facesim/src/Public_Library/Deformable_Objects/DEFORMABLE_OBJECT.cpp), [File 4](parsec-ff/pkgs/apps/facesim/src/Public_Library/Arrays/ARRAY_PARALLEL_OPERATIONS.cpp) |                                                                            |                                                                                  |
| Ferret        | Pipeline of Farms      | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-pipeoffarms.cpp)                                                                                                                                                                                                                                                                                                                 |                                                                            | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-caf-pipeoffarms.cpp)    |
| "             | Farm of Pipelines      | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farmofpipes.cpp)                                                                                                                                                                                                                                                                                                                 |                                                                            | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-caf-farmofpipes.cpp)    |
| "             | Farm                   | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farm.cpp)                                                                                                                                                                                                                                                                                                                        |                                                                            |                                                                                  |
| "             | **Farm (Optimized)**   | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farm-optimized.cpp)                                                                                                                                                                                                                                                                                                              |                                                                            | [File 1](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-caf-farm-optimized.cpp) |
| Fluidanimate  | **Maps**               | [File 1](parsec-ff/pkgs/apps/fluidanimate/src/ff.cpp)                                                                                                                                                                                                                                                                                                                                        |                                                                            |                                                                                  |
| Freqmine      | **Maps**               | [File 1](parsec-ff/pkgs/apps/freqmine/src/fp_tree_ff.cpp)                                                                                                                                                                                                                                                                                                                                    |                                                                            |                                                                                  |
| Raytrace      | **Map**                | [File 1](parsec-ff/pkgs/apps/raytrace/src/LRT/render.cxx)                                                                                                                                                                                                                                                                                                                                    | [File 1](parsec-ff/pkgs/apps/raytrace/src/LRT/render_skepu.cxx)            | [File 1](parsec-ff/pkgs/apps/raytrace/src/LRT/render.cxx)                        |
| Streamcluster | **Maps and MapReduce** | [File 1](parsec-ff/pkgs/kernels/streamcluster/src/streamcluster.cpp)                                                                                                                                                                                                                                                                                                                         | [File 1](parsec-ff/pkgs/kernels/streamcluster/src/streamcluster_skepu.cpp) |                                                                                  |
| Swaptions     | **Map**                | [File 1](parsec-ff/pkgs/apps/swaptions/src/HJM_Securities.cpp)                                                                                                                                                                                                                                                                                                                               | [File 1](parsec-ff/pkgs/apps/swaptions/src/HJM_Securities_skepu.cpp)       |                                                                                  |
| Vips          | **Farm**               | [File 1](parsec-ff/pkgs/apps/vips/src/libvips/iofuncs/threadpool.cc)                                                                                                                                                                                                                                                                                                                         |                                                                            |                                                                                  |
| x264          | Not available.         |                                                                                                                                                                                                                                                                                                                                                                                              |                                                                            |                                                                                  |


These implementations have been engineered in order to be used with the standard PARSEC tools.
Accordingly, you can use and evaluate the parallel patterns implementations together with the *Pthreads*, *OpenMP* and *TBB* versions already present in PARSEC.
After following this guide, more details can be found on [PARSEC Website](http://parsec.cs.princeton.edu/).

# Download
To download the last version of P<sup>3</sup>RSEC, run the
following commands:

```
wget https://github.com/ParaGroup/p3arsec/archive/v1.0.tar.gz
```

```
tar -xvf v1.0.tar.gz 
```

```
cd p3arsec-1.0
```

Then, run:
```
./install.sh 
```

These commands could take few minutes to complete, since it will download the original PARSEC implementations with all the input datasets (around 3GB) and all the needed dependencies.

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

For Arch Linux, the following:

```
sudo pacman -Sy git m4 xorgproto glu libxi libxmu intel-tbb openssl
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
* *gcc-caf for the CAF implementation

Note that not all these implementations are available for all the benchmarks.
For more details on supported implementations, please refer to the original [PARSEC documentation](http://wiki.cs.princeton.edu/index.php/PARSEC) (and to the top table in this file for the SkePU2 and FastFlow versions).

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

# Enforcing performance and power consumption objectives

It is possible to specify requirements on performance (throughput or execution time) and/or power and energy
consumption for all the benchmarks. We provide this possibility by exploiting dynamic reconfiguration of the applications 
by relying on [Nornir](http://danieledesensi.github.io/nornir/) runtime. The runtime will automatically change the number of cores
allocated to the application and their clock frequency. 
To exploit this possibility, you need to put an XML file (called ```parameters.xml```) in the p3arsec root directory, 
containing requirements in terms of performance and power consumption. The XML file must have the following format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<nornirParameters>
    <requirements>
        <throughput>100</throughput>
        <powerConsumption>MIN</powerConsumption> 
    </requirements>
</nornirParameters>
```
In this specific example, we require the application to have a troughput greater
than 100 iterations per second. Moreover, since many configurations may provide
such throughput, we require Nornir to choose the configuration with the 
lowest power consumption among those with a feasible throughput.
For more details about the type of parameters that can be specified please refer 
to [Nornir Documentation](http://danieledesensi.github.io/nornir/description.html#parameters).
The meaning of *iteration* (i.e. the way in which we measure the throughput) is application-specific. In the
following table we show what do we mean for *iteration* for each benchmark application:

| Application   | Iteration                             |
| ------------- | ------------------------------------- |
| Blacksholes   | 1 Stock Option                        |
| Bodytrack     | 1 Frame                               |
| Canneal       | 1 Move                                |
| Dedup         | 1 Chunk                               |
| Facesim       | 1 Frame                               |
| Ferret        | 1 Query                               |
| Fluidanimate  | 1 Frame                               |
| Freqmine      | 1 Call of the *FP_growth* function    |
| Raytrace      | 1 Frame                               |
| Streamcluster | 1 Evaluation for opening a new center |
| Swaptions     | 1 Simulation                          |
| Vips          | 1 Image Tile                          |
| x264          | 1 Frame                               |

For example, the example XML file we shown before would
enforce *Blackscholes* to process at least 100 Stock Options per second.

If you want to compile/run applications with dynamic reconfiguration enabled, use the following
configurations (to be specified through the ```-c``` parameter):

* *gcc-ff-nornir* for the FastFlow implementation.
* *gcc-pthreads-nornir* for the Pthreads implementation.
* *gcc-openmp-nornir* for the OpenMP implementation.
* *gcc-tbb-nornir* for the Intel TBB implementation.

**ATTENTION: To run *gcc-&ast;-nornir* configurations sudo rights are required since we need to perform some high-priviledge operations such as: reading the power consumption,
dynamically scaling the clock frequency, etc...**

# Contributors
P<sup>3</sup>ARSEC has been developed by [Daniele De Sensi](mailto:d.desensi.software@gmail.com) and [Tiziano De Matteis](mailto:dematteis@di.unipi.it).
