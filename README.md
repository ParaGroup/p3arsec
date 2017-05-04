# Description
This repository contains parallel patterns implementations of some applications contained in the PARSEC benchmark.

The structure and modelling of the provided applications is described in the paper:
**P<sup>3</sup>ARSEC: Towards Parallel Patterns Benchmarking** by Marco Danelutto, Tiziano De Matteis, Daniele De Sensi, Gabriele Mencagli and Massimo Torquati. The paper has been presented at the [32nd ACM Symposium on Applied Computing (SAC)](http://www.sigapp.org/sac/sac2017/).

At the moment, the following applications have been implemented (remaining benchmarks are currently under development) (**bold** versions are those used by default):

Application  | Patterns implementations
------------ | ---------------------------------------------------------
Blackscholes | [**Map**](parsec-ff/pkgs/apps/blackscholes/src/blackscholes.c)
Canneal      | [**Master-Worker**](parsec-ff/pkgs/kernels/canneal/src/main.cpp)
Dedup        | [**Pipeline of Farms**](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_pipeoffarms.cpp)
"            | [Farm](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farm.cpp)
"            | [Farm of Pipelines](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_farmofpipes.cpp)
"            | [Ordering Farm](parsec-ff/pkgs/kernels/dedup/src/encoder_ff_ofarm.cpp)
Ferret       | [**Pipeline of Farms**](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-pipeoffarms.cpp)
"            | [Farm of Pipelines](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farmofpipes.cpp)
"            | [Farm](parsec-ff/pkgs/apps/ferret/src/benchmark/ferret-ff-farm.cpp)
Swaptions    | [**Map**](parsec-ff/pkgs/apps/swaptions/src/HJM_Securities.cpp)

Applications have been implemented by using the [FastFlow](http://calvados.di.unipi.it/) 
and the [SkePU2](https://www.ida.liu.se/labs/pelab/skepu/)
pattern-based parallel programming frameworks.
*x264* is the only application missing at the moment.

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
* `--skepu`: This parameter compiles and install the [SkePU2](https://www.ida.liu.se/labs/pelab/skepu/) 
source to source compiler. This is not mandatory and you only need it if you want to modify the 
`*_skepu.cpp` files.


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

* *gcc-pthreads* for the Pthreads implementation.
* *gcc-openmp* for the OpenMP implementation.
* *gcc-tbb* for the Intel TBB implementation.
* *gcc-skepu* for the SkePU2 implementation. Only available if `--skepu` flag has been used during installation.

Note that not all these implementations are available for all the benchmarks. For more details on supported 
implementations, please refer to the original [PARSEC documentation](http://wiki.cs.princeton.edu/index.php/PARSEC).

# Run
Once you compiled a benchmark, you can run it:
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
and it is the *minimum* number of threads that will be activated by the application. We try to kept the same number 
of threads activated in the original PARSEC versions. Accordingly, we have the following values:

* *blackscholes*: n+1 threads.
* *canneal*: n+1 threads.
* *dedup*: n threads for each pipeline stage (3n + 3 threads).
* *ferret*:  n threads for each pipeline stage (4n + 4 threads).
* *swaptions*: n+1 threads.

The *non-default* parallel patterns implementations may not follow this rule. For example, the *ordered farm* implementation
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
	If you need more detailed measurements (e.g. separating the consumption of individual sockets, please [contact us](mailto:Paragroup@di.unipi.it)).
* `PLUG`: In this case only one value will be printed, corresponding to the total energy consumption of the machine (measured at the
	  power plug level. This counter is available on:
	* Architectures using a [SmartPower](http://odroid.com/dokuwiki/doku.php?id=en:odroidsmartpower).
	* IBM Power8 machines. This support is still experimental. If you need to use it, please [contact us](mailto:Paragroup@di.unipi.it).
	
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
* `encoder_ff_farmofpipes.o` if you want to run the *farm of pipelines* version.
* `encoder_ff_ofarm.o` if you want to run the *ordered farm* version.

After that, build and run *dedup* as usual.

## Ferret
At line 78 of the [Makefile](p3arsec/pkgs/apps/swaptions/src/Makefile), replace `ferret-ff-pipeoffarms` with:

* `ferret-ff-farm` if you want to run the *farm* version.
* `ferret-ff-farmofpipes` if you want to run the *farm of pipelines* version.

After that, build and run *ferret* as usual.
