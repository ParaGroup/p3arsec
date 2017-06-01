#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel
//Modified by Christian Bienia for the PARSEC Benchmark Suite
//
// SkePU2 version by Daniele De Sensi (d.desensi.software@gmail.com)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"


#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

#include <skepu2.hpp>

int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
//FTYPE dYears = 5.5;
int iFactors = 3; 
parm *swaptions;

long seed = 1979; //arbitrary (but constant) default value (birth year of Christian Bienia)
long swaption_seed;

// =================================================
FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;

typedef struct{
	FTYPE meanPrice;
	FTYPE stdError;
}MapOutput;


MapOutput mapFunction(skepu2::Index1D index, parm elem){
	FTYPE pdSwaptionPrice[2];
	int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  elem.dStrike, 
				   elem.dCompounding, elem.dMaturity, 
				   elem.dTenor, elem.dPaymentInterval,
				   elem.iN, elem.iFactors, elem.dYears, 
				   elem.pdYield, elem.ppdFactors,
				   swaption_seed+index.i, NUM_TRIALS, BLOCK_SIZE, 0);
	assert(iSuccess == 1);
	MapOutput mo;
	mo.meanPrice = pdSwaptionPrice[0];
	mo.stdError = pdSwaptionPrice[1];
	return mo;
}
struct skepu2_userfunction_map_mapFunction
{
constexpr static size_t totalArity = 2;
constexpr static bool indexed = 1;
using ElwiseArgs = std::tuple<parm>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
};

using Ret = MapOutput;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE MapOutput OMP(skepu2::Index1D index, parm elem)
{
	FTYPE pdSwaptionPrice[2];
	int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  elem.dStrike, 
				   elem.dCompounding, elem.dMaturity, 
				   elem.dTenor, elem.dPaymentInterval,
				   elem.iN, elem.iFactors, elem.dYears, 
				   elem.pdYield, elem.ppdFactors,
				   swaption_seed+index.i, NUM_TRIALS, BLOCK_SIZE, 0);
	assert(iSuccess == 1);
	MapOutput mo;
	mo.meanPrice = pdSwaptionPrice[0];
	mo.stdError = pdSwaptionPrice[1];
	return mo;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE MapOutput CPU(skepu2::Index1D index, parm elem)
{
	FTYPE pdSwaptionPrice[2];
	int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  elem.dStrike, 
				   elem.dCompounding, elem.dMaturity, 
				   elem.dTenor, elem.dPaymentInterval,
				   elem.iN, elem.iFactors, elem.dYears, 
				   elem.pdYield, elem.ppdFactors,
				   swaption_seed+index.i, NUM_TRIALS, BLOCK_SIZE, 0);
	assert(iSuccess == 1);
	MapOutput mo;
	mo.meanPrice = pdSwaptionPrice[0];
	mo.stdError = pdSwaptionPrice[1];
	return mo;
}
#undef SKEPU_USING_BACKEND_CPU
};



//print a little help message explaining how to use this program
void print_usage(char *name) {
  fprintf(stderr,"Usage: %s OPTION [OPTIONS]...\n", name);
  fprintf(stderr,"Options:\n");
  fprintf(stderr,"\t-ns [number of swaptions (should be > number of threads]\n");
  fprintf(stderr,"\t-sm [number of simulations]\n");
  fprintf(stderr,"\t-nt [number of threads]\n");
  fprintf(stderr,"\t-sd [random number seed]\n");
}

//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
	int iSuccess = 0;
	int i,j;
	
	FTYPE **factors=NULL;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version " __PARSEC_XSTRING(PARSEC_VERSION)"\n"); 
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_swaptions);
#endif
	
        if(argc == 1)
        {
          print_usage(argv[0]);
          exit(1);
        }

        for (int j=1; j<argc; j++) {
	  if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
	  else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);} 
	  else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);} 
	  else if (!strcmp("-sd", argv[j])) {seed = atoi(argv[++j]);} 
          else {
            fprintf(stderr,"Error: Unknown option: %s\n", argv[j]);
            print_usage(argv[0]);
            exit(1);
          }
        }

        if(nSwaptions < nThreads) {
          fprintf(stderr,"Error: Fewer swaptions than threads.\n");
          print_usage(argv[0]);
          exit(1);
        }

        printf("Number of Simulations: %d,  Number of threads: %d Number of swaptions: %d\n", NUM_TRIALS, nThreads, nSwaptions);
        swaption_seed = (long)(2147483647L * RanUnif(&seed));

        // initialize input dataset
	factors = dmatrix(0, iFactors-1, 0, iN-2);
	//the three rows store vol data for the three factors
	factors[0][0]= .01;
	factors[0][1]= .01;
	factors[0][2]= .01;
	factors[0][3]= .01;
	factors[0][4]= .01;
	factors[0][5]= .01;
	factors[0][6]= .01;
	factors[0][7]= .01;
	factors[0][8]= .01;
	factors[0][9]= .01;

	factors[1][0]= .009048;
	factors[1][1]= .008187;
	factors[1][2]= .007408;
	factors[1][3]= .006703;
	factors[1][4]= .006065;
	factors[1][5]= .005488;
	factors[1][6]= .004966;
	factors[1][7]= .004493;
	factors[1][8]= .004066;
	factors[1][9]= .003679;

	factors[2][0]= .001000;
	factors[2][1]= .000750;
	factors[2][2]= .000500;
	factors[2][3]= .000250;
	factors[2][4]= .000000;
	factors[2][5]= -.000250;
	factors[2][6]= -.000500;
	factors[2][7]= -.000750;
	factors[2][8]= -.001000;
	factors[2][9]= -.001250;
	
        // setting up multiple swaptions
        swaptions = (parm *)malloc(sizeof(parm)*nSwaptions);

        int k;
        for (i = 0; i < nSwaptions; i++) {
          swaptions[i].Id = i;
          swaptions[i].iN = iN;
          swaptions[i].iFactors = iFactors;
          swaptions[i].dYears = 5.0 + ((int)(60*RanUnif(&seed)))*0.25; //5 to 20 years in 3 month intervals

          swaptions[i].dStrike = 0.1 + ((int)(49*RanUnif(&seed)))*0.1; //strikes ranging from 0.1 to 5.0 in steps of 0.1
          swaptions[i].dCompounding = 0;
          swaptions[i].dMaturity = 1.0;
          swaptions[i].dTenor = 2.0;
          swaptions[i].dPaymentInterval = 1.0;

          swaptions[i].pdYield = dvector(0,iN-1);;
          swaptions[i].pdYield[0] = .1;
          for(j=1;j<=swaptions[i].iN-1;++j)
            swaptions[i].pdYield[j] = swaptions[i].pdYield[j-1]+.005;

          swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
          for(k=0;k<=swaptions[i].iFactors-1;++k)
                 for(j=0;j<=swaptions[i].iN-2;++j)
                        swaptions[i].ppdFactors[k][j] = factors[k][j];
        }


	
	skepu2::Vector<parm> swaptions_sk(swaptions, nSwaptions, false);	
	skepu2::Vector<MapOutput> output_sk(nSwaptions);	
	skepu2::backend::Map<1, skepu2_userfunction_map_mapFunction, bool, void> map(false);
	auto spec = skepu2::BackendSpec{skepu2::Backend::Type::OpenMP};
	spec.setCPUThreads(nThreads);
	map.setBackend(spec);

	// **********Calling the Swaption Pricing Routine*****************
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif
	map(output_sk, swaptions_sk);
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

        for (i = 0; i < nSwaptions; i++) {
			const MapOutput& tmp = output_sk[i];
          fprintf(stderr,"Swaption %d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
                   i, tmp.meanPrice, tmp.stdError);
        }

        for (i = 0; i < nSwaptions; i++) {
          free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN-1);
	  free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
        }

        free(swaptions);

	//***********************************************************

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	return iSuccess;
}
