// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,
//
// FastFlow version by Daniele De Sensi (d.desensi.software@gmail.com)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Multi-threaded pthreads header
#ifdef ENABLE_THREADS
// Add the following line so that icc 9.0 is compatible with pthread lib.
#define __thread __threadp
MAIN_ENV
#undef __thread
#endif

// Multi-threaded OpenMP header
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_TBB
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

using namespace std;
using namespace tbb;
#endif //ENABLE_TBB

#ifdef ENABLE_NORNIR_NATIVE
#include <nornir.hpp>
#endif // ENABLE_NORNIR_NATIVE

#ifdef ENABLE_NORNIR
#include <instrumenter.hpp>
#include <stdlib.h>
#include <iostream>
#endif //ENABLE_NORNIR

#if defined(ENABLE_NORNIR) || defined(ENABLE_NORNIR_NATIVE)
std::string getParametersPath(){
    return std::string(getenv("PARSECDIR")) + std::string("/parameters.xml");
}
#endif

#ifdef ENABLE_FF
#include <iostream>
#include <ff/farm.hpp>
#include <ff/map.hpp>

using namespace ff;
#endif //ENABLE_FF

#ifdef ENABLE_CAF
#include "caf/all.hpp"
#define CAF_DETACHED_WORKER false
#define CAF_N_WORKER nThreads
#endif //ENABLE_CAF

// Multi-threaded header for Windows
#ifdef WIN32
#pragma warning(disable : 4305)
#pragma warning(disable : 4244)
#include <windows.h>
#endif

//Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

OptionData *data;
fptype *prices;
int numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;
int nThreads;

#ifdef ENABLE_NORNIR
nornir::Instrumenter* instr;
#endif //ENABLE_NORNIR


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

fptype CNDF ( fptype InputX ) 
{
    int sign;

    fptype OutputX;
    fptype xInput;
    fptype xNPrimeofX;
    fptype expValues;
    fptype xK2;
    fptype xK2_2, xK2_3;
    fptype xK2_4, xK2_5;
    fptype xLocal, xLocal_1;
    fptype xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * xInput;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
    
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;
    
    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    
    return OutputX;
} 

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype, float timet )
{
    fptype OptionPrice;

    // local private working variables for the calculation
    fptype xStockPrice;
    fptype xStrikePrice;
    fptype xRiskFreeRate;
    fptype xVolatility;
    fptype xTime;
    fptype xSqrtTime;

    fptype logValues;
    fptype xLogTerm;
    fptype xD1; 
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;    
    
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );
        
    xLogTerm = logValues;
        
    
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
        
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;
    
    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );

    FutureValueX = strike * ( exp( -(rate)*(time) ) );        
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    
    return OptionPrice;
}

#ifdef ENABLE_TBB
struct mainWork {
  mainWork() {}
  mainWork(mainWork &w, tbb::split) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    fptype price;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i++) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */

      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                   rate[i], volatility[i], otime[i], 
                                   otype[i], 0);
      prices[i] = price;

#ifdef ERR_CHK 
      fptype priceDelta = data[i].DGrefval - price;
      if( fabs(priceDelta) >= 1e-5 ){
        fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
               i, price, data[i].DGrefval, priceDelta);
        numError ++;
      }
#endif
    }
  }
};

#endif // ENABLE_TBB

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_TBB
int bs_thread(void *tid_ptr) {
    int j;
    tbb::affinity_partitioner a;

    mainWork doall;
    for (j=0; j<NUM_RUNS; j++) {
      tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
    }

    return 0;
}
#else // !ENABLE_TBB

#ifdef ENABLE_FF

struct map: ff_Map<int> {
    int *svc(int *in) {
        int i, j;
        for (j=0; j<NUM_RUNS; j++) {
            parallel_for_thid(0, numOptions, 1, PARFOR_STATIC(0), [](const long i, const int thid) {
#ifdef ENABLE_NORNIR
                instr->begin(thid);
#endif //ENABLE_NORNIR 
                fptype price;
                fptype priceDelta;
                /* Calling main function to calculate option value based on
                 * Black & Scholes's equation.
                 */
                price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                             rate[i], volatility[i], otime[i],
                                             otype[i], 0);
                prices[i] = price;

#ifdef ERR_CHK
                priceDelta = data[i].DGrefval - price;
                if( fabs(priceDelta) >= 1e-4 ){
                    printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                           i, price, data[i].DGrefval, priceDelta);
                    numError ++;
                }
#endif
#ifdef ENABLE_NORNIR
                instr->end(thid);
#endif //ENABLE_NORNIR 
            }, nThreads);
        }
        return NULL;
    }
};

#else // !ENABLE_FF

#ifdef ENABLE_CAF
caf::behavior map_worker(caf::event_based_actor *self, uint32_t i, uint32_t nw) {
    return {
        [=](const size_t& start, const size_t& end) {
            for (size_t i = start; i < end; ++i) {
                fptype price;
                fptype priceDelta;
               /* Calling main function to calculate option value based on
                * Black & Scholes's equation.
                */
                price = BlkSchlsEqEuroNoDiv(sptprice[i], strike[i],
                                            rate[i], volatility[i], otime[i],
                                            otype[i], 0);
                prices[i] = price;
#ifdef ERR_CHK
                priceDelta = data[i].DGrefval - price;
                if( fabs(priceDelta) >= 1e-4 ){
                    printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                        i, price, data[i].DGrefval, priceDelta);
                    numError ++;
                }
#endif
            }
      }};
}

struct map_state {
  std::vector<caf::actor> worker;
};
caf::behavior map(caf::stateful_actor<map_state> *self,
                  uint32_t nw, bool detached) {
  // create workers
  self->state.worker.resize(nw);
  for (uint32_t i = 0; i < nw; i++) {
    caf::actor a;
    if (detached) {
      a = self->spawn<caf::detached>(map_worker, i, nw);
    } else {
      a = self->spawn<caf::lazy_init>(map_worker, i, nw);
    }
    self->state.worker[i] = a;
  }
  return {[=](const size_t& start, const size_t& end) {
    size_t nv = end - start + 1;
    size_t chunk = nv / nw;
    size_t plus = nv % nw;

    size_t p_start = start;
    for (uint32_t iw = 0; iw < nw; iw++) {
        size_t p_end = p_start+chunk;
        if (plus > 0){
          p_end++;
          plus--;
        }
        self->send(self->state.worker[iw], p_start, p_end);
        p_start = p_end;
    }
  }};
}
#else // !ENABLE_CAF

#ifdef ENABLE_NORNIR_NATIVE
void nornirloop(){
    int i, j;
    nornir::ParallelFor pf(nThreads, new nornir::Parameters(getParametersPath()));
    for (j=0; j<NUM_RUNS; j++) {
        pf.parallel_for(0, numOptions, 1, 0, 
            [](const long long i, const uint thid) {
            fptype price;
            fptype priceDelta;
            /* Calling main function to calculate option value based on
             * Black & Scholes's equation.
             */
            price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                         rate[i], volatility[i], otime[i],
                                         otype[i], 0);
            prices[i] = price;

#ifdef ERR_CHK
            priceDelta = data[i].DGrefval - price;
            if( fabs(priceDelta) >= 1e-4 ){
                printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                       i, price, data[i].DGrefval, priceDelta);
                numError ++;
            }
#endif
        });
    }
}
#else // ENABLE_NORNIR_NATIVE

#ifdef WIN32
DWORD WINAPI bs_thread(LPVOID tid_ptr){
#else
int bs_thread(void *tid_ptr) {
#endif
    int i, j;
    fptype price;
    fptype priceDelta;
    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions / nThreads);
    int end = start + (numOptions / nThreads);

    for (j=0; j<NUM_RUNS; j++) {
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i++) {
#else  //ENABLE_OPENMP
        for (i=start; i<end; i++) {
#endif //ENABLE_OPENMP

#ifdef ENABLE_NORNIR
#ifdef ENABLE_OPENMP
        instr->begin();
#else
        instr->begin(tid);
#endif
#endif //ENABLE_NORNIR 
            /* Calling main function to calculate option value based on 
             * Black & Scholes's equation.
             */
            price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                         rate[i], volatility[i], otime[i], 
                                         otype[i], 0);
            prices[i] = price;

#ifdef ERR_CHK
            priceDelta = data[i].DGrefval - price;
            if( fabs(priceDelta) >= 1e-4 ){
                printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                       i, price, data[i].DGrefval, priceDelta);
                numError ++;
            }
#endif

#ifdef ENABLE_NORNIR
#ifdef ENABLE_OPENMP
        instr->end();
#else
        instr->end(tid);
#endif
#endif //ENABLE_NORNIR 
        }
    }
    return 0;
}
#endif //ENABLE_FF
#endif //ENABLE_CAF
#endif //ENABLE_TBB
#endif

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    fptype * buffer;
    int * buffer2;
    int rv;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version " __PARSEC_XSTRING(PARSEC_VERSION) "\n");
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_blackscholes);
#endif

   if (argc != 4)
        {
                printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
                exit(1);
        }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    //Read input data from file
    file = fopen(inputFile, "r");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }
    if(nThreads > numOptions) {
      printf("WARNING: Not enough work, reducing number of threads to match number of options.\n");
      nThreads = numOptions;
    }

#if !defined(ENABLE_THREADS) && !defined(ENABLE_OPENMP) && !defined(ENABLE_TBB) && !defined(ENABLE_FF) && !defined(ENABLE_CAF) && !defined(ENABLE_NORNIR_NATIVE)
    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version)\n");
        exit(1);
    }
#endif

#ifdef ENABLE_NORNIR
#ifdef ENABLE_OPENMP
    instr = new nornir::Instrumenter(getParametersPath());
#else
    instr = new nornir::Instrumenter(getParametersPath(), nThreads);
#endif
#endif //ENABLE_NORNIR
    
    // alloc spaces for the option data
    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }

#ifdef ENABLE_THREADS
    MAIN_INITENV(,8000000,nThreads);
#endif
    printf("Num of Options: %d\n", numOptions);
    printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;    
        otime[i]      = data[i].t;
    }

    printf("Size of data: %d\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS
#ifdef WIN32
    HANDLE *threads;
    int *nums;
    threads = (HANDLE *) malloc (nThreads * sizeof(HANDLE));
    nums = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        nums[i] = i;
        threads[i] = CreateThread(0, 0, bs_thread, &nums[i], 0, 0);
    }
    WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
    free(threads);
    free(nums);
#else
    int *tids;
    tids = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        tids[i]=i;
        CREATE_WITH_ARG(bs_thread, &tids[i]);
    }
    WAIT_FOR_END(nThreads);
    free(tids);
#endif //WIN32
#else //ENABLE_THREADS
#ifdef ENABLE_OPENMP
    {
        int tid=0;
        omp_set_num_threads(nThreads);
        bs_thread(&tid);
    }
#else //ENABLE_OPENMP
#ifdef ENABLE_TBB
    tbb::task_scheduler_init init(nThreads);

    int tid=0;
    bs_thread(&tid);
#else //ENABLE_TBB
#ifdef ENABLE_FF
    map m;
    m.run();
    m.wait();
#else //ENABLE_FF
#ifdef ENABLE_CAF
{
    caf::actor_system_config cfg;
    cfg.set("scheduler.max-threads", nThreads);
    caf::actor_system sys{cfg};
    // caf::scoped_actor self{sys};

    auto map_inst = sys.spawn(map, CAF_N_WORKER, CAF_DETACHED_WORKER);
    
    for (uint32_t j=0; j<NUM_RUNS; j++) {
        caf::anon_send(map_inst, size_t{0}, (size_t)numOptions);
    }
}
#else //ENABLE_CAF
#ifdef ENABLE_NORNIR_NATIVE
    nornirloop();
#else // ENABLE_NORNIR_NATIVE
    //serial version
    int tid=0;
    bs_thread(&tid);
#endif //ENABLE_NORNIR_NATIVE
#endif //ENABLE_CAF
#endif //ENABLE_FF
#endif //ENABLE_TBB
#endif //ENABLE_OPENMP
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

#ifdef ENABLE_NORNIR
    instr->terminate();
    std::cout << "riff.time|" << instr->getExecutionTime() << std::endl;
    std::cout << "riff.iterations|" << instr->getTotalTasks() << std::endl;
    delete instr;
#endif //ENABLE_NORNIR

    //Write prices to output file
    file = fopen(outputFile, "w");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", outputFile);
      exit(1);
    }
    rv = fprintf(file, "%i\n", numOptions);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
      fclose(file);
      exit(1);
    }
    for(i=0; i<numOptions; i++) {
      rv = fprintf(file, "%.18f\n", prices[i]);
      if(rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
      }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", outputFile);
      exit(1);
    }

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif
    free(data);
    free(prices);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}

