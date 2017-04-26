// main.cpp
//
// Created by Daniel Schwartz-Narbonne on 13/04/07.
// Modified by Christian Bienia
// FastFlow version by Daniele De Sensi (d.desensi.software@gmail.com)
//
// Copyright 2007-2008 Princeton University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#ifdef ENABLE_THREADS
#include <pthread.h>
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

#include "annealer_types.h"
#include "annealer_thread.h"
#include "netlist.h"
#include "rng.h"

#ifdef ENABLE_FF
#include <ff/farm.hpp>
#endif

using namespace std;

void* entry_pt(void*);

#ifdef ENABLE_FF
typedef struct{
    int accepted_good_moves;
    int accepted_bad_moves;
}CTask;

static CTask dummyTask;
static CTask* allTasks;

class Emitter: public ff::ff_node{
private:
    int _temp_steps_completed;
    uint _activeWorkers;
    size_t _tasksRcvd;
    int _number_temp_steps;
    bool _keep_going_global_flag;
    ff::ff_loadbalancer* _lb;
public:
    Emitter(uint maxNumWorkers, int number_temp_steps, ff::ff_loadbalancer* lb):
        _temp_steps_completed(0),
        _activeWorkers(maxNumWorkers),
        _tasksRcvd(_activeWorkers - 1),
        _number_temp_steps(number_temp_steps),
        _keep_going_global_flag(true),
        _lb(lb){
        allTasks = new CTask[maxNumWorkers];
    }

    ~Emitter(){
        std::cout << "Generated " << _temp_steps_completed << " tasks." << std::endl;
    }

    bool keep_going(int temp_steps_completed, int accepted_good_moves, int accepted_bad_moves)
    {
        bool rv;

        if(_number_temp_steps == -1) {
            //run until design converges
            rv = _keep_going_global_flag && (accepted_good_moves > accepted_bad_moves);
            if(!rv) _keep_going_global_flag = false; // signal we have converged
        } else {
            //run a fixed amount of steps
            rv = temp_steps_completed < _number_temp_steps;
        }

        return rv;
    }

    void* svc(void*){
        ++_tasksRcvd;
        if(_tasksRcvd == _activeWorkers){
            _tasksRcvd = 0;
            ++_temp_steps_completed;

            for(size_t i = 0; i < _activeWorkers; i++){
                if(!keep_going(_temp_steps_completed,
                               allTasks[i].accepted_good_moves,
                               allTasks[i].accepted_bad_moves)){
                    return EOS;
                }
            }

            _lb->broadcast_task((void*) &dummyTask);
        }
        return GO_ON;
    }
};

class CWorker: public ff::ff_node{
private:
    Rng _rng;
    netlist* _netlist;
    double _T;
    int _swapsPerTemp;
    int _movesPerThreadTemp;
    long _a_id;
    long _b_id;
    netlist_elem* _a;
    netlist_elem* _b;
public:
    enum move_decision_t{
        move_decision_accepted_good,
        move_decision_accepted_bad,
        move_decision_rejected
    };

    move_decision_t accept_move(routing_cost_t delta_cost, double T, Rng* rng)
    {
        //always accept moves that lower the cost function
        if (delta_cost < 0){
            return move_decision_accepted_good;
        } else {
            double random_value = rng->drand();
            double boltzman = exp(- delta_cost/T);
            if (boltzman > random_value){
                return move_decision_accepted_bad;
            } else {
                return move_decision_rejected;
            }
        }
    }


    //*****************************************************************************************
    //  If get turns out to be expensive, I can reduce the # by passing it into the swap cost fcn
    //*****************************************************************************************
    routing_cost_t calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b)
    {
        location_t* a_loc = a->present_loc.Get();
        location_t* b_loc = b->present_loc.Get();

        routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
        delta_cost += b->swap_cost(b_loc, a_loc);

        return delta_cost;
    }

    CWorker(netlist* netlist, double startTemp, int swapsPerTemp, int maxNumWorkers):
        _netlist(netlist), _T(startTemp), _swapsPerTemp(swapsPerTemp),
        _movesPerThreadTemp(swapsPerTemp/maxNumWorkers),
        _a_id(0), _b_id(0){
        _a = _netlist->get_random_element(&_a_id, NO_MATCHING_ELEMENT, &_rng);
        _b = _netlist->get_random_element(&_b_id, NO_MATCHING_ELEMENT, &_rng);
    }

    void notifyRethreading(size_t oldNumWorkers, size_t newNumWorkers){
        _movesPerThreadTemp = _swapsPerTemp/newNumWorkers;
    }

    void* svc(void* task){
        _T = _T / 1.5;
        int accepted_good_moves = 0;
        int accepted_bad_moves = 0;

        for (int i = 0; i < _movesPerThreadTemp; i++){
            //get a new element. Only get one new element, so that reuse should help the cache
            _a = _b;
            _a_id = _b_id;
            _b = _netlist->get_random_element(&_b_id, _a_id, &_rng);

            routing_cost_t delta_cost = calculate_delta_routing_cost(_a, _b);
            move_decision_t is_good_move = accept_move(delta_cost, _T, &_rng);

            //make the move, and update stats:
            if (is_good_move == move_decision_accepted_bad){
                accepted_bad_moves++;
                _netlist->swap_locations(_a,_b);
            } else if (is_good_move == move_decision_accepted_good){
                accepted_good_moves++;
                _netlist->swap_locations(_a,_b);
            } else if (is_good_move == move_decision_rejected){
                //no need to do anything for a rejected move
            }
        }
        allTasks[get_my_id()].accepted_good_moves = accepted_good_moves;
        allTasks[get_my_id()].accepted_bad_moves = accepted_bad_moves;
        return &dummyTask;
    }
};
#endif //ENABLE_FF

int main (int argc, char * const argv[]) {
#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        cout << "PARSEC Benchmark Suite Version " __PARSEC_XSTRING(PARSEC_VERSION) << endl << flush;
#else
        cout << "PARSEC Benchmark Suite" << endl << flush;
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_canneal);
#endif

	srandom(3);

	if(argc != 5 && argc != 6) {
		cout << "Usage: " << argv[0] << " NTHREADS NSWAPS TEMP NETLIST [NSTEPS]" << endl;
		exit(1);
	}	
	
	//argument 1 is numthreads
	int num_threads = atoi(argv[1]);
	cout << "Threadcount: " << num_threads << endl;
#if !defined(ENABLE_THREADS) && !defined(ENABLE_FF)
	if (num_threads != 1){
		cout << "NTHREADS must be 1 (serial version)" <<endl;
		exit(1);
	}
#endif
		
	//argument 2 is the num moves / temp
	int swaps_per_temp = atoi(argv[2]);
	cout << swaps_per_temp << " swaps per temperature step" << endl;

	//argument 3 is the start temp
	int start_temp =  atoi(argv[3]);
	cout << "start temperature: " << start_temp << endl;
	
	//argument 4 is the netlist filename
	string filename(argv[4]);
	cout << "netlist filename: " << filename << endl;
	
	//argument 5 (optional) is the number of temperature steps before termination
	int number_temp_steps = -1;
        if(argc == 6) {
		number_temp_steps = atoi(argv[5]);
		cout << "number of temperature steps: " << number_temp_steps << endl;
        }

	//now that we've read in the commandline, run the program
	netlist my_netlist(filename);
	
	annealer_thread a_thread(&my_netlist,num_threads,swaps_per_temp,start_temp,number_temp_steps);
	
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS
#ifdef ENABLE_FF
    ff::ff_farm<> farm;
    std::vector<ff::ff_node*> workers;
    for(int i = 0; i < num_threads; i++){
        workers.push_back(new CWorker(&my_netlist, start_temp, swaps_per_temp, num_threads));
    }
    ff::ff_node* emitter = new Emitter(num_threads, number_temp_steps, farm.getlb());
    farm.add_emitter(emitter);
    farm.add_workers(workers);
    farm.wrap_around();
    farm.run_and_wait_end();
    farm.ffStats(std::cout);
    delete emitter;
    for(int i = 0; i < num_threads; i++){
        delete workers[i];
    }

#else
	std::vector<pthread_t> threads(num_threads);
	void* thread_in = static_cast<void*>(&a_thread);
	for(int i=0; i<num_threads; i++){
		pthread_create(&threads[i], NULL, entry_pt, thread_in);
	}
	for (int i=0; i<num_threads; i++){
		pthread_join(threads[i], NULL);
	}
#endif
#else
	a_thread.Run();
#endif
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif
	
	cout << "Final routing is: " << my_netlist.total_routing_cost() << endl;
    cout << "Terminated" << endl;

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	return 0;
}

void* entry_pt(void* data)
{
	annealer_thread* ptr = static_cast<annealer_thread*>(data);
	ptr->Run();
	return NULL;
}
