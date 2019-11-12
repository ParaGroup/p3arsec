// annealer_thread.h
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
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


#ifndef ANNEALER_THREAD_FF_H
#define ANNEALER_THREAD_FF_H

#include <assert.h>

#include "annealer_types.h"
#include "netlist.h"
#include "netlist_elem.h"
#include "rng.h"
#include <nornir/interface.hpp>

typedef struct CTask{
    int accepted_good_moves;
    int accepted_bad_moves;
    CTask():accepted_good_moves(0), accepted_bad_moves(-1){;}
} CTask;

class Emitter: public nornir::Scheduler<CTask, CTask> {
private:
    int temp_steps_completed;
    uint activeWorkers;
    size_t tasksRcvd;
    int _number_temp_steps;
    bool _keep_going_global_flag;
    bool _firstRun;
protected:
    void notifyRethreading(size_t oldn, size_t newn);
public:
    Emitter(uint maxNumWorkers, int number_temp_steps);
    bool keep_going(int accepted_good_moves, int accepted_bad_moves);
    CTask* schedule(CTask*);
};

class annealer_thread: public nornir::Worker<CTask, CTask>{
public:
    enum move_decision_t {
        move_decision_accepted_good,
        move_decision_accepted_bad,
        move_decision_rejected
    };

    annealer_thread(netlist* netlist, int nthreads, int swaps_per_temp, double start_temp):
        _netlist(netlist), T(start_temp), _swaps_per_temp(swaps_per_temp),
        _moves_per_thread_temp(swaps_per_temp/nthreads),
        a_id(0), b_id(0){
            assert(_netlist != NULL);
            a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng);
            b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng);
    }

    ~annealer_thread(){;}

    CTask* compute(CTask* task);
protected:
    move_decision_t accept_move(routing_cost_t delta_cost, double T, Rng* rng);
    routing_cost_t calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b);
    void notifyRethreading(size_t oldn, size_t newn);

    netlist* _netlist;
    double T;
    int _swaps_per_temp;
    int _moves_per_thread_temp;
    Rng rng; //store of randomness 
    long a_id;
    long b_id;
    netlist_elem* a;
    netlist_elem* b;
};

#endif
