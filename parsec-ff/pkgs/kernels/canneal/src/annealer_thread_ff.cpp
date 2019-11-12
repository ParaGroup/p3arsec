// annealer_thread.cpp
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
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

#ifdef ENABLE_NORNIR
#include <nornir/instrumenter.hpp>
#include <iostream>
extern nornir::Instrumenter* instr;
#endif // ENABLE_NORNIR

#include <cassert>
#include "annealer_thread_ff.h"
#include "location_t.h"
#include "annealer_types.h"
#include "netlist_elem.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "rng.h"

using std::cout;
using std::endl;

static CTask dummyTask, *allTasks;

void* annealer_thread::svc(void* task) {
    T = T / 1.5;
    int accepted_good_moves = 0;
    int accepted_bad_moves = 0;
    for (int i = 0; i < _moves_per_thread_temp; i++) {
#ifdef ENABLE_NORNIR
        instr->begin(get_my_id());
#endif // ENABLE_NORNIR
        a = b;
        a_id = b_id;
        b = _netlist->get_random_element(&b_id, a_id, &rng);
        routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
        move_decision_t is_good_move = accept_move(delta_cost, T, &rng);
        if (is_good_move == move_decision_accepted_bad) {
            accepted_bad_moves++;
            _netlist->swap_locations(a,b);
        }else if (is_good_move == move_decision_accepted_good) {
            accepted_good_moves++;
            _netlist->swap_locations(a,b);
        }else if (is_good_move == move_decision_rejected) {
            ;
        }
#ifdef ENABLE_NORNIR
        instr->end(get_my_id());
#endif // ENABLE_NORNIR
    }
    allTasks[get_my_id()].accepted_good_moves = accepted_good_moves;
    allTasks[get_my_id()].accepted_bad_moves = accepted_bad_moves;
    return &dummyTask;
}

Emitter::Emitter(uint maxNumWorkers, int number_temp_steps, ff::ff_loadbalancer* lb):
        temp_steps_completed(0), activeWorkers(maxNumWorkers), tasksRcvd(0),
        _number_temp_steps(number_temp_steps), _keep_going_global_flag(true), lb(lb), _firstRun(true) {
    allTasks = new CTask[maxNumWorkers];
}

void* Emitter::svc(void*) {
    if(_firstRun){
        lb->broadcast_task((void*) &dummyTask);
        _firstRun = false;
    }else if(++tasksRcvd == activeWorkers) {
        tasksRcvd = 0;
        ++temp_steps_completed;
        for(size_t i = 0; i < activeWorkers; i++) {
            if(!keep_going(allTasks[i].accepted_good_moves, allTasks[i].accepted_bad_moves)) {
                return EOS;
            }
        }
        lb->broadcast_task((void*) &dummyTask);
    }
    return GO_ON;
}

annealer_thread::move_decision_t annealer_thread::accept_move(routing_cost_t delta_cost, double T, Rng* rng) {
    if (delta_cost < 0) {
        return move_decision_accepted_good;
    }else {
        double random_value = rng->drand();
        double boltzman = exp(- delta_cost/T);
        if (boltzman > random_value) {
            return move_decision_accepted_bad;
        }else{
            return move_decision_rejected;
        }
    }
}

routing_cost_t annealer_thread::calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b) {
    location_t* a_loc = a->present_loc.Get();
    location_t* b_loc = b->present_loc.Get();
    routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
    delta_cost += b->swap_cost(b_loc, a_loc);
    return delta_cost;
}

bool Emitter::keep_going(int accepted_good_moves, int accepted_bad_moves) {
    bool rv;
    if(_number_temp_steps == -1){
        rv = _keep_going_global_flag && (accepted_good_moves > accepted_bad_moves);
        if(!rv) _keep_going_global_flag = false; // signal we have converged
    }else{
        rv = temp_steps_completed < _number_temp_steps;
    }
    return rv;
}
