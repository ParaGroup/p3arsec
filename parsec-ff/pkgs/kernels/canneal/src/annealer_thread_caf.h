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

#ifndef ANNEALER_THREAD_CAF_H
#define ANNEALER_THREAD_CAF_H

#include <array>
#include <assert.h>

#include "annealer_types.h"
#include "caf/all.hpp"
#include "netlist.h"
#include "netlist_elem.h"
#include "rng.h"

using namespace std;

struct CTask {
    int accepted_good_moves;
    int accepted_bad_moves;
    CTask() : accepted_good_moves(0), accepted_bad_moves(-1) {}
};
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(CTask);

enum move_decision_t {
    move_decision_accepted_good,
    move_decision_accepted_bad,
    move_decision_rejected
};

move_decision_t accept_move(routing_cost_t delta_cost, double T, Rng *rng) {
    if (delta_cost < 0) {
        return move_decision_accepted_good;
    } else {
        double random_value = rng->drand();
        double boltzman = exp(-delta_cost / T);
        if (boltzman > random_value) {
            return move_decision_accepted_bad;
        } else {
            return move_decision_rejected;
        }
    }
}

routing_cost_t calculate_delta_routing_cost(netlist_elem *a, netlist_elem *b) {
    location_t *a_loc = a->present_loc.Get();
    location_t *b_loc = b->present_loc.Get();
    routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
    delta_cost += b->swap_cost(b_loc, a_loc);
    return delta_cost;
}

struct worker_state {
    netlist *netlist_;
    double T;
    Rng rng; // store of randomness
    long a_id;
    long b_id;
    netlist_elem *a;
    netlist_elem *b;
};
caf::behavior worker_actor(caf::stateful_actor<worker_state> *self,
                           netlist *netlist_, int nthreads, int swaps_per_temp,
                           double start_temp) {
    // init internal state
    int _moves_per_thread_temp = swaps_per_temp / nthreads;
    self->state.netlist_ = netlist_;
    self->state.T = start_temp;
    self->state.a_id = 0;
    self->state.b_id = 0;
    assert(netlist_ != NULL);
    self->state.a = netlist_->get_random_element(
        &self->state.a_id, NO_MATCHING_ELEMENT, &self->state.rng);
    self->state.b = netlist_->get_random_element(
        &self->state.b_id, NO_MATCHING_ELEMENT, &self->state.rng);
    
    // declare handler
    return {[=](uint i, CTask &task) {
        self->state.T /= 1.5;
        int accepted_good_moves = 0;
        int accepted_bad_moves = 0;
        for (int i = 0; i < _moves_per_thread_temp; i++) {
            self->state.a = self->state.b;
            self->state.a_id = self->state.b_id;
            self->state.b = self->state.netlist_->get_random_element(
                &self->state.b_id, self->state.a_id, &self->state.rng);
            routing_cost_t delta_cost =
                calculate_delta_routing_cost(self->state.a, self->state.b);
            move_decision_t is_good_move =
                accept_move(delta_cost, self->state.T, &self->state.rng);
            if (is_good_move == move_decision_accepted_bad) {
                accepted_bad_moves++;
                self->state.netlist_->swap_locations(self->state.a,
                                                     self->state.b);
            } else if (is_good_move == move_decision_accepted_good) {
                accepted_good_moves++;
                self->state.netlist_->swap_locations(self->state.a,
                                                     self->state.b);
            } else if (is_good_move == move_decision_rejected) {
                ;
            }
        }
        // update and send back the task
        task.accepted_good_moves = accepted_good_moves;
        task.accepted_bad_moves = accepted_bad_moves;
        auto sender = caf::actor_cast<caf::actor>(self->current_sender());
        self->send(sender, i, move(task));
    }};
}

bool keep_going(CTask task, int temp_steps_completed, int number_temp_steps) {
    bool rv;
    if (number_temp_steps == -1)
        rv = task.accepted_good_moves > task.accepted_bad_moves;
    else
        rv = temp_steps_completed < number_temp_steps;
    return rv;
}

struct master_state {
    uint n_res;
    int temp_steps_completed;
    vector<CTask> tasks;
};
caf::behavior master_actor(caf::stateful_actor<master_state> *self, uint nw,
                           int number_temp_steps, netlist *netlist_,
                           int swaps_per_temp, double start_temp) {
    // init state
    self->state.n_res = nw;
    self->state.temp_steps_completed = 0;
    self->state.tasks = vector<CTask>(nw);
    // spawn workers
    auto spawn_worker = [=]() -> caf::actor {
#ifdef DETACHED_WORKER
        return self->spawn<caf::detached>(worker_actor, netlist_, nw,
                                          swaps_per_temp, start_temp);
#else
        return self->spawn(worker_actor, netlist_, nw,
                           swaps_per_temp, start_temp);
#endif
    };
    auto workers = caf::actor_pool::make(self->context(), nw, spawn_worker,
                                         caf::actor_pool::round_robin());
    // send initial work
    for (uint i = 0; i < nw; i++) {
        self->send(workers, i, move(CTask()));
    }
    // handler worker responce
    return {[=](uint iw, CTask &task) {
        self->state.tasks[iw] = move(task);
        if (--self->state.n_res == 0) {
            ++self->state.temp_steps_completed;
            // caf::aout(self) << "DEBUG: iteration "
            //                 << self->state.temp_steps_completed
            //                 << " ended" << endl;
            for (uint i = 0; i < nw; i++) {
                if (!keep_going(self->state.tasks[i],
                                self->state.temp_steps_completed,
                                number_temp_steps)) {
                    std::cout << "QUIT" << std::endl;
                    self->quit();
                    return;
                }
            }
            self->state.n_res = nw;
            for (uint i = 0; i < nw; i++) {
                self->send(workers, i, self->state.tasks[i]);
            }
        }
    }};
}

#endif // ANNEALER_THREAD_CAF_H
