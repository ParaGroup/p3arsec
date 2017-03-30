//-------------------------------------------------------------
//      ____                        _      _
//     / ___|____ _   _ ____   ____| |__  | |
//    | |   / ___| | | |  _  \/ ___|  _  \| |
//    | |___| |  | |_| | | | | |___| | | ||_|
//     \____|_|  \_____|_| |_|\____|_| |_|(_) Media benchmarks
//                  
//	  2006, Intel Corporation, licensed under Apache 2.0 
//
//  file : TrackingModelFF.h
//  author : Scott Ettinger - scott.m.ettinger@intel.com
//  description : Observation model for kinematic tree body 
//				  tracking threaded with OpenMP.
//				  
//  modified : FastFlow version written by Daniele De Sensi
//             (d.desensi.software@gmail.com)
//--------------------------------------------------------------

#ifndef TRACKINGMODELFF_H
#define TRACKINGMODELFF_H

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif

#include <ff/parallel_for.hpp>
#include "TrackingModel.h"
#include <iostream>
#include <iomanip>
#include <sstream>

class TrackingModelFF : public TrackingModel {
	//Generate an edge map from the original camera image - threaded
	void CreateEdgeMap(FlexImage8u &src, FlexImage8u &dst);

public:
	//load and process images - overloaded for future threading
	bool GetObservation(float timeval);

    ff::ParallelFor* getParallelFor();
    void setParallelFor(ff::ParallelFor* pf);
private:
    ff::ParallelFor* _pf;
};

#endif

