//#####################################################################
// Copyright 2004-2006, Ron Fedkiw, Andrew Selle, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
// Class SOLIDS_FLUIDS_DRIVER
//#####################################################################
#include "SOLIDS_FLUIDS_DRIVER.h"
#include "../Utilities/LOG.h"
#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif
#ifdef ENABLE_NORNIR
#include <instrumenter.hpp>
#include <stdlib.h>
#include <iostream>
std::string getParametersPath(){
    return std::string(getenv("PARSECDIR")) + std::string("/parameters.xml");
}
#endif //ENABLE_NORNIR
using namespace PhysBAM;
//#####################################################################
// Function Execute_Main_Program
//#####################################################################
template<class T> void SOLIDS_FLUIDS_DRIVER<T>::
Execute_Main_Program()
{
	Initialize();

	if (!example.restart && example.write_output_files && !example.write_substeps) Write_Output_Files (example.first_frame);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif
	Simulate_To_Frame (example.last_frame);
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

	//Always write last frame for verification purposes
	if (!example.write_output_files && !example.write_substeps) Write_Output_Files (example.last_frame);

	LOG::Dump_Log();
}
//#####################################################################
// Function Initialize
//#####################################################################
template<class T> void SOLIDS_FLUIDS_DRIVER<T>::
Initialize()
{
	if (example.restart)
	{
		current_frame = example.restart_frame;
		Read_Time (current_frame);
	}
	else current_frame = example.first_frame;

	output_number = current_frame;
	time = Time_At_Frame (current_frame);
}
//#####################################################################
// Function Simulate_To_Frame
//#####################################################################
template<class T> void SOLIDS_FLUIDS_DRIVER<T>::
Simulate_To_Frame (const int frame_input)
{
#ifdef ENABLE_NORNIR
	nornir::Instrumenter instr(getParametersPath(), 1, NULL, true);
#endif //ENABLE_NORNIR
	while (current_frame < frame_input)
	{
#ifdef ENABLE_NORNIR
		instr.begin();
#endif //ENABLE_NORNIR
		LOG::Push_Scope ("FRAME", "Frame %d", current_frame + 1);
		Preprocess_Frame (current_frame + 1);
		Advance_To_Target_Time (Time_At_Frame (current_frame + 1));
		Postprocess_Frame (++current_frame);

		if (example.write_output_files && !example.write_substeps) Write_Output_Files (current_frame);

		if (example.verbose) std::cout << "TIME = " << time << std::endl;

		LOG::Pop_Scope();
		/*
		        std::cout<<"Printing log..."<<std::endl;
		        FILE_UTILITIES::Create_Directory(example.output_directory+"/xml");
		        std::ostream* output=FILE_UTILITIES::Safe_Open_Output(STRING_UTILITIES::string_sprintf("%s/xml/timing_xml.%d",example.output_directory.c_str(),current_frame));
		        LOG::Dump_Log_XML(*output);
		        delete output;
		*/
#ifdef ENABLE_NORNIR
		instr.end();
#endif //ENABLE_NORNIR
	}
#ifdef ENABLE_NORNIR
	instr.terminate();
	std::cout << "riff.time|" << instr.getExecutionTime() << std::endl;
    std::cout << "riff.iterations|" << instr.getTotalTasks() << std::endl;
#endif //ENABLE_NORNIR
}
//#####################################################################

template class SOLIDS_FLUIDS_DRIVER<float>;
template class SOLIDS_FLUIDS_DRIVER<double>;
