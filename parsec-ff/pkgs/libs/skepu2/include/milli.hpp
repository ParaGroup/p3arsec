// Simple little unit for timing using the gettimeofday() call.
// By Ingemar 2009

#pragma once

#include <stdlib.h>
#include <sys/time.h>

namespace milli
{

	static struct timeval timeStart;
	static char hasStart = 0;
	
	inline int GetMilliseconds()
	{
		struct timeval tv;
		
		gettimeofday(&tv, NULL);
		if (!hasStart)
		{
			hasStart = 1;
			timeStart = tv;
		}
		return (tv.tv_usec - timeStart.tv_usec) / 1000 + (tv.tv_sec - timeStart.tv_sec)*1000;
	}
	
	inline int GetMicroseconds()
	{
		struct timeval tv;
		
		gettimeofday(&tv, NULL);
		if (!hasStart)
		{
			hasStart = 1;
			timeStart = tv;
		}
		return (tv.tv_usec - timeStart.tv_usec) + (tv.tv_sec - timeStart.tv_sec)*1000000;
	}
	
	inline double GetSeconds()
	{
		struct timeval tv;
		
		gettimeofday(&tv, NULL);
		if (!hasStart)
		{
			hasStart = 1;
			timeStart = tv;
		}
		return (double)(tv.tv_usec - timeStart.tv_usec) / 1000000.0 + (double)(tv.tv_sec - timeStart.tv_sec);
	}
	
	// If you want to start from right now.
	inline void Reset()
	{
		struct timeval tv;
		
		gettimeofday(&tv, NULL);
		hasStart = 1;
		timeStart = tv;
	}
	
	// If you want to start from a specific time.
	inline void Set(int seconds, int microseconds)
	{
		hasStart = 1;
		timeStart.tv_sec = seconds;
		timeStart.tv_usec = microseconds;
	}

}