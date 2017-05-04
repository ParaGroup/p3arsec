#ifndef SKEPU_TIMING_H___
#define SKEPU_TIMING_H___

#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>
#include <vector>
#include <cmath>

#define HAVE_CLOCK_GETTIME
// #define CLOCK_MONOTONIC

namespace skepu2
{
	class Timer
	{
	public:
		void start()
		{
			skepu_clock_gettime(&start_ts);
		}
		
		void stop()
		{
			skepu_clock_gettime(&stop_ts);
			m_data.push_back(skepu_timing_timespec_delay_us(&start_ts, &stop_ts));
		}
		
		double getAverageTime()
		{
			if(m_data.empty())
				return 0;
			
			return (getTotalTime()/m_data.size());
		}
		
		double getTotalTime()
		{
			double totTime = 0.0;
			for(int i=0; i<m_data.size(); ++i)
			{
				totTime += m_data[i];
			}
			return totTime;
		}
		
		Timer()
		{
			skepu_timing_init();
		}
		
		void reset()
		{
			m_data.clear();
		}
		
	private:
		
		void skepu_timing_init(void);
		void skepu_clock_gettime(struct timespec *ts);
		
		std::vector<double> m_data;
		
		struct timespec start_ts;
		struct timespec stop_ts;
		struct timespec skepu_reference_start_time_ts;
		
		/*! computes difference between two times */
		void skepu_timespec_sub(const struct timespec *a, const struct timespec *b, struct timespec *result)
		{
			result->tv_sec = a->tv_sec - b->tv_sec;
			result->tv_nsec = a->tv_nsec - b->tv_nsec;
			
			if ((result)->tv_nsec < 0)
			{
				--(result)->tv_sec;
				result->tv_nsec += 1000000000;
			}
		}
		
		
		/* Returns the time elapsed between start and end in microseconds */
		double skepu_timing_timespec_delay_us(struct timespec *start, struct timespec *end)
		{
			struct timespec diff;
			skepu_timespec_sub(end, start, &diff);
			double us = (diff.tv_sec*1e6) + (diff.tv_nsec*1e-3);
			return us;
		}
		
		
		double skepu_timing_timespec_to_us(struct timespec *ts)
		{
			return (1000000.0*ts->tv_sec) + (0.001*ts->tv_nsec);
		}
		
		
		double skepu_timing_now(void)
		{
			struct timespec now;
			skepu_clock_gettime(&now);
			return skepu_timing_timespec_to_us(&now);
		}
	};
	
	
#if defined(HAVE_CLOCK_GETTIME) && defined(CLOCK_MONOTONIC)

#include <time.h>
#ifndef _POSIX_C_SOURCE
/* for clock_gettime */
#define _POSIX_C_SOURCE 199309L
#endif

#ifdef __linux__
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#endif

	static struct timespec skepu_reference_start_time_ts;

	/* Modern CPUs' clocks are usually not synchronized so we use a monotonic clock
	* to have consistent timing measurements. The CLOCK_MONOTONIC_RAW clock is not
	* subject to NTP adjustments, but is not available on all systems (in that
	* case we use the CLOCK_MONOTONIC clock instead). */
	static inline void skepu_clock_readtime(struct timespec *ts)
	{
#ifdef CLOCK_MONOTONIC_RAW
		static int raw_supported = 0;
		switch (raw_supported)
		{
			case -1:
			break;
			case 1:
			clock_gettime(CLOCK_MONOTONIC_RAW, ts);
			return;
			case 0:
			if (clock_gettime(CLOCK_MONOTONIC_RAW, ts))
			{
				raw_supported = -1;
				break;
			}
			else
			{
				raw_supported = 1;
				return;
			}
		}
#endif
		clock_gettime(CLOCK_MONOTONIC, ts);
	}
	
	
	inline void Timer::skepu_timing_init(void)
	{
		skepu_clock_gettime(&skepu_reference_start_time_ts);
	}
	
	
	inline void Timer::skepu_clock_gettime(struct timespec *ts)
	{
		struct timespec absolute_ts;
		
		/* Read the current time */
		skepu_clock_readtime(&absolute_ts);
		
		/* Compute the relative time since initialization */
		skepu_timespec_sub(&absolute_ts, &skepu_reference_start_time_ts, ts);
	}
	
#else // !HAVE_CLOCK_GETTIME
	
	union skepu_u_tick
	{
		uint64_t tick;
		
		struct
		{
			uint32_t low;
			uint32_t high;
		}
		sub;
	};
	
#define SKEPU_MIN(a,b)	((a)<(b)?(a):(b))
	
#define SKEPU_GET_TICK(t) __asm__ volatile("rdtsc" : "=a" ((t).sub.low), "=d" ((t).sub.high))
#define SKEPU_TICK_RAW_DIFF(t1, t2) ((t2).tick - (t1).tick)
#define SKEPU_TICK_DIFF(t1, t2) (SKEPU_TICK_RAW_DIFF(t1, t2) - skepu_residual)
	
	static union skepu_u_tick skepu_reference_start_tick;
	static double skepu_scale = 0.0;
	static unsigned long long skepu_residual = 0;
	static int skepu_inited = 0;
	
	inline void Timer::skepu_timing_init(void)
	{
		static union skepu_u_tick t1, t2;
		int i;
		
		if (skepu_inited) return;
		
		skepu_residual = (unsigned long long)1 << 63;
		
		for(i = 0; i < 20; i++)
		{
			SKEPU_GET_TICK(t1);
			SKEPU_GET_TICK(t2);
			skepu_residual = SKEPU_MIN(skepu_residual, SKEPU_TICK_RAW_DIFF(t1, t2));
		}
		
		{
			struct timeval tv1,tv2;
			SKEPU_GET_TICK(t1);
			gettimeofday(&tv1,0);
			usleep(500000);
			SKEPU_GET_TICK(t2);
			gettimeofday(&tv2,0);
			skepu_scale = ((tv2.tv_sec*1e6 + tv2.tv_usec) -
				(tv1.tv_sec*1e6 + tv1.tv_usec)) /
					(double)(SKEPU_TICK_DIFF(t1, t2));
		}
		
		SKEPU_GET_TICK(skepu_reference_start_tick);
		
		skepu_inited = 1;
	}
	
	inline void Timer::skepu_clock_gettime(struct timespec *ts)
	{
		union skepu_u_tick tick_now;
		
		SKEPU_GET_TICK(tick_now);
		
		uint64_t elapsed_ticks = SKEPU_TICK_DIFF(skepu_reference_start_tick, tick_now);
		
		/* We convert this number into nano-seconds so that we can fill the
		* timespec structure. */
			uint64_t elapsed_ns = (uint64_t)(((double)elapsed_ticks)*(skepu_scale*1000.0));
		
		long tv_nsec = (elapsed_ns % 1000000000);
		time_t tv_sec = (elapsed_ns / 1000000000);
		
		ts->tv_sec = tv_sec;
		ts->tv_nsec = tv_nsec;
	}
	
#endif // HAVE_CLOCK_GETTIME
	
} // end namespace skepu2

#endif
