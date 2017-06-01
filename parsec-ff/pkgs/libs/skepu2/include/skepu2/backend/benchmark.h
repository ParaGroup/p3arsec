#pragma once

#include <chrono>

namespace skepu2
{
	namespace containerutils
	{
		inline void updateHostAndInvalidateDevice() {}
		
		template<typename Arg, typename... Args>
		inline void updateHostAndInvalidateDevice(Arg& arg, Args&&... args)
		{
			arg.updateHostAndInvalidateDevice();
			updateHostAndInvalidateDevice(std::forward<Args>(args)...);
		}
		
		inline void reserve(size_t size) {}
		
		template<typename T, typename... Args>
		inline void reserve(size_t size, skepu2::Matrix<T> &arg, Args&&... args);
		
		template<typename T, typename... Args>
		inline void reserve(size_t size, skepu2::Vector<T> &arg, Args&&... args)
		{
			arg.reserve(size);
			reserve(size, std::forward<Args>(args)...);
		}
		
		template<typename T, typename... Args>
		inline void reserve(size_t size, skepu2::Matrix<T> &arg, Args&&... args)
		{
		//	arg.reserve(size);
			reserve(size, std::forward<Args>(args)...);
		}
		
		inline void resize(size_t size) {}
		
		template<typename T, typename... Args>
		inline void resize(size_t size, skepu2::Matrix<T> &arg, Args&&... args);
		
		template<typename T, typename... Args>
		inline void resize(size_t size, skepu2::Vector<T> &arg, Args&&... args)
		{
			arg.resize(size);
			resize(size, std::forward<Args>(args)...);
		}
		
		template<typename T, typename... Args>
		inline void resize(size_t size, skepu2::Matrix<T> &arg, Args&&... args)
		{
			arg.resize(size, size);
			resize(size, std::forward<Args>(args)...);
		}
	}
	
	
	
	namespace benchmark
	{
		using TimeSpan = std::chrono::microseconds;
		
		inline std::tuple<size_t, size_t, size_t>
		parseArgs(int argc, char *argv[], size_t min = 1, size_t max = 1000, size_t reps = 10)
		
		{
			size_t minSize{min}, maxSize{max}, repetitions{reps};
			if (argc == 1)
				std::cerr << "Usage: " << argv[0] << " min-size max-size repetitions\n";
			if (argc > 1)
				minSize = std::stoul(argv[1]);
			if (argc > 2)
				maxSize = std::stoul(argv[2]);
			if (argc > 3)
				repetitions = std::stoul(argv[3]);
			
			return std::make_tuple(minSize, maxSize, repetitions);
		}
		
		class BenchmarkResult
		{
			using Key = std::pair<Backend::Type, size_t>;
			
		public:
			
			BenchmarkResult(std::string benchname): name(benchname)
			{}
			
			void set(Backend::Type type, size_t size, TimeSpan duration)
			{
				Key key = std::make_pair(type, size);
				data[key] = duration;
			}
			
			TimeSpan get(Backend::Type type, size_t size)
			{
				Key key = std::make_pair(type, size);
				return data[key];
			}
			
			void serialize(std::ostream &o = std::cout)
			{
				o << this->name << "\n";
				for (auto &entry : this->data)
				{
					o << entry.first.first << " " << entry.first.second << " " << entry.second.count() << "\n";
				} 
			}
			
		private:
			std::string name;
			std::map<Key, TimeSpan> data;
		};
		
		using BenchmarkFunc = std::function<void(size_t size)>;
		
		template<typename... Args>
		inline std::chrono::microseconds measureExecTime(BenchmarkFunc f, Args&&... args)
		{
			auto t1 = std::chrono::high_resolution_clock::now();
			f(std::forward<Args>(args)...);
			auto t2 = std::chrono::high_resolution_clock::now();
			return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		}
		
		template<typename T>
		inline T median(std::vector<T> &v)
		{
			std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
			return *std::next(v.begin(), v.size() / 2);
		}
		
		
		using Callback = std::function<void(TimeSpan duration)>;
		
		// Runs a skeleton instance with the currently set ExecPlan and BackendSpec ´repeats´ times
		// and returns the median execution time
		inline TimeSpan basicBenchmark(size_t repeats, size_t size, BenchmarkFunc f, Callback callback)
		{
			std::vector<TimeSpan> durations(repeats);
			
			// Run multiple tests to smooth out fluctuations
			for (auto &duration : durations)
			{
				duration = measureExecTime(f, size);
				callback(duration);
			}
			
			// Find median execution time
			return median(durations);
		}
		
		// Returns the next size to run, or ´BENCHMARK_TERMINATE´ to stop the benchmarking
		using UpdateFunc = std::function<void(size_t nextSize)>;
		
		template<class Tuple, size_t...Is>
		inline void set_backend_tuple(Tuple&& tuple, BackendSpec &spec, future_std::index_sequence<Is...>)
		{
			int dummy[] = { 0, ((void)std::get<Is>(std::forward<Tuple>(tuple))->setBackend(spec), 0)... };
		}
		
		template<class Tuple>
		inline void set_backend_tuple(Tuple&& tuple, BackendSpec &spec)
		{
			set_backend_tuple(std::forward<Tuple>(tuple), spec,
				typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
		}
		
		template<class Tuple, size_t...Is>
		inline void reset_backend_tuple(Tuple&& tuple, future_std::index_sequence<Is...>)
		{
			int dummy[] = { 0, ((void)std::get<Is>(std::forward<Tuple>(tuple))->resetBackend(), 0)... };
		}
		
		template<class Tuple>
		inline void reset_backend_tuple(Tuple&& tuple)
		{
			reset_backend_tuple(std::forward<Tuple>(tuple),
				typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
		}
		
		// Run tests for all available backends
		template<typename... Skeletons>
		inline BenchmarkResult fullBenchmark(
			std::string name, std::tuple<Skeletons...> &instances, size_t minSize, size_t maxSize, size_t repeats,
			BenchmarkFunc f, UpdateFunc update, Callback callback)
		{
			skepu2::benchmark::BenchmarkResult result(name);
			const size_t stepFactor = 2;
			
			for (auto backend : Backend::availableTypes())
			{
				BackendSpec spec{backend};
				set_backend_tuple(instances, spec);
				for (size_t size = minSize; size <= maxSize; size *= stepFactor)
				{
					update(size);
					auto duration = basicBenchmark(repeats, size, f, callback);
					result.set(backend, size, duration);
				}
			}
			
			reset_backend_tuple(instances);
			for (size_t size = minSize; size <= maxSize; size *= stepFactor)
			{
				update(size);
				auto duration = basicBenchmark(repeats, size, f, callback);
				result.set(Backend::Type::Auto, size, duration);
			}
			return result;
		}
		
		template<typename Skeleton>
		inline BenchmarkResult fullBenchmark(
			std::string name, Skeleton &instance, size_t minSize, size_t maxSize, size_t repeats,
			BenchmarkFunc f, UpdateFunc update, Callback callback)
		{
			auto tuple = std::make_tuple(&instance);
			return fullBenchmark(name, tuple, minSize, maxSize, repeats, f, update, callback);
		}
		
	}
}
