#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <memory>

#include <driver_types.h>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/timer/timer.hpp>

#include <exceptions.h>
#include <gpu_kernels.h>
#include <my_log.h>

namespace po = boost::program_options;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static block_descr_t get_block_size(size_t threads_per_block)
{
	block_descr_t bd;
	bd.x = 32;
	bd.y = threads_per_block / 32;
	return bd;
}

int main (int argc, char** argv)
{
	int option_index = 0;
	int c;
	cudaError_t err;
	int bypass_error = 0;
	struct cudaDeviceProp cuda_dev_prop;
	cudaEvent_t process_start, process_stop;

	cudaEventCreate(&process_start);
	cudaEventCreate(&process_stop);

    try
    {
		po::options_description desc("Allowed options");

		desc.add_options()
				("help", "produce help message")
				("duration,D", po::value<unsigned int>()->default_value(30), "test duration in seconds")
				("image,I", po::value<std::string>(), "input image file name")
				("output,O", po::value<std::string>(), "output image file name");

		po::variables_map vars;
		po::store(po::parse_command_line(argc, argv, desc), vars);
		if(vars.size() <= 1 || vars.count("help"))
		{
			std::cout << desc << std::endl;
			return 1;
		}

		if(!vars.count("image"))
			THROW("'image' parameter is not set");
		if(!vars.count("output"))
			THROW("'output' parameter is not set");

		unsigned int test_duration = vars["duration"].as<unsigned int>();
		std::string image_filename = vars["image"].as<std::string>();
		std::string output_filename = vars["output"].as<std::string>();

    	float w_interval = 0, af_interval = 0, r_interval = 0, t_interval = 0, f_interval = 0;

    	int iterations = 0;
    	void* dev_image_in;
    	void* dev_image_out;
    	int dev_number;

    	std::ifstream fin(image_filename, std::ios::binary);
    	std::cout << "Input file: " << image_filename << std::endl;

    	std::vector<char> buffer(
    			(std::istreambuf_iterator<char>(fin)),
				(std::istreambuf_iterator<char>()));

    	std::cout << "Image size - " << buffer.size() << " bytes" << "\n";

    	std::unique_ptr<char[]> output(new char(buffer.size()));

    	// ------- Explore CUDA device -------
    	gpuErrchk(cudaGetDeviceCount(&dev_number));
    	std::cout << "CUDA devices available - " << dev_number << std::endl;

    	gpuErrchk(cudaGetDeviceProperties(&cuda_dev_prop, 0));
    	std::cout << "CUDA device name: "				<< cuda_dev_prop.name				<< std::endl;
    	std::cout << "CUDA device total global mem: "	<< cuda_dev_prop.totalGlobalMem		<< std::endl;
    	std::cout << "CUDA device sharedMemPerBlock: "	<< cuda_dev_prop.sharedMemPerBlock	<< std::endl;
    	std::cout << "CUDA device maxThreadsPerBlock: "	<< cuda_dev_prop.maxThreadsPerBlock	<< std::endl;
    	std::cout << "CUDA device maxThreadsDim[0]: "	<< cuda_dev_prop.maxThreadsDim[0]	<< std::endl;
    	std::cout << "CUDA device maxThreadsDim[1]: "	<< cuda_dev_prop.maxThreadsDim[1]	<< std::endl;
    	std::cout << "CUDA device maxThreadsDim[2]: "	<< cuda_dev_prop.maxThreadsDim[2]	<< std::endl;
    	std::cout << "CUDA device maxGridSize[0]: "		<< cuda_dev_prop.maxGridSize[0]		<< std::endl;
    	std::cout << "CUDA device maxGridSize[1]: "		<< cuda_dev_prop.maxGridSize[1]		<< std::endl;
    	std::cout << "CUDA device maxGridSize[2]: "		<< cuda_dev_prop.maxGridSize[2]		<< std::endl;
    	std::cout << "CUDA device pciBusID: "			<< cuda_dev_prop.pciBusID			<< std::endl;

    	// -----------------------------------

    	gpuErrchk(cudaMalloc(&dev_image_in, buffer.size()));
    	gpuErrchk(cudaMalloc(&dev_image_out, buffer.size()));
    	gpuErrchk(cudaMemset(dev_image_out, 0, buffer.size()));

		gpuErrchk(cudaMemcpy(dev_image_in, &buffer[0], buffer.size(), cudaMemcpyHostToDevice));
		cudaStreamSynchronize(0);

		block_descr_t bd = get_block_size(cuda_dev_prop.maxThreadsPerBlock);

		// ----- Process image on the GPU -----
		{
			boost::timer::cpu_timer operation_time;
			int iterations = 0;
			float pure_time_ms = 0;

			while ((float) operation_time.elapsed().wall / 1000000 < test_duration * 1000)	// 30 sec
			{
				if (0 == iterations++ % 1000)
					LOG_MSG("Iteration %d is in progress, test time %2.1f sec\n", iterations, (float) operation_time.elapsed().wall / (1000 * 1000 * 1000));
				boost::timer::cpu_timer pure_time;

		   		cuda_affine((unsigned short*) dev_image_in,
		   					(unsigned short*) dev_image_out,
							bd);

		   		cudaStreamSynchronize(0);
		   		pure_time_ms += ((float) pure_time.elapsed().wall) / 1000000;
			}
			MSG("Single kernel execution time %2.2f ms", pure_time_ms / iterations);
			MSG("Iterations passed %d", iterations);
		}

		gpuErrchk(cudaMemcpy(&buffer[0], dev_image_out, buffer.size(), cudaMemcpyDeviceToHost));
		cudaStreamSynchronize(0);

		{
			std::ofstream fout(output_filename, std::ios::binary); // | std::ios_base::out);
			fout.write(&buffer[0], buffer.size());
			fout.close();
		}

    	gpuErrchk(cudaFree(dev_image_out));
    	gpuErrchk(cudaFree(dev_image_in));

    	std::cout << "Success" << "\n";
    }
	catch (std::exception const& e)
	{
		std::cout << "Exception: " << e.what() << "\n";
	}
}
