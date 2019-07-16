// Compile: nvcc -o clcc.exe clcc.c -lOpenCL
// Tool for debug OpenCL kernels. Example usage: clcc test.cl
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
	cl_int err;
	cl_platform_id platform;
	if (argc < 2)
	{
		printf("Usage: ./compile source [DEVICE]\n");
		return 1;
	}	
	char *filename = argv[1];
	int index = 0;
	if (argc > 2)
		index = atoi(argv[2]);
	clGetPlatformIDs(1, &platform, NULL);
	cl_uint num;
	cl_device_id devices[3];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 3, devices, &num);
	if (index >= num)
	{
		printf("Invalid device index (%d).\n", index);
		return 1;
	}
	cl_device_id device = devices[index];
	char name[256];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 256, name, NULL);
	printf("Using device: %s\n", name);
	cl_device_fp_config cfg;
	clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
	printf("Double FP config = %llu\n", cfg);
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	FILE *file = fopen(filename, "r");
	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	rewind(file);
	char *source = (char*) malloc(size + 1);
	source[size] = '\0';
	fread(source, sizeof(char), size, file);
	fclose(file);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &size, NULL);
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if (err == CL_SUCCESS)
	{
		printf("Program built successfully.\n");
	}
	else
	{
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t sz;
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
			char *log = malloc(++sz);
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
			printf("%s\n", log);
			free(log);
		}
		else
		{
			printf("Other build error\n");
		}
	}
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}