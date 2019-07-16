// Compile: nvcc -o clptx.exe clptx.c -lOpenCL
// Tool for generation of PTX assembly from OpenCL C. Example usage: clptx test.cl
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) 
{
	cl_device_id device;
	cl_platform_id platform;
	char *filename = argv[1];
	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	FILE *file = fopen(filename, "r");
	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	rewind(file);
	char *source = (char*) malloc(size + 1);
	source[size] = '\0';
	fread(source, sizeof(char), size, file);
	fclose(file); 	
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
	clBuildProgram(program, 1, &device, "", NULL, NULL);
	size_t bsize;	
	clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bsize, NULL);
	char *binary = (char*)malloc(bsize);
	clGetProgramInfo(program, CL_PROGRAM_BINARIES, bsize, &binary, NULL);
	FILE *f = fopen("output.ptx", "w");
	fwrite(binary, bsize, 1, f);
	fclose(f);   
	return 0;
}