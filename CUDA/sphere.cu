//nvcc -o sphere.exe sphere.cu 
//Program generates image where is visible procedural lit sphere.
//Image is saved to PPM file (then it can be read by GIMP, for example).
//Copy ".\helper_math.h" from CUDA SDK path "Samples\common\inc"
//Code written by Przemyslaw Zaworski, 2017

#include <iostream>
#include <cuda_runtime.h>
#include ".\helper_math.h"

#define width 1024  //image width
#define height 1024 //image height

__device__ float sphere (float3 p,float3 c,float r) //sphere distance field function
{
	return length (p-c)-r;
}

__device__ float map (float3 p) //make geometry
{
	return sphere (p,make_float3(0.0,0.0,0.0),1.0);
}

__device__ float3 set_normal (float3 p) //make normal vector
{
	float3 x = make_float3 (0.001f,0.000f,0.000f);
	float3 y = make_float3 (0.000f,0.001f,0.000f);
	float3 z = make_float3 (0.000f,0.000f,0.001f);
	return normalize(make_float3(map(p+x)-map(p-x),map(p+y)-map(p-y),map(p+z)-map(p-z))); 
}

__device__ float3 lighting (float3 p) //make Lambert light model
{
	float3 AmbientLight = make_float3 (0.1f,0.1f,0.1f);
	float3 LightDirection = normalize(make_float3(4.0f,10.0f,-10.0f));
	float3 LightColor = make_float3 (0.9f,0.9f,0.9f);
	float3 NormalDirection = set_normal(p);
	float3 DiffuseColor = clamp(dot(LightDirection,NormalDirection),0.0f,1.0f)*LightColor+AmbientLight;
	return clamp(DiffuseColor,0.0f,1.0f);
}

__device__ float4 raymarch (float3 ro,float3 rd) //raymarching
{
	for (int i=0;i<128;i++)
	{
		float t = map(ro);
		if (t<0.01) return make_float4(lighting(ro),1.0);
		ro+=t*rd;
	}
	return make_float4(0.0,0.0,0.0,1.0);
}

__global__ void rendering(float3 *color) //scene rendering
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x; 	
	float2 resolution = make_float2 ((float)width,(float)height);  
	float2 coordinates = make_float2 ((float)x,(float)y);
	float2 p = (2.0*coordinates-resolution)/resolution.y;
	float3 ro = make_float3 (0.0f,0.0f,-4.0f);
	float3 rd = normalize(make_float3(p,2.0f));
	float4 c = raymarch(ro,rd);
	color[i] = make_float3(c.x,c.y,c.z);
}

int to_int(float x)  //convert from 0..1 to 0..255
{ 
	return int(x*255+0.5); 
}  

int main() //main function
{
	float3* host = new float3[width*height]; //pointer to memory on the host (CPU RAM)
	float3* device;   //pointer to memory on the device (GPU VRAM)
	cudaMalloc(&device, width * height * sizeof(float3));  //allocate memory on the GPU VRAM
	dim3 block(8, 8, 1);  //Manage threads 
	dim3 grid(width / block.x, height / block.y, 1); ////Manage block array 
	rendering <<< grid, block >>>(device);  //run kernel 
	cudaMemcpy(host, device, width * height *sizeof(float3), cudaMemcpyDeviceToHost);  //copy result from VRAM to system RAM
	cudaFree(device);  //free GPU VRAM
	FILE *file = fopen("sphere.ppm", "w");  	//open file to save image
	fprintf(file, "P3 %d %d %d ", width, height, 255);  //save PPM file header
	for (int i = 0; i < width*height; i++)  //save pixels
	{
		fprintf(file, "%d %d %d ", to_int(host[i].x),to_int(host[i].y),to_int(host[i].z));
	}
	fclose(file); //close file
	delete[] host; //free CPU RAM
}