// Simple CUDA example that demonstrates using Tensor Cores on an NVIDIA RTX GPU. 
// This example will use CUDA's Warp Matrix Multiply and Accumulate (WMMA).
// nvcc -arch=sm_75 -o TensorCore TensorCore.cu
#include <iostream>
#include <cuda.h>
#include <mma.h>

#define M 16
#define N 16
#define K 16

__global__ void TensorCoreGEMM(half* a, half* b, float* c) 
{
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;
	nvcuda::wmma::load_matrix_sync(a_frag, a, K);
	nvcuda::wmma::load_matrix_sync(b_frag, b, K);
	nvcuda::wmma::fill_fragment(c_frag, 0.0f);
	nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	nvcuda::wmma::store_matrix_sync(c, c_frag, N, nvcuda::wmma::mem_row_major);
}

int main() 
{
	half host_a[M * K];
	half host_b[K * N];
	float host_c[M * N] = {0};
	for (int i = 0; i < M * K; i++) 
	{
		host_a[i] = __float2half(1.0f);
	}
	for (int i = 0; i < K * N; i++) 
	{
		host_b[i] = __float2half(1.0f);
	}
	half *dev_a, *dev_b;
	float *dev_c;
	cudaMalloc((void**)&dev_a, M * K * sizeof(half));
	cudaMalloc((void**)&dev_b, K * N * sizeof(half));
	cudaMalloc((void**)&dev_c, M * N * sizeof(float));
	cudaMemcpy(dev_a, host_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, K * N * sizeof(half), cudaMemcpyHostToDevice);
	TensorCoreGEMM<<<1, 32>>>(dev_a, dev_b, dev_c);
	cudaMemcpy(host_c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);	
	printf("Result matrix C:\n");
	for (int i = 0; i < M; i++) 
	{
		for (int j = 0; j < N; j++) 
		{
			printf("%.0f ", host_c[i * N + j]);
		}
		printf("\n");
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}
