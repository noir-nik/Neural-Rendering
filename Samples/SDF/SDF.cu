#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

#include "SDF.cuh"

#define N 128

__global__ void add(int* a, int* b) {
	int i = blockIdx.x;
	if (i < N) {
		b[i] = 2 * a[i];
	}
}

int run() {
	return 0;
}
