typedef struct
{
	unsigned int X;
	unsigned int Y;
} ResultPoint;

extern "C"
{
	__global__ void Multiply(const int N, ResultPoint* __restrict results, ResultPoint* __restrict results2, int factor)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
		{
			(results + i)->Y *= factor;
			(results2 + i)->Y += factor;
		}
	}
}