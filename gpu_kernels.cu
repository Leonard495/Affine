#include <gpu_kernels.h>

#define PI     3.14159265359f
#define WHITE  (short)(1)
#define BLACK  (short)(0)

#define X_SIZE 512
#define Y_SIZE 512

__global__ void affine_kernel(
	unsigned short* image1,
	unsigned short* image2)
{
	float lx_rot = 1.0;		// 30.0
	float ly_rot = -1.0;	// 0.0;
	float lx_expan = 1.0;
	float ly_expan = 1.0;
	
	float affine[2][2];
	float i_affine[2][2];
	float det;
	float x_new, y_new;
	float x_frac, y_frac;
	float gray_new;
	int   m, n;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	// Forward affine transformation
	affine[0][0] = lx_expan * cos((float)(lx_rot*PI/180.0f));
	affine[0][1] = ly_expan * sin((float)(ly_rot*PI/180.0f));
	affine[1][0] = lx_expan * sin((float)(lx_rot*PI/180.0f));
	affine[1][1] = ly_expan * cos((float)(ly_rot*PI/180.0f));
	
	// determination of inverse affine transformation
	det = (affine[0][0] * affine[1][1]) - (affine[0][1] * affine[1][0]);
	if (det == 0.0f)
	{
    	i_affine[0][0]   = 1.0f;
		i_affine[0][1]   = 0.0f;
		i_affine[1][0]   = 0.0f;
		i_affine[1][1]   = 1.0f;
	} 
	else 
	{
		i_affine[0][0]   =  affine[1][1]/det;
		i_affine[0][1]   = -affine[0][1]/det;
		i_affine[1][0]   = -affine[1][0]/det;
		i_affine[1][1]   =  affine[0][0]/det;
	}
	
	   // Output image generation by inverse affine transformation and bilinear transformation
   //for (y = threadIdx.y; y < Y_SIZE; y += blockDim.y)

	for (int iterations = 0; iterations < N; ++iterations)
	{
		int y = iterations * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
		if (y < Y_SIZE)
		{
		  unsigned short* output_buffer = (unsigned short*) &image2[y * X_SIZE];

		  //for (x = threadIdx.x; x < X_SIZE; x += blockDim.x)
		  if (x < X_SIZE)
		  {
			 x_new    = i_affine[0][0]*(x-X_SIZE/2.0f) + i_affine[0][1]*(y-Y_SIZE/2.0f) + X_SIZE/2.0f;
			 y_new    = i_affine[1][0]*(x-X_SIZE/2.0f) + i_affine[1][1]*(y-Y_SIZE/2.0f) + Y_SIZE/2.0f;

			 m        = (int)floor(x_new);
			 n        = (int)floor(y_new);

			 x_frac   = x_new - m;
			 y_frac   = y_new - n;

			 if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n+1 < Y_SIZE))
			 {
				gray_new = (1.0f - y_frac) * ((1.0f - x_frac) * (image1[(n * X_SIZE) + m])       + x_frac * (image1[(n * X_SIZE) + m + 1])) +
								   y_frac  * ((1.0f - x_frac) * (image1[((n + 1) * X_SIZE) + m]) + x_frac * (image1[((n + 1) * X_SIZE) + m + 1]));

				output_buffer[x] = (unsigned short) gray_new;
			 }
			 else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) || ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE)))
			 {
				output_buffer[x] = image1[(n * X_SIZE) + m];
			 }
			 else
			 {
				output_buffer[x] = 0;
			 }
		  }
	   }
	}
}

void cuda_affine(
    unsigned short* image_in,
	unsigned short* image_out,
	block_descr_t block_descr)
{
	dim3 threads_num (block_descr.x, block_descr.y);
	dim3 block_nmb (X_SIZE / block_descr.x + 1, Y_SIZE / (N * block_descr.y) + 1);
	affine_kernel <<< block_nmb, threads_num, 0, 0 >>>  (image_in, image_out);
}
