#pragma once

typedef struct {
	int x;
	int y;
} block_descr_t;

enum {N = 20};

void cuda_affine(unsigned short* dev_image_in,
			  	     unsigned short* dev_image_out,
					 block_descr_t block_descr);
