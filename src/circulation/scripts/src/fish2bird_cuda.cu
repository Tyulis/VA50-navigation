#include <cuda.h>



// FIXME : alignments ?
__global__ void ker_target_to_image(int num_points, double* target_points, double* image_points, double* target_to_camera, double* camera_to_image, double xi) {
	__shared__ double sh_target_to_camera[4][4];
	__shared__ double sh_camera_to_image[3][3];

	// Copy the transforms to shared memory
	if (threadIdx.x < 16)
		sh_target_to_camera[threadIdx.x >> 2][threadIdx.x & 0x03] = target_to_camera[threadIdx.x];
	else if (threadIdx.x < 25)
		sh_camera_to_image[(threadIdx.x - 16) / 3][(threadIdx.x - 16) % 3] = camera_to_image[threadIdx.x - 16];
	__syncthreads();

	int point_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (point_index < num_points) {
		// Those arrays are in row-major order, so those accesses should be coalesced across the warp
		double target_x = target_points[0*num_points + point_index];
		double target_y = target_points[1*num_points + point_index];
		double target_z = target_points[2*num_points + point_index];

		// Matrix product, target_to_camera @ (target_x, target_y, target_z) -> camera frame
		double camera_x = target_x*sh_target_to_camera[0][0] + target_y*sh_target_to_camera[0][1] + target_z*sh_target_to_camera[0][2] + sh_target_to_camera[0][3];
		double camera_y = target_x*sh_target_to_camera[1][0] + target_y*sh_target_to_camera[1][1] + target_z*sh_target_to_camera[1][2] + sh_target_to_camera[1][3];
		double camera_z = target_x*sh_target_to_camera[2][0] + target_y*sh_target_to_camera[2][1] + target_z*sh_target_to_camera[2][2] + sh_target_to_camera[2][3];
		double camera_norm = rnorm3d(camera_x, camera_y, camera_z);

		// Project on the unit sphere then on the sensor plane
		double sphere_biased_normalized_z = 1 / (camera_z * camera_norm + xi) * camera_norm;
		double projected_x = camera_x * sphere_biased_normalized_z;
		double projected_y = camera_y * sphere_biased_normalized_z;

		// Convert to pixel coordinates (matrix product, camera_to_image @ (projected_x, projected_y))
		image_points[0*num_points + point_index] = projected_x*sh_camera_to_image[0][0] + projected_y*sh_camera_to_image[0][1] + sh_camera_to_image[0][2];
		image_points[1*num_points + point_index] = projected_x*sh_camera_to_image[1][0] + projected_y*sh_camera_to_image[1][1] + sh_camera_to_image[1][2];
	}
}


#ifdef __cplusplus
extern "C" {
#endif

void _target_to_image_cuda(int num_points, double* target_points, double* image_points, double* target_to_camera, double* camera_to_image, double xi) {
	// Allocate and copy the data to the device
	double* dev_buffer;
	cudaMalloc((void**)&dev_buffer, sizeof(double) * (3*num_points + 2*num_points + 4*4 + 3*3));
	
	double* dev_target_points = dev_buffer;
	double* dev_image_points = &dev_target_points[3*num_points];
	double* dev_target_to_camera = &dev_image_points[2*num_points];
	double* dev_camera_to_image = &dev_image_points[4*4];

	cudaMemcpy(dev_target_points, target_points, sizeof(double) * 3*num_points, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_target_to_camera, target_to_camera, sizeof(double) * 4*4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_camera_to_image, camera_to_image, sizeof(double) * 3*3, cudaMemcpyHostToDevice);

	// And launch the kernel with the right dimensions
	int num_blocks = num_points/256 + ((num_points % 256 > 0)? 1 : 0);
	ker_target_to_image<<<num_blocks, 256>>>(num_points, dev_target_points, dev_image_points, dev_target_to_camera, dev_camera_to_image, xi);

	// Free the resources and return
	cudaMemcpy(image_points, dev_image_points, sizeof(double) * 2*num_points, cudaMemcpyDeviceToHost);
	cudaFree(dev_buffer);
}

#ifdef __cplusplus
}
#endif