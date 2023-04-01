#ifndef _FISH2BIRD_CUDA_H
#define _FISH2BIRD_CUDA_H

void _target_to_image_cuda(int num_points, double* target_points, double* image_points, double* target_to_camera, double* camera_to_image, double xi);

#endif