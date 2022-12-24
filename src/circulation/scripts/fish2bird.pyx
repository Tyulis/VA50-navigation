# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

"""
Retrieve the actual 3D position of points from an image
This assumes an omnidirection distortion model as described in the following paper by Christopher Mei :

https://www-sop.inria.fr/icare/personnel/Christopher.Mei/articles/projection_model.pdf

Frame and coordinate systems reference :
 - image  : (u, v)       : input image pixel coordinates
 - input  : (x, y)       : metric plane associated to the image ("sensor frame")
 - sphere : (x, y, z)    : projection on the surface of a unit sphere (Mei distortion model)
 - angles : (θ, φ)       : spheric coordinates angles of the 3D point in the camera frame relative to the center of the unit sphere (θ = colatitude, φ = longitude)
 - camera : (X, Y, Z, 1) : camera frame associated to the input image
 - target : (X, Y, Z, 1) : target plane frame (all target points are assumed to be on a plane where Z=0 in the target frame)
 - output : (u, v)       : output image frame (orthogonal projection along the z axis and scaling)

This module helps retrieving original 3D points from an image.
The original 3D points are assumed to be on a 3D plane
To do that, this modules needs three main parameters :
- camera_to_image  : The K matrix, that projects from metric coordinates to pixel coordinates
					 |αᵤ  0 u₀|
					 | 0 αᵥ v₀|
					 | 0  0  1|
					 camera_to_image @ input_points = image_points
- xi			   : The ξ parameter in Christopher Mei’s paper, that gives the displacement from the unit sphere center for the reprojection
- camera_to_target : A full, homogeneous 4×4 transform matrix, that converts from the camera frame to another 3D frame,
					 where all relevant points in the image are assumed to be on the (X, Y) plane (such that Z = 0)
					 For instance, for the forward camera of a vehicle, this may convert from the camera frame
					 to a frame with its origin point on the road and the Z axis pointing upward,
					 all points are then assumed to be on the road (Z = 0 in the road frame)
					 |\ | / Tx|
					 |- R - Ty|
					 |/ | \ Tz|
					 |0 0 0  1|
					 camera_to_target @ camera_point = target_points, target_points[2] == 0

Main functions :
- to_birdeye(image, camera_to_image, camera_to_target, xi, x_range, y_range, output_size_y, flip_x=False, flip_y=False) -> bird-eye view, scale factor
	Project the given image directly to a bird-eye view, with a simple scale factor to convert back and forth between pixel and metric measurements

- image_to_target(image_points, camera_to_image, camera_to_target, xi) -> target_points
	Convert points from the initial image to the target plane
 
- target_to_output(target_points, x_range, y_range, int output_size_y, bint flip_x=False, bint flip_y=False) -> output_points, scale_factor
	Project points from the target frame to the bird-eye view pixel coordinates, and give the scale factor from pixel to metric coordinates

- birdeye_to_target(points, x_range, y_range, image_shape, flip_x=False, flip_y=False) -> target_points
	Convert bird-eye view pixel coordinates back to 3D target points, assuming Z = 0

- target_to_image(target_points, double[:, :] target_to_camera, double[:, :] camera_to_image, double xi) -> image_points
	Project 3D points from the target frame to image pixel coordinates, like the camera would
"""

import cv2 as cv
import numpy as np

import cython
from numpy cimport uint8_t
from libc.math cimport sqrt, sin, cos, floor, ceil
from cython.parallel cimport prange

# Legacy Python functions, for reference

def get_image_points(image):
	"""Get the [x, y] coordinates of all pixels in the image
	   - image : ndarray[y, x] : source image
	<----------- ndarray[2, N] : Coordinates of all image points as columns (x, y)"""
	y_positions = np.arange(0, image.shape[0], 1)
	x_positions = np.arange(0, image.shape[1], 1)
	x_2d, y_2d = np.meshgrid(x_positions, y_positions)
	x_2d = x_2d.ravel()
	y_2d = y_2d.ravel()
	image_points = np.asarray((x_2d, y_2d))
	return image_points

def image_to_input(image_points, camera_to_image):
	"""Convert from image pixel coordinates to input frame ("sensor") homogeneous coordinates
	   - image_points    : ndarray[2, N] : Pixel coordinates in the image (vectors as columns)
	   - camera_to_image : ndarray[3, 3] : Projection matrix (K) to pixel coordinates
	<--------------------- ndarray[2, N] : Coordinates in the metric plane associated to the image (input frame)"""
	image_homogeneous = np.asarray((image_points[0], image_points[1], np.ones(image_points.shape[1])))
	return (np.linalg.inv(camera_to_image) @ image_homogeneous)[:2]

def input_to_sphere(input_points, xi):
	"""Project from the input ("sensor") plane to the unit sphere
	   - input_points : ndarray[2, N] : Coordinates in the metric plane associated to the image (input frame)
	   - xi           : float         : Mei’s model center shifting parameter (ξ in the paper)
	<------------------ ndarray[3, N] : 3D coordinates of the points’ projection on the unit sphere"""
	projected_x, projected_y = input_points

	# Equations directly taken from Christopher Mei’s paper (h⁻¹(mᵤ) = ...)
	# We assume no other distortion factors so we can go directly from the image plane to the sphere
	return np.asarray((
		projected_x * (xi + np.sqrt(1 + (1-xi**2)*(projected_x**2 + projected_y**2))) / (projected_x**2 + projected_y**2 + 1),
		projected_y * (xi + np.sqrt(1 + (1-xi**2)*(projected_x**2 + projected_y**2))) / (projected_x**2 + projected_y**2 + 1),
		(xi + np.sqrt(1 + (1-xi**2)*(projected_x**2 + projected_y**2))) / (projected_x**2 + projected_y**2 + 1) - xi,
	))

def sphere_to_angles(sphere_points):
	"""Convert 3D cartesian coordinates on the surface of the unit sphere to spheric coordinates angles (colatitude, longitude)
	   - sphere_points : ndarray[3, N] : 3D coordinates on the surface of the unit sphere
	<------------------- ndarray[2, N] : Spheric coordinates angles (colatitude θ, longitude φ)"""

	# We are on a unit sphere, so the radius is 1
	# Those are traditional physics-convention spheric coordinates (radius, colatitude, longitude)
	# Except at this stage we don’t know the actual radius of the original 3D point yet
	return np.asarray((
		np.arccos(sphere_points[2]),					 # Colatitude θ = arccos(z/ρ), ρ = 1
		np.arctan2(sphere_points[1], sphere_points[0]),  # Longitude  φ = atan2(y, x)
	))

def angles_to_camera(angles_points, camera_to_target):
	"""Retrieve the original 3D point in the camera frame from their spheric coordinates angles relative to the camera unit sphere
	   - angles_points    : ndarray[2, N] : Spheric coordinates angles (colatitude θ, longitude φ, the radius is assumed to be 1)
	   - camera_to_target : ndarray[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	<---------------------- ndarray[4, N] : Homogeneous 3D coordinates of the points in the camera frame"""
	colatitude, longitude = angles_points

	# When we convert those points in the camera frame to the target frame (camera_to_target @ camera_points),
	# the resulting points have their Z coordinate equal to 0
	# So according to the matrix product, aX + bY + cZ + d = 0, with [a, b, c, d] the 3rd row of camera_to_target
	# (that, multiplied with a point vector, makes its target Z coordinate)
	# Those are our plane’s parameters in the camera frame
	a, b, c, d = camera_to_target[2]

	# Then we take the system that converts from spheric coordinates to cartesian and add the plane equation,
	# solving it for the radius ρ with the plane equation, then the 3D camera frame coordinates X, Y, Z
	# | X = ρ·sin(θ)·cos(φ)
	# | Y = ρ·sin(θ)·sin(φ)
	# | Z = ρ·cos(θ)
	# | aX + bY + cZ + d = 0
	camera_radius = -d / (a*np.sin(colatitude)*np.cos(longitude) + b*np.sin(colatitude)*np.sin(longitude) + c*np.cos(colatitude))
	camera_x = camera_radius * np.sin(colatitude) * np.cos(longitude)
	camera_y = camera_radius * np.sin(colatitude) * np.sin(longitude)
	camera_z = camera_radius * np.cos(colatitude)
	camera_points = np.asarray((camera_x, camera_y, camera_z, np.ones(angles_points.shape[1])))
	return camera_points

def camera_to_target_(camera_points, camera_to_target):
	"""Convenience function that converts the 3D homogeneous points in the camera frame to the target frame. Only does `camera_to_target @ camera_points`
	   - camera_points    : ndarray[4, N] : Homogeneous 3D coordinates of the points in the camera frame
	   - camera_to_target : ndarray[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	<---------------------- ndarray[4, N] : Homogeneous 3D coordinates of the points in the target frame. All Z coordinates are 0, give or take floating-point shenanigans."""
	target_points = camera_to_target @ camera_points



#### ---------- GAS GAS GAS ---------- ####


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _postprocess_gray(uint8_t[:, :] output_image) noexcept nogil:
	"""Quick-and-dirty bilinear filtering. Unused now"""
	cdef Py_ssize_t width = output_image.shape[1], height = output_image.shape[0]
	cdef Py_ssize_t x, y, inter, last_index = -1

	for x in range(width):
		for y in range(height):
			if output_image[y, x] > 0:
				if last_index >= 0:
					for inter in range(last_index + 1, y):
						output_image[inter, x] = <uint8_t>((1 - <double>(inter - last_index) / (y - last_index)) * output_image[last_index, x] + <double>((inter - last_index) / (y - last_index)) * output_image[y, x])
				last_index = y

	last_index = -1
	for y in range(height):
		for x in range(width):
			if output_image[y, x] > 0:
				if last_index >= 0:
					for inter in range(last_index + 1, x):
						output_image[y, inter] = <uint8_t>((1 - <double>(inter - last_index) / (x - last_index)) * output_image[y, last_index] + <double>((inter - last_index) / (x - last_index)) * output_image[y, x])
				last_index = x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _image_to_target(double[:, :] image_points, double[:, :] target_points, double[:, :] camera_to_image, double[:, :] camera_to_target, double xi) noexcept:
	"""Internal : Convert 2D points from the input camera image to 3D points in the target plane. See `image_to_target`"""
	cdef Py_ssize_t i, npoints = image_points.shape[1]
	
	# Compute the inverse of the camera matrix. As it is an upper-triangular matrix with 1 at [2, 2], a few optimisations can be performed
	cdef double[2][3] inverse_camera_matrix = [[1 / camera_to_image[0, 0], -camera_to_image[0, 1] / (camera_to_image[0, 0] * camera_to_image[1, 1]), (camera_to_image[0, 1]*camera_to_image[1, 2] - camera_to_image[0, 2]*camera_to_image[1, 1]) / (camera_to_image[0, 0]*camera_to_image[1, 1])],
	                                           [0                        , 1 / camera_to_image[1, 1]                                               , -camera_to_image[1, 2] / camera_to_image[1, 1]                                                                                             ]]
	
	cdef double a = camera_to_target[2, 0], b = camera_to_target[2, 1], c = camera_to_target[2, 2], d = camera_to_target[2, 3]
	cdef double projected_x, projected_y, projected_square
	cdef double sphere_factor, sphere_x, sphere_y, sphere_z
	cdef double sin_colatitude, sin_longitude, cos_longitude
	cdef double camera_radius, camera_x, camera_y, camera_z
	for i in range(npoints):
		# Convert image pixel coordinates to metric coordinates on the sensor plane (inverse_camera_matrix @ (x, y))
		projected_x = image_points[0, i] * inverse_camera_matrix[0][0] + image_points[1, i] * inverse_camera_matrix[0][1] + inverse_camera_matrix[0][2]
		projected_y = image_points[0, i] * inverse_camera_matrix[1][0] + image_points[1, i] * inverse_camera_matrix[1][1] + inverse_camera_matrix[1][2]
		projected_square = projected_x**2 + projected_y**2
		
		# Project onto the unit sphere, using the equations in Christopher Mei’s paper
		sphere_factor = (xi + sqrt(1 + (1-xi*xi)*projected_square)) / (projected_square + 1)
		sphere_x = projected_x * sphere_factor
		sphere_y = projected_y * sphere_factor
		sphere_z = sphere_factor - xi

		# Basically, convert cartesian coordinates at the surface of the unit sphere to spheric coordinates
		# But as those are arccos and arctans and we only reuse the sin() and cos() of those values afterwards,
		# we can save some time and complexity by calculating those subresults directly
		sin_colatitude = sqrt(1 - sphere_z*sphere_z)
		sin_longitude = sphere_y / sqrt(sphere_x*sphere_x + sphere_y*sphere_y)
		cos_longitude = sphere_x / sqrt(sphere_x*sphere_x + sphere_y*sphere_y)

		# Solve the projection equations to get the cartesian point in the camera frame using the spheric coordinates and the target plane
		camera_radius = -d / (a*sin_colatitude*cos_longitude + b*sin_colatitude*sin_longitude + c*sphere_z)
		camera_x = camera_radius * sin_colatitude * cos_longitude
		camera_y = camera_radius * sin_colatitude * sin_longitude
		camera_z = camera_radius * sphere_z
		
		# Project to the target frame (matrix product camera_to_target @ (camera_x, camera_y, camera_z))
		# No need to compute target_z as it is assumed to be 0
		target_points[0, i] = camera_to_target[0, 0] * camera_x + camera_to_target[0, 1] * camera_y + camera_to_target[0, 2] * camera_z + camera_to_target[0, 3]
		target_points[1, i] = camera_to_target[1, 0] * camera_x + camera_to_target[1, 1] * camera_y + camera_to_target[1, 2] * camera_z + camera_to_target[1, 3]
		target_points[2, i] = 0
		target_points[3, i] = 1
	

def image_to_target(image_points, double[:, :] camera_to_image, double[:, :] camera_to_target, double xi):
	"""Directly calculate the 3D coordinates in the target frame of the given points in input image pixel coordinates
	   - image_points     : ndarray[2, N] : Pixel coordinates in the image (vectors as columns)
	   - camera_to_image  : ndarray[3, 3] : Projection matrix (K) to pixel coordinates
	   - camera_to_target : ndarray[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	   - xi               : float         : Mei’s model center shifting parameter (ξ in his paper)
	<---------------------- ndarray[4, N] : Homogeneous 3D coordinates of the points in the target frame. All Z coordinates are 0, give or take floating-point shenanigans."""
	target_points = np.empty((4, image_points.shape[1]))
	_image_to_target(image_points.astype(float), target_points, camera_to_image, camera_to_target, xi)
	return target_points

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _target_to_output(double[:, :] target_points, double[:, :] output_points, double x_range_min, double x_range_max, double y_range_min, double y_range_max, int output_height, bint flip_x, bint flip_y) noexcept nogil:
	"""Internal : Convert points from the target plane to 2D bird-eye view pixel coordinates. See `target_to_output`"""
	cdef double scale_factor = output_height / (y_range_max - y_range_min)
	cdef int output_width = <int>((x_range_max - x_range_min) * scale_factor)
	cdef Py_ssize_t i, npoints = target_points.shape[1]
	cdef double output_x, output_y
	# This is just a scaling, and flipping if needed
	for i in range(npoints):
		output_x = (target_points[0, i] - x_range_min) * scale_factor
		output_y = (target_points[1, i] - y_range_min) * scale_factor
		if flip_x:
			output_x = output_width - output_x
		if flip_y:
			output_y = output_height - output_y
		output_points[0, i] = output_x
		output_points[1, i] = output_y
	return 1/scale_factor


def target_to_output(target_points, (double, double) x_range, (double, double) y_range, Py_ssize_t output_size_y, bint flip_x=False, bint flip_y=False):
	"""Make a bird-eye view image from the points in the target frame.
	   Make an orthogonal projection relative to the target Z axis (so parallel to the target plane)
	   Warning : the resulting coordinates are not rounded to integers and may lie outside of the image
	   - target_points : ndarray[4, N] : Homogeneous 3D coordinates of the points in the target frame
	   - x_range       : [xmin, xmax]  : Metric range to display on the image’s x axis (for instance, [-25, 25] makes an image that shows all points within 25 meters on each side of the origin)
	   - y_range       : [ymin, ymax]  : Metric range to display on the image’s y axis (for instance, [0, 50] makes an image that shows all points from the origin to 50 meters forward)
	   - output_size_y : int           : Height of the output image in pixels. The width is calculated accordingly to scale points right
	   - flip_x=False  : bool          : If `True`, reverse the X axis
	   - flip_y=False  : bool          : If `True`, reverse the Y axis
	<------------------- ndarray[2, N] : Pixel coordinates of the given points for the output image described by the parameters
	<------------------- float         : Scale factor from pixels to metric (pixels × scale_factor = metric. If flip_x or flip_y are set, don’t forget to flip it back beforhand, scale_factor * (height-y) or scale_factor * (width-x)
	"""
	output_points = np.empty((2, target_points.shape[1]))
	cdef double scale_factor = _target_to_output(target_points, output_points, <double>x_range[0], <double>x_range[1], <double>y_range[0], <double>y_range[1], output_size_y, flip_x, flip_y)
	return output_points, scale_factor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _to_birdeye(uint8_t[:, ::1] image, uint8_t[:, ::1] birdeye, double[:, :] camera_to_image, double[:, :] target_to_camera, double xi, double x_range_min, double x_range_max, double y_range_min, double y_range_max, int output_size_x, int output_size_y, bint interpolate, bint flip_x, bint flip_y) noexcept nogil:
	"""Internal : Project the input image from the camera into bird-eye view along the (x, y) plane of the target frame. See `to_birdeye`"""
	cdef Py_ssize_t x, y, image_width = image.shape[1], image_height = image.shape[0]
	cdef Py_ssize_t output_x, output_y
	cdef double target_x, target_y
	cdef double camera_x, camera_y, camera_z
	cdef double projected_x, projected_y, projection_norm
	cdef double image_x, image_y
	cdef double scale_factor = (y_range_max - y_range_min) / output_size_y
	cdef double x_factor, y_factor

	# Here the problem is kinda reversed : we take each point of the final bird-eye view,
	# and see which point of the original image it projects onto
	# Then that bird-eye view pixel is assigned the value at that point of the original image, with bilinear interpolation if needed
	for output_y in prange(birdeye.shape[0]):
		for output_x in range(birdeye.shape[1]):
			# Get the metric point on the target plane corresponding to that point
			if flip_x:
				target_x = (output_size_x - output_x) * scale_factor + x_range_min
			else:
				target_x = output_x * scale_factor + x_range_min
			if flip_y:
				target_y = (output_size_y - output_y) * scale_factor + y_range_min
			else:
				target_y = output_y * scale_factor + y_range_min

			# Matrix product, target_to_camera @ (target_x, target_y, target_z) -> camera frame
			camera_z = target_x*target_to_camera[2, 0] + target_y*target_to_camera[2, 1] + target_to_camera[2, 3]
			if camera_z < 0:
				birdeye[output_y, output_x] = 0
				continue
				
			camera_x = target_x*target_to_camera[0, 0] + target_y*target_to_camera[0, 1] + target_to_camera[0, 3]
			camera_y = target_x*target_to_camera[1, 0] + target_y*target_to_camera[1, 1] + target_to_camera[1, 3]
			projection_norm =  1 / (camera_z + xi * sqrt(camera_x*camera_x + camera_y*camera_y + camera_z*camera_z))

			# Project on the sensor plane
			projected_x = camera_x * projection_norm
			projected_y = camera_y * projection_norm

			# Convert to pixel coordinates (matrix product, camera_to_image @ (projected_x, projected_y))
			image_x = projected_x*camera_to_image[0, 0] + projected_y*camera_to_image[0, 1] + camera_to_image[0, 2]
			image_y = projected_x*camera_to_image[1, 0] + projected_y*camera_to_image[1, 1] + camera_to_image[1, 2]
			
			if 0 <= image_x and image_x < image.shape[1] and 0 <= image_y and image_y < image.shape[0]:
				# Without interpolation
				if not interpolate or ceil(image_x) >= image.shape[1] or ceil(image_y) >= image.shape[0] or floor(image_x) == image_x:
					birdeye[output_y, output_x] = image[<Py_ssize_t>floor(image_y), <Py_ssize_t>floor(image_x)]
				# With interpolation
				else:
					x_factor = ceil(image_x) - image_x
					y_factor = ceil(image_y) - image_y
					birdeye[output_y, output_x] = <uint8_t>(     x_factor  * (     y_factor  * image[<Py_ssize_t>floor(image_y), <Py_ssize_t>floor(image_x)] +
					                                                          (1 - y_factor) * image[<Py_ssize_t> ceil(image_y), <Py_ssize_t>floor(image_x)]) +
					                                        (1 - x_factor) * (     y_factor  * image[<Py_ssize_t>floor(image_y), <Py_ssize_t> ceil(image_x)] +
					                                                          (1 - y_factor) * image[<Py_ssize_t> ceil(image_y), <Py_ssize_t> ceil(image_x)]))
			# Outside of the image, no data
			else:
				birdeye[output_y, output_x] = 0
	

def to_birdeye(uint8_t[:, ::1] image, double[:, :] camera_to_image, double[:, :] target_to_camera, double xi, (double, double) x_range, (double, double) y_range, int output_size_y, bint interpolate=True, bint flip_x=False, bint flip_y=False):
	"""Directly creates a bird-eye view from an image
	   - image            : ndarray[y, x] : Original image
	   - camera_to_image  : ndarray[3, 3] : Projection matrix (K) to pixel coordinates
	   - camera_to_target : ndarray[4, 4] : Transform matrix from the camera frame to the target frame where all points’ Z coordinate is null
	   - xi               : float         : Mei’s model center shifting parameter (ξ in his paper)
	   - x_range          : [xmin, xmax]  : Metric range to display on the image’s x axis (for instance, [-25, 25] makes an image that shows all points within 25 meters on each side of the origin)
	   - y_range          : [ymin, ymax]  : Metric range to display on the image’s y axis (for instance, [0, 50] makes an image that shows all points from the origin to 50 meters forward)
	   - output_size_y    : int           : Height of the output image in pixels. The width is calculated accordingly to scale points right
	   - interpolate=True : bool          : If `True, apply bilinear interpolation to bird-eye view pixels, otherwise nearest-neighbor
	   - flip_x=False     : bool          : If `True`, reverse the X axis
	   - flip_y=False     : bool          : If `True`, reverse the Y axis
	<---------------------- ndarray[y, x] : Resulting bird-eye view image (with bilinear interpolation)
	<---------------------- float         : Scale factor from pixels to metric (pixels × scale_factor = metric. If flip_x or flip_y are set, don’t forget to flip it back beforhand, scale_factor * (height-y) or scale_factor * (width-x)"""
	cdef double x_range_min = x_range[0], x_range_max = x_range[1], y_range_min = y_range[0], y_range_max = y_range[1]
	cdef int output_size_x = <int>(output_size_y * (x_range_min - x_range_max) / (y_range_min - y_range_max))
	output_image = np.zeros((output_size_y, output_size_x), dtype=np.uint8)
	_to_birdeye(image, output_image, camera_to_image, target_to_camera, xi, x_range_min, x_range_max, y_range_min, y_range_max, output_size_x, output_size_y, interpolate, flip_x, flip_y)
	return output_image, (y_range_max - y_range_min)/output_size_y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _birdeye_to_target(double[:, :] points, double[:, :] target_points, double x_range_min, double x_range_max, double y_range_min, double y_range_max, int image_height, int image_width, bint flip_x, bint flip_y) noexcept nogil:
	"""Internal : Convert pixel coordinates on the bird-eye view to 3D metric points in the target frame. See `birdeye_to_target`"""
	cdef Py_ssize_t i, npoints = points.shape[1]
	cdef double scale_factor = (y_range_max - y_range_min) / image_height
	cdef double point_x, point_y
	# This is just scaling and flipping when necessary
	for i in range(npoints):
		point_x = points[0, i]
		point_y = points[1, i]
		if flip_x:
			point_x = image_width - point_x
		if flip_y:
			point_y = image_height - point_y

		target_points[0, i] = point_x*scale_factor + x_range_min
		target_points[1, i] = point_y*scale_factor + y_range_min
		target_points[2, i] = 0
		target_points[3, i] = 1


def birdeye_to_target(points, x_range, y_range, image_shape, flip_x=False, flip_y=False):
	"""Convert pixel points on the bird-eye view to the actual 3D point in the target frame, using the parameters given to to_birdeye
	   - points           : ndarray[2, N]   : Pixel coordinates in the bird-eye view to convert
	   - x_range          : [xmin, xmax]    : Metric range displayed on the bird-eye view’s x axis
	   - y_range          : [ymin, ymax]    : Metric range displayed on the bird-eye view’s y axis
	   - output_shape     : [height, width] : Shape of the bird-eye view in pixels
	   - flip_x=False     : bool            : Set to the value given to to_birdeye
	   - flip_y=False     : bool            : Set to the value given to to_birdeye
	<---------------------- ndarray[4, N]   : Corresponding points in the target frame, as homogeneous vectors in columns"""
	target_points = np.empty((4, points.shape[1]))
	_birdeye_to_target(points, target_points, <double>x_range[0], <double>x_range[1], <double>y_range[0], <double>y_range[1], image_shape[0], image_shape[1], flip_x, flip_y)
	return target_points

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _target_to_image(double[:, :] target_points, double[:, :] image_points, double[:, :] target_to_camera, double[:, :] camera_to_image, double xi) noexcept nogil:
	"""Internal : Project 3D points in the target frame to image pixel coordinates, like the camera would. See `target_to_image`"""
	cdef Py_ssize_t i, npoints = target_points.shape[1]
	cdef double target_x, target_y, target_z
	cdef double camera_x, camera_y, camera_z, camera_norm
	cdef double sphere_x, sphere_y, sphere_biased_z
	for i in range(npoints):
		target_x = target_points[0, i]
		target_y = target_points[1, i]
		target_z = target_points[2, i]

		# Matrix product, target_to_camera @ (target_x, target_y, target_z) -> camera frame
		camera_x = target_x*target_to_camera[0, 0] + target_y*target_to_camera[0, 1] + target_z*target_to_camera[0, 2] + target_to_camera[0, 3]
		camera_y = target_x*target_to_camera[1, 0] + target_y*target_to_camera[1, 1] + target_z*target_to_camera[1, 2] + target_to_camera[1, 3]
		camera_z = target_x*target_to_camera[2, 0] + target_y*target_to_camera[2, 1] + target_z*target_to_camera[2, 2] + target_to_camera[2, 3]
		camera_norm = 1 / sqrt(camera_x*camera_x + camera_y*camera_y + camera_z*camera_z)

		# Project on the unit sphere
		sphere_x = camera_x * camera_norm
		sphere_y = camera_y * camera_norm
		sphere_biased_z = 1 / (camera_z * camera_norm + xi)

		# Project on the sensor plane
		projected_x = sphere_x * sphere_biased_z
		projected_y = sphere_y * sphere_biased_z

		# Convert to pixel coordinates (matrix product, camera_to_image @ (projected_x, projected_y))
		image_points[0, i] = projected_x*camera_to_image[0, 0] + projected_y*camera_to_image[0, 1] + camera_to_image[0, 2]
		image_points[1, i] = projected_x*camera_to_image[1, 0] + projected_y*camera_to_image[1, 1] + camera_to_image[1, 2]

def target_to_image(double[:, :] target_points, double[:, :] target_to_camera, double[:, :] camera_to_image, double xi):
	"""Projects 3D points from the target frame to image pixel coordinates, like the camera would
	   - target_points    : ndarray[3, N] : Points in the target frame
	   - target_to_camera : ndarray[4, 4] : Transform matrix from the target frame to the camera frame
	   - camera_to_image  : ndarray[3, 3] : Projection matrix (K) to pixel coordinates
	   - xi               : float         : Mei’s model center shifting parameter (ξ in his paper)
	<---------------------- ndarray[2, N] : Projected points as image pixel coordinates. Those coordinates may not be integers nor within the actual image
	"""
	image_points = np.empty((2, target_points.shape[1]))
	_target_to_image(target_points, image_points, target_to_camera, camera_to_image, xi)
	return image_points