#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np

import numba

@numba.njit
def UnitVector2Equirect(arr):
	"""Convert from 3D unit vectors to equirectangular projection (long/lat in radians)
	"""	
	out = np.empty( (arr.shape[0], 2) )

	# longitude
	out[:, 0] = np.arctan2(arr[:, 0], arr[:, 1])
	# latitude
	out[:, 1] = np.arcsin(arr[:, 2])

	return out

@numba.njit
def Equirect2UnitVector(arr):
	"""Convert from equirectangular projection (long/lat in radians) to 3D unit vectors
	"""	
	out = np.empty( (arr.shape[0], 3) )

	# X
	out[:, 0] = np.sin(arr[:,1]) * np.cos(arr[:,0])
	# Y
	out[:, 1] = np.sin(arr[:,1]) * np.sin(arr[:,0])
	# Z
	out[:, 2] = np.cos(arr[:,1])

	return out

@numba.njit
def Equirect2Mercator(arr):
	"""Convert from equirectangular projection to Mercator projection
	"""
	out = np.empty( (arr.shape[0], 2) )

	# Shift origin
	out = arr - np.array([0, np.pi/2])
	# longitude does not change
	# latitude
	out[:, 1] = np.log(np.tan(np.pi/4 + (out[:, 1])/2))

	return out

def UnitVector2EquirectArrCheck(arr):
	"""Convert from 3D unit vectors to equirectangular projection (long/lat in radians)
	"""	
	if len(arr.shape) == 1: arr = arr[None, :]
	return UnitVector2Equirect(arr)

def Equirect2UnitVectorArrCheck(arr):
	"""Convert from equirectangular projection (long/lat in radians) to 3D unit vectors
	"""	
	if len(arr.shape) == 1: arr = arr[None, :]
	return Equirect2UnitVector(arr)

def Equirect2MercatorArrCheck(arr):
	"""Convert from equirectangular projection to Mercator projection
	"""
	if len(arr.shape) == 1: arr = arr[None, :]
	return Equirect2Mercator(arr)
