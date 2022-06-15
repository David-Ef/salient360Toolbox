#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: Scanpath maps/videos comparison tools and example as main
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
import scipy as sp
import numba

@numba.njit
def dist_starttime(t1, t2):
	"""Difference between fixation starting timestamp fitted with exponential
	"""
	return 1- np.exp(-.15 * np.abs(t1-t2))

@numba.njit
def angleDiffMetric(s1, s2):
	return np.abs(s1 - s2)

@numba.njit
def sumMetric(s1, s2):
	return np.linalg.norm(s1)*np.linalg.norm(s2)

# @numba.jit
def computeWeightMatrix(VEC1: np.ndarray, VEC2: np.ndarray, weight: np.ndarray):
	"""Return weight matrix for use in MultiMatch alignment process.
	"""
	if weight is None: weight = np.array([1, 1, 1, 1, 1])

	Vals = np.zeros( (VEC1.shape[0], VEC2.shape[0], 5))

	# Values that cannot be computed are preset to NaNs
	Vals[0, 0, 2:5] = np.nan
	Vals[-1, -1, 4] = np.nan

	# #####################################
	# Position
	Fposition = np.einsum("ji,ki->jk", VEC1[:, :3], VEC2[:, :3])
	# 	No need to divide by prod of vector lengths: these are unit vectors
	Fposition = np.arccos(Fposition)

	# Duration
	FDur = sp.spatial.distance.cdist(VEC1[:, 3][:, None], VEC2[:, 3][:, None], metric=angleDiffMetric)

	# Length
	Slength = sp.spatial.distance.cdist(VEC1[1:, 6][:, None], VEC2[1:, 6][:, None], metric=angleDiffMetric)

	# Shape
	Sshape = np.einsum("ji,ki->jk", VEC1[1:, 8:10], VEC2[1:, 8:10]) /\
			sp.spatial.distance.cdist(VEC1[1:, 8:10], VEC2[1:, 8:10], metric=sumMetric)
	Sshape = np.arccos(Sshape)

	# Direction
	Sdir = sp.spatial.distance.cdist(VEC1[1:-1, 7][:, None], VEC2[1:-1, 7][:, None], metric=angleDiffMetric)

	Vals[:,:,0] = Fposition / np.pi
	Vals[:,:,1] = FDur / (np.max(np.append(VEC1[:,3], VEC2[:,3])) + np.finfo(float).eps)
	Vals[1:,1:,2] = Slength / np.pi
	Vals[1:,1:,3] = Sshape / np.pi
	Vals[1:-1,1:-1,4] = Sdir / np.pi

	# Do not add to the weight sum, the values of metrics that are all NaNs
	#	Values that cannot be computed should not lower the result
	nanmask = np.isnan(Vals)
	nanmask = ~np.all(np.all(nanmask, axis=0), axis=0)
	WMat = np.nansum(Vals * weight[None, None, :], axis=2) / (weight*nanmask).sum()

	return WMat, Vals

def simplify(VEC):
	pass
