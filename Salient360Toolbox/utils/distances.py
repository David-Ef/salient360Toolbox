#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
from .misc import assertC

import numba

def vec_magnitude(vec):
	""" Magnitude (norm, length) of a vector
	"""
	if len(vec.shape)>1:
		return np.linalg.norm(vec, axis=1)

	return np.linalg.norm(vec)

@numba.njit
def dist_orthodromic(pos1, pos2):
	"""Distance between two positions on a sphere
	Normal and Haversine implementations.
	`pos1` and `pos2` are sets of positions on sphere (lat/long)
	"""
	# # Orthodromic
	# return np.arccos(
	# 		np.sin(pos1[:, 1]) * np.sin(pos2[:, 1]) +
	# 		np.cos(pos1[:, 1]) * np.cos(pos2[:, 1]) +
	# 		np.cos(pos1[:, 0] - pos2[:, 0])
	# 	)
	# Orthodromic - Haversine
	dlon = pos2[:, 0] - pos1[:, 0]
	dlat = pos2[:, 1] - pos1[:, 1]
	a = np.sin(dlat/2)**2 + np.cos(pos1[:, 1]) * np.cos(pos2[:, 1]) * np.sin(dlon/2)**2
	return 2 * np.arcsin(np.sqrt(a))

def dist_angle_arrays_unsigned(vecs1, vecs2):
	# Unsigned distance computed pair-wise between two arrays of unit vectors
	dot = np.einsum("ji,ji->j", vecs1, vecs2)

	return np.arccos(dot)

def dist_angle_topoint_unsigned(vecs1, vec2):
	# Unsigned distance computed between an arrays of unit vectors and a unit vector
	vec2 = np.array(vec2)

	if len(vec2.shape) != 2:
		vec2 = vec2[None, :]

	return dist_angle_arrays_unsigned(vecs1, vec2)

def dist_angle_vectors_unsigned(vec1, vec2):
	# Signed distance computed between two unit vectors

	if len(vec1.shape) != 2:
		vec1 = vec1[None, :]

	if len(vec2.shape) != 2:
		vec2 = vec2[None, :]

	return dist_angle_arrays_unsigned(vec1, vec2)[0]

def mean_dist_angle_topoint(vecs1, vec2):
	"""Mean distance to a centroid
	vecs1: array of unit vectors
	vec2: centroid (unit vector)
	"""
	return dist_angle_topoint_unsigned(vecs1, vec2).mean()

def dist_angle_arrays_signed(vecs1, vecs2):
	# Signed distance computed pair-wise between two arrays of unit vectors

	assertC(vecs1.shape[1] == 2, "[utils.distances.dist_angle_vectors_signed] Takes as input 2-dimensional data (e.g., equirectangular or mercactor projections).")

	cross = np.cross(vecs1, vecs2)
	dot = np.einsum("ji,ji->j", vecs1, vecs2)

	return -np.arctan2( cross, dot)

def dist_angle_topoint_signed(vecs1, vec2):
	# Signed distance computed between an arrays of unit vectors and a unit vector
	vec2 = np.array(vec2)

	if len(vec2.shape) != 2:
		vec2 = vec2[None, :]

	return dist_angle_arrays_signed(vecs1, vec2)

def dist_angle_vectors_signed(vec1, vec2):
	# Signed distance computed between two unit vectors

	if len(vec1.shape) != 2:
		vec1 = vec1[None, :]

	if len(vec2.shape) != 2:
		vec2 = vec2[None, :]

	return dist_angle_arrays_signed(vec1, vec2)[0]

@numba.njit
def dist_starttime(t1, t2):
	"""Difference between fixation starting timestamps fitted with exponential function, reaching asymptote at approx. 75ms of difference
	"""
	return 1 - np.exp(-.05 * np.abs(t1-t2))
