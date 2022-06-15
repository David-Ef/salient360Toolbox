#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np
from .commons import *

def custom_distance(sample1, sample2):
	# return if samples are too far away temporally
	if abs(sample2[3] - sample1[3]) > 10:
		return 0
	# Orthodromic distance
	dist_pos = np.abs(dist_angle_vectors_unsigned(sample1[:3], sample2[:3]))

	return dist_pos

def parse(data, eps=.005, minpts=3,
	callback=None, **kwargs):

	from scipy.spatial.distance import pdist, squareform
	from scipy.sparse import csr_matrix
	from sklearn import cluster

	dist_matrix = csr_matrix(squareform(pdist(data, metric=custom_distance)))
	if callback is not None: callback(.5)

	dbscan = cluster.DBSCAN(
		eps=eps,
		min_samples=minpts,
		metric="precomputed",
		n_jobs=-1)
	dbscan.fit(dist_matrix)

	y_pred = dbscan.labels_.astype(int)

	# Separate noise samples (saccades) from groups (fixations)
	typeMat = (y_pred == -1).astype(int)
	if callback is not None: callback(.8)
	
	# fix_sacc(typeMat, 10)
	# fix_fix(typeMat, 10)

	fix_gen(typeMat)
	
	if callback is not None: callback(.99)

	return typeMat
