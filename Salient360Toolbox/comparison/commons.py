#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: functions shared by dynamic and static stimuli comparison methods
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np

# Saliency comparison

def quasiUniformSphereSampling(N):
	"""Equirectangular weighting by quasi-uniform sampling of a sphere.
	Used in Rai, Y., Gutiérrez, J., & Le Callet, P. (2017, June). A dataset of head and eye movements for 360 degree images. In Proceedings of the 8th ACM on Multimedia Systems Conference (pp. 205-210). ACM.
	"""
	gr = (1 + np.sqrt(5))/2
	ga = 2 * np.pi * (1 - 1/gr)

	ix = iy = np.arange(N)

	lat = np.arccos(1 - 2*ix/(N-1))
	lon = iy * ga
	lon %= 2*np.pi

	return np.concatenate([lat[:, None], lon[:, None]], axis=1)

def getStartPositions(fixationList):
	"""Return positions of first fixation in list of scanpaths.
	Get starting indices of individual fixation sequences.
	"""
	return np.where(fixationList[:, 0] == 0)[0]
