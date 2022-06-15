#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: Scanpath maps/videos comparison tools and example as main
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np

def compareScanpath(scanpath1, scanpath2, frame_window_size=20, weight=None):
	if weight is None:
		weight = [1., 1., 1., 1., 1.]
	weight = np.array(weight)

	minFrame = int(np.min([scanpath1[0, 10], scanpath2[0, 10]]))
	maxFrame = int(np.min([scanpath1[-1, 11], scanpath2[-1, 11]]))

	score = 0
	scores = np.zeros_like(weight, dtype=float)

	n_compared = 0
	for iFrame in range(minFrame, maxFrame, frame_window_size):
		lowerFrame = iFrame
		highFrame = iFrame + frame_window_size

		# Get index of points that happened during this frame
		scanpath1_frame = np.where(np.logical_and(
					scanpath1[:, 10] >= lowerFrame,
					scanpath1[:, 11] <= highFrame))[0]
		scanpath2_frame = np.where(np.logical_and(
					scanpath2[:, 10] >= lowerFrame,
					scanpath2[:, 11] <= highFrame))[0]

		# We need at least one pair of elements to compare
		if len(scanpath1_frame) == 0 or len(scanpath2_frame) == 0:
			continue

		# Compute similarity metrics
		from ..scanpathCompare import compareScanpath
		score_, scores_ = compareScanpath(scanpath1[scanpath1_frame, :], scanpath2[scanpath2_frame, :],
			weight=weight)

		# Add to weighted average score
		score += score_
		# Add values to score array
		scores += scores_

		n_compared += 1

	return score/n_compared, scores/n_compared
