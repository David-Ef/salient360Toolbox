#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np

from .commons import *

def parse(timestamp, velocity,
	threshold=100, minFixationTime=80,
	callback=None, **kwargs):
	"""
	Parse gaze data on sphere and output a saccade/fixation label list with a velocity-base algorithm.
	Returns an integer array where 0 = fixations, 1 = saccades
	"""

	threshold = np.deg2rad(threshold)/1000 # Eye threshold rad/ms

	# Label as part of fixations samples with velocity below threshold
	#	i.e., Fixation == 1, Saccade == 0
	fixationMarkers = np.array(velocity <= threshold, dtype=np.bool)

	fix_gen(fixationMarkers)

	Nsamples = velocity.shape[0]

	iDisp = 1
	startMarker = 0

	# Remove short fixations
	change = False
	iDisp = 1
	fixStart = np.zeros(1)
	fixEnd = np.zeros(1)
	while iDisp < (Nsamples-1):
		# print("  \rremove sf", iDisp, end="");
		# Point where sacc ends and fix starts
		if not fixationMarkers[iDisp] and fixationMarkers[iDisp+1]:
			# print("  \rremove sf", iDisp, end="");
			startMarker = iDisp
			fixStart[:] = timestamp[iDisp]

			# Loop ahead until we find the start of a new saccade
			while iDisp < (Nsamples-1) and fixationMarkers[iDisp+1]:
				iDisp+=1

			fixEnd[:] = timestamp[iDisp]

			if fixEnd-fixStart < minFixationTime:
				fixationMarkers[startMarker: iDisp+1] = False
				change = True
		else:
			iDisp += 1

		# Reset until no small fixations are found
		if iDisp==(Nsamples-1) and change:
			iDisp = 1
			change = False
	
	return fixationMarkers
