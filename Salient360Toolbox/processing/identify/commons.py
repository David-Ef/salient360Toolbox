#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np

from ...utils.misc import *
from ...utils.distances import *

def fix_sacc(seq, resample):
	"""Remove short saccades samples and link saccades separated by short fixations
	"""
	# max gap between two saccades (ms.)
	maxDel = 50 / resample
	minDel = 12 / resample

	i = 0
	sacc = False
	stS = None
	# pass: remove short saccade samples
	while i < seq.shape[0]-1:
		if seq[i] == 1:
			if not sacc:
				sacc = True
				stS = i

			if seq[i+1] == 0 and ((i+1)-stS) < minDel:
				seq[stS:i+1] = 0
				sacc = False
		else:
			sacc = False

		i+=1

	i = 0
	sacc = False
	stS = None
	# pass: fill gap between close saccades
	while i < seq.shape[0]-1:
		if seq[i] == 1:
			sacc = True
			stS = i
		elif sacc:
			if seq[i+1] == 1:
				seq[stS:i+1] = 1
				sacc = False

			elif ((i+1)-stS) > maxDel:
				sacc = False
		i+=1

def fix_fix(seq, resample):
	"""Remove short fixations
	"""
	minDur = 80 / resample

	i = 0
	fix = False
	stS = None
	while i < seq.shape[0]-1:
		if seq[i] == 0:
			if not fix:
				fix = True
				stS = i

			if seq[i+1] != 0 and ((i+1)-stS) < minDur:
				seq[stS:i+1] = seq[i+1]
		else:
			fix = False

		i+=1

def fix_gen(label_list):
	# Removes unique True and False values surrounded by their complement (would usually disappear when the signal is smoothed)

	for i in range(1, label_list.shape[0]-1):
		lc = label_list[i]
		lm = label_list[i-1]
		lp = label_list[i+1]

		if lc != lm and lc != lp:
			label_list[i] = lm

	for i in [0, label_list.shape[0]-1]:
		lc = label_list[i]

		if i > 0:
			lm = label_list[i-1]
		else: lm = not lc

		if i < (label_list.shape[0]-1):
			lp = label_list[i+1]
		else: lp = not lc

		if lc != lm and lc != lp:
			label_list[i] = lm

# Common functions
def getVelocity(gp, return_keep=False, outlierSigma=5):
	"""Input data: gaze data as unit vector followed by a timestamp
	0: X
	1: Y
	2: Z
	3: timestamp
	"""

	diffT = gp[1:, 3] - gp[:-1, 3]

	distance = dist_angle_arrays_unsigned(gp[1:, :3], gp[:-1, :3])

	velocity = distance/diffT
	velocity = np.append(velocity, [velocity[0]])

	if return_keep:
		# Remove samples farther than X std from the mean
		keep = np.abs( (velocity-np.nanmean(velocity))/np.nanstd(velocity) ) < outlierSigma
		printNeutral("Removed {} samples more than {} sigmas away from the mean".format((keep==0).sum(), outlierSigma),
			bold=False, verbose=2)

		keep &= np.logical_not(np.isnan(velocity) | np.isinf(velocity))

		return keep, velocity
	else:
		return None, velocity
