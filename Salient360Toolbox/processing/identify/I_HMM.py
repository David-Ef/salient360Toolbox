#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# -----saccadeVelThreshsaccadeVelThresh----------------------------

import numpy as np
import pomegranate as HMM

from .commons import *

def parse(data,
		  nStates=2,
		  n_jobs=-1,
		  callback=None):
	"""
	Train a HMM on a sequence of gaze features. If data are a velocity signal, the model will ideally separate high (saccades) from low velocities (fixations).
	*data*: data must be of shape [N, M], where N is the number of samples and M the number of features
	*nStates*: number of hidden state in the model
	*saccadeVelThresh*: 
	*n_jobs*: number of cpu cores to use for training HMM (def.: all)

	^*The probability distributions available are listed on the promegranate repo.: github.com/jmschrei/pomegranate/tree/master/pomegranate/distributions

	Returns an integer array containig state indices for the current sequence data.
	"""

	try: nStates = int(nStates)
	except: printWarning("nStates must be castable as int. Can't cast {} to int.".format(type(nStates))); exit()
	assert nStates > 1, "nStates cannot be less than 2. Got {}.".format(nStates)

	assert len(data.shape) in [1, 2], "Parameter \"data\" must be 2D numpy array. Got {}.".format(data.shape)
	if len(data.shape) == 1: data = data[:, None]

	if n_jobs == -1:
		from multiprocessing import cpu_count
		n_jobs = cpu_count()
	elif type(n_jobs) is not int:
		n_jobs = 1

	hmm = HMM.HiddenMarkovModel.from_samples(HMM.NormalDistribution,
		n_components=nStates, X=data,  n_jobs=n_jobs)
	if callback is not None: callback(.8)
	# Get state prediction with Viterbi's algorithm
	_, states = hmm.viterbi(data)
	states = np.array([int(state[0]) for state in states[1:]], dtype=int)
	# Infer state order from emission probabilities mean
	#	Lowest gaussian mean models fixation
	means = np.array([state.distribution.parameters[0] for state in hmm.states if state.name.split("-")[-1] not in ["start", "end"]])

	order = np.argsort(means, axis=0)
	states = (states==order[0]).astype(np.int)

	# fix_sacc(states, 10)
	# fix_fix(states, 10)

	fix_gen(states)

	return states
