#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: Numpy implementations of Saliency maps/videos comparison tools (by Chencan Qian, Sep 2014 [*repo]) and example as main - https://github.com/herrlich10/saliency
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# Note:
#	Numpy metrics are ported from Matlab implementation provided by http://saliency.mit.edu/
#	Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.
#	Python/numpy implementation: Chencan Qian, Sep 2014 [*repo]
#	Python/pyorch implementation: Erwan David, 2018
#	[*repo] https://github.com/herrlich10/saliency
#	Refer to the MIT saliency Benchmark (website)[http://saliency.mit.edu/results_cat2000.html] for information about saliency measures
# ---------------------------------

import numpy as np
from numpy import random
from skimage.transform import resize
from statsmodels.stats.weightstats import DescrStatsW

import numba

EPSILON = np.finfo('float').eps

class SineWeightMap():
	"""Sine weighting map
	An equirectangular projection displays distortions as a function of the sin or cos of the latitude (sin/cos according to Y map limits).
	"""
	def __init__(self, height = None):
		if height is None:
			self.height = None
			self.WMap = None
		else:
			self.createWMap(height)

	def __call__(self, height):
		if height != self.height:
			self.createWMap(height)
		return self.WMap

	def createWMap(self, height):
		self.height = height
		self.WMap = np.sin(np.linspace(0, np.pi, self.height))
		self.WMap = np.repeat(self.WMap[:, None], height*2, axis=1)
		# self.WMap[:] = 1 # Uniform weighting

WeightMap = SineWeightMap()

# @numba.jit
def normalize(x, method='standard', axis=None):
	"""Normalize data
	`standard`: i.e. z-score. Substract mean and divide by standard deviation

	`range`: normalize data to new bounds [0, 1]

	`sum`: normalize so that the sum of all element in tensor sum up to 1.
	"""
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res

@numba.jit(forceobj=True)
def KLD(p, q):
	"""Weighted Kullback-Leibler Divergence

	Moharana, R., & Kayal, S. (2017). On weighted Kullback-Leibler divergence for doubly truncated random variables. RevStat.
	"""

	wmap = WeightMap(p.shape[0])

	p *= wmap
	q *= wmap

	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

@numba.jit(parallel=True)
def AUC_Judd(saliency_map, fixation_map, jitter=False):
	"""AUC_Judd
	"""
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		# printWarning('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Jitter the saliency map slightly to disrupt ties of the same saliency value
	if jitter:
		saliency_map += random.rand(*saliency_map.shape) * 1e-7
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# Calculate AUC
	thresholds = sorted(S_fix, reverse=True)
	tp = np.zeros(len(thresholds)+2)
	fp = np.zeros(len(thresholds)+2)
	tp[0] = 0; tp[-1] = 1
	fp[0] = 0; fp[-1] = 1
	for k in numba.prange(len(thresholds)):
		thresh = thresholds[k]
		above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
		tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
		fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
	return np.trapz(tp, fp) # y, x

@numba.jit(parallel=True)
def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	"""AUC_Borji
	"""
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		# print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the saliency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in numba.prange(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k in numba.prange(len(thresholds)):
			thresh = thresholds[k]
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits

@numba.jit
def NSS(saliency_map, fixation_map):
	"""Normalized Scanpath Saliency
	"""
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])

@numba.jit(forceobj=True)
def CC(saliency_map1, saliency_map2):
	"""Weighted Cross-Correlation (Pearson's linear coefficient)
	from statsmodels.stats.weightstats import DescrStatsW (method "corrcoef").
	Set "weights" with ones for unweighted variant.
	"""
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')

	wmap = WeightMap(map1.shape[0])
	# Compute correlation coefficient
	return DescrStatsW(np.concatenate([map1.ravel()[:, None], map2.ravel()[:, None]], axis=1), weights=wmap.ravel()).corrcoef[0, 1]

@numba.jit(forceobj=True)
def SIM(saliency_map1, saliency_map2):
	"""SIMilarity measure (aka histogram intersection)
	"""
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default

	wmap = WeightMap(map1.shape[0])
	map1 *= wmap
	map2 *= wmap

	# Normalize the two maps to have values between [0,1] and sum up to 1
	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute histogram intersection
	intersection = np.minimum(map1, map2)
	return np.sum(intersection)

@numba.jit
def InfoGain(saliency_map, fixation_map, baseline_map):
	"""InfoGain
	ref: Kümmerer, M., Wallis, T. S., & Bethge, M. (2015). Information-theoretic model comparison unifies saliency metrics. Proceedings of the National Academy of Sciences, 112(52), 16054-16059.
	repo matlab code: github.com/cvzoya/saliency/blob/master/code_forMetrics/InfoGain.m
	"""
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	b_map = np.array(baseline_map, copy=False) # get baseline

	# Resize if necessary
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	if b_map.shape != f_map.shape:
		b_map = resize(b_map, f_map.shape)

	# No need to use "normalize(..., method='range')" here, right ?
	# Normalize to sum to 1 (PDF)
	s_map = normalize(s_map, method='sum')
	b_map = normalize(b_map, method='sum')

	# Vectorize maps
	s_map = s_map.ravel()
	f_map = f_map.ravel()
	b_map = b_map.ravel()

	return np.mean( np.log2(EPSILON + s_map[f_map]) -\
				    np.log2(EPSILON + b_map[f_map]) )

# Name: func, compute AB & BA?, second map should be saliency or fixation?
metrics = {}
metrics["AUC_Judd"] = [AUC_Judd, False, 'fix'] # Binary fixation map
metrics["AUC_Borji"] = [AUC_Borji, False, 'fix'] # Binary fixation map
metrics["NSS"] = [NSS, False, 'fix'] # Binary fixation map
metrics["CC"] = [CC, False, 'sal'] # Saliency map
metrics["SIM"] = [SIM, False, 'sal'] # Saliency map
metrics["KLD"] = [KLD, True, 'sal'] # Saliency map
metrics["InfoGain"] = [InfoGain, False, 'fix'] # Saliency map & Binary fixation map

# Possible float precision of bin files
dtypes = {16: np.float16,
		  32: np.float32,
		  64: np.float64}
