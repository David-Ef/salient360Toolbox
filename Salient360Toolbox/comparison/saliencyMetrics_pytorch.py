#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: pytorch impementation of saliency map comparison metrics. Implements weighting variants for comparisons of equirectangular saliency maps/videos.
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import torch
import numpy as np

GPU = torch.cuda.is_available()
if GPU:
	dtype = torch.cuda.FloatTensor
	cltype = torch.cuda.LongTensor
else:
	dtype = torch.FloatTensor
	cltype = torch.LongTensor

EPSILON = torch.tensor(np.finfo('float').eps).type(dtype)
nan = torch.tensor(np.nan).type(dtype)

def readBinaryMap(dataptr):
	"""Load binary map, return saliency map as numpy 2D array
	"""
	return np.fromfile(dataptr, count=1024*2048, dtype=np.float32).reshape([1024, 2048])

def readFixationMap(path):
	"""Load fixation list and return fixation map (numpy 2D array)
	"""
	fixations = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,2, 5,6))
	fixations = fixations * [2048, 1024, 1,1] - [1,1, 0,0]

	sf_map = np.zeros([1024, 2048], dtype=int)
	for sf in fixations: sf_map[int(sf[1]), int(sf[0])] += 1

	return sf_map

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
WeightMap = SineWeightMap()
wmap = torch.tensor(WeightMap(1024)).type(dtype)

def dt_N(x):
	"""Return number of elements in tensor
	"""
	return torch.tensor(x.size()).prod()

def dt_mean(x):
	"""Return tensor mean
	"""
	return x.sum()/dt_N(x).type(dtype)

def dt_flatten(x):
	"""Return flattened tensor
	"""
	return x.view(-1, 1)

def normalize(x, method='standard', axis=None):
	"""Normalize data
	
	`standard`: i.e. z-score. Substract mean and divide by standard deviation

	`range`: normalize data to new bounds [0, 1]

	`sum`: normalize so that the sum of all element in tensor sum up to 1.
	"""
	if method == 'standard':
		res = (x - dt_mean(x)) / x.std()
	elif method == 'range':
		res = (x - x.min()) / (x.max() - x.min())
	elif method == 'sum':
		res = x / x.sum()
	else:
		raise ValueError('method not in {"standard", "range", "sum"}')
	return res

def KLD(saliency_map1, saliency_map2):
	"""Weighted Kullback-Leibler Divergence

	Moharana, R., & Kayal, S. (2017). On weighted Kullback-Leibler divergence for doubly truncated random variables. RevStat.
	"""
	saliency_map1[saliency_map1<0] = EPSILON
	saliency_map2[saliency_map2<0] = EPSILON

	saliency_map1 = normalize(saliency_map1 * wmap, method='sum')
	saliency_map2 = normalize(saliency_map2 * wmap, method='sum')

	mask = (saliency_map1 > EPSILON) | (saliency_map2 > EPSILON)
	return (saliency_map1[mask] *\
				torch.log( (saliency_map1[mask]+EPSILON) / (saliency_map2[mask]+EPSILON) )
			).sum()

def NSS(saliency_map, fixation_map):
	"""Normalized Scanpath Saliency
	"""
	f_map = fixation_map > 0.5
	s_map = normalize(saliency_map, method='standard')

	return dt_mean(s_map[f_map])

def CC(saliency_map1, saliency_map2):
	"""Weighted Cross-Correlation (Pearson's linear coefficient)
	Adapted from statsmodels.stats.weightstats import DescrStatsW (method "corrcoef").
	Set "weights" to ones for unweighted variant.
	"""
	map1 = normalize(saliency_map1, method='standard')
	map2 = normalize(saliency_map2, method='standard')

	data = torch.cat([dt_flatten(map1), dt_flatten(map2)], 1)

	weights = dt_flatten(wmap)
	ddof = 2

	sum_ = torch.mm( data.transpose(0,1), weights )
	sum_weight = weights.sum()

	mean = sum_ / sum_weight
	demeaned = data - mean.view(1,-1)

	sumsquare = torch.mm( torch.pow(demeaned, 2).transpose(0,1), weights )

	var = sumsquare / (sum_weight - ddof)
	std = torch.sqrt(var)

	cov = torch.mm( (weights * demeaned).transpose(0,1), demeaned )
	cov /= sum_weight - ddof

	corrcoef = cov / std.view(-1) / std.view(-1, 1)

	return corrcoef[0, 1]

def SIM(saliency_map1, saliency_map2):
	"""Weighted SIMilarity measure (aka histogram intersection)
	"""
	map1 = dt_flatten(saliency_map1 * wmap)
	map2 = dt_flatten(saliency_map2 * wmap)

	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')

	return torch.min(map1, map2).sum()

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, ):
	"""AUC_Borji.
	Not implemented.
	"""
	return NotImplemented

def AUC_Judd(saliency_map, fixation_map, jitter=False):
	"""AUC_Judd.
	Not implemented.
	"""
	return NotImplemented

def InfoGain(saliency_map, fixation_map, baseline_map):
	"""InfoGain.
	Not implemented.
	"""
	return NotImplemented

# Name: func, compute AB & BA?, second map should be saliency or fixation?
metrics = {}
metrics["AUC_Judd"] = [AUC_Judd, False, 'fix'] # Binary fixation map
metrics["AUC_Borji"] = [AUC_Borji, False, 'fix'] # Binary fixation map
metrics["NSS"] = [NSS, False, 'fix'] # Binary fixation map
metrics["CC"] = [CC, False, 'sal'] # Saliency map
metrics["SIM"] = [SIM, False, 'sal'] # Saliency map
metrics["KLD"] = [KLD, True, 'sal'] # Saliency map
metrics["InfoGain"] = [InfoGain, False, 'fix'] # Saliency map & Binary fixation map
