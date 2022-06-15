#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: saliency comparison functions
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import re, os
from .saliencyMetrics import *

get_binsalmap_infoRE = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")
def get_binsalmap_info(filename):
	"""Return binary saliency maps metadata: stimuli name, frame width and height and number of frames
	"""
	name, width, height, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
	width, height, dtype = int(width), int(height), int(dtype)
	return name, width, height

def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None, basemap=None):
	"""Compute and return similarity values specified in metrics `keys_order`.
	"""
	results = {}

	for metric in metrics.keys():

		func = metrics[metric][0]
		sim = metrics[metric][1]
		compType = metrics[metric][2]

		if not sim:
			if compType == "fix" and "NoneType" not in [type(fixmap1), type(fixmap2)]:
				m = (func(salmap1, fixmap2)
				   + func(salmap2, fixmap1))/2
			else:
				m = (func(salmap1, salmap2)
				   + func(salmap2, salmap1))/2
		else:
			if func == "InfoGain" and basemap is not None:
				m = func(salmap2, fixmap1, basemap)
			elif compType == "fix" and fixmap2 is not None:
				m = func(salmap1, fixmap2)
			else:
				m = func(salmap1, salmap2)

		results[metric] = m

	return results

def compareSaliency(*args, **kargs):
	return getSimVal(*args, **kargs)
