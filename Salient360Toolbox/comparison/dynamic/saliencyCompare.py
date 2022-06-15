#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: saliency comparison functions
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import re
from ..saliencyMetrics import *
from ...utils import misc

get_binsalmap_infoRE = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
def get_binsalmap_info(filename):
	"""Return binary saliency maps metadata: stimuli name, frame width and height and number of frames
	"""
	name, width, height, Nframes, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
	width, height, Nframes, dtype = int(width), int(height), int(Nframes), int(dtype)
	return name, width, height, Nframes, dtype

def getFramePoolingIdx(tempWindowSize, FrameCount):
	"""Return start and end frames of frame pools
	Comparisons are made on the basis of groups of framed pooled according to a window size (in msec.).
	"""
	tempWindowSize = int(np.round(FrameCount / 20 * (tempWindowSize/1000)))

	framePooling = np.arange(0, FrameCount+1, tempWindowSize)
	framePooling = np.concatenate([ framePooling[:-1, None],
									framePooling[ 1:, None]-1], axis=1).astype(int)
	return framePooling

def getPooledFramesSM(file, range_, shape, dtype=32):
	"""Given a frame pool range, return the normalized sum of all saliency (map) frames within the pool.
	"""
	iStart, iEnd = range_
	height, width = shape
	N = iEnd-iStart + 1

	if type(file) == str:
		file = open(file, "rb")

	file.seek(width*height * iStart * (dtype//8))

	data = np.fromfile(file, count=N*height*width, dtype=dtypes[dtype])
	data = data.reshape([N, height, width])

	# Collapse on the Frame axis
	salmap = data.sum(axis=0)
	sum_ = salmap.sum()

	if sum_ == 0:
		# Empty saliency map
		return salmap
	else:
		# Return saliency maps normalized
		return salmap / salmap.sum()

def getPooledFramesFM(fixations, range_, shape):
	"""Given a frame pool range, return the sum of all fixation (map) frames within the pool.
	"""
	iStart, iEnd = range_

	fixationmap = np.zeros(shape, dtype=int)
	for iFrame in range(iStart, iEnd+1):
		FIX = np.where(
				np.logical_and( fixations[:, 2] <= iFrame,
								fixations[:, 3] >= iFrame ) )[0]
		for iFix in FIX:
			fixationmap[ int(fixations[iFix, 1]), int(fixations[iFix, 0]) ] += 1

	return fixationmap

def getWindValue(salmap1_p, salmap2_p,
				 FL1, FL2,
				 DIM, fPool,
				 i_window):
	misc.printWarning("{}/{} - {:.2f}%".format(i_window+1, fPool.shape[0], (i_window+1)/fPool.shape[0]*100), end="", header="", clear=True, verbose=0)

	# Retrieve saliency map for frame range
	salmap1_win = getPooledFramesSM(salmap1_p, fPool[i_window, :], DIM)
	salmap2_win = getPooledFramesSM(salmap2_p, fPool[i_window, :], DIM)

	# Retrieve fixations map for frame range
	fixmap1_win = getPooledFramesFM(FL1, fPool[i_window, :], DIM)
	fixmap2_win = getPooledFramesFM(FL2, fPool[i_window, :], DIM)

	if salmap1_win.sum() < 1e-12 or\
	   salmap2_win.sum() < 1e-12 or\
	   fixmap1_win.sum() < 1e-12 or\
	   fixmap2_win.sum() < 1e-12:
		return {key: np.nan for key in metrics.keys()}

	# Normalization is left to the metric functions

	from ..saliencyCompare import getSimVal
	values = getSimVal(salmap1_win, salmap2_win, fixmap1_win, fixmap2_win)

	return values
