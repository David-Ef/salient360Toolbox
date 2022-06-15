#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

from glob import glob
import numpy as np
import os, builtins

# First call to some of the functions below will take some time if numba is available
#	On my computer the speed up is 1.43, but the first iteration takes 100 times longer
# 	The speed up gained from numba will be negative with small sample sizes
#		due to the time it takes to compile functions, disable it this way:
os.environ["NUMBA_DISABLE_JIT"] = "1" # Set to 0 to enable numba
# Only print most important messages
builtins.verbose = 0

try:
	import Salient360Toolbox
except:
	import sys
	sys.path.append("../")

from Salient360Toolbox.utils import misc

PATH_DATA = "../data/raw_gaze/*Static"
PATH_OUT = "./out_static/"
PATH_STIM = "../data/stimuli/1_PortoRiverside.png"
os.makedirs(PATH_OUT, exist_ok=True)

# Tracking can be HE (Head+Eye) or H (Head alone)
tracking = "HE"
# Targeted eye
eye = "R"
# Resampling rate
resample = 90 # Resample to original head movement sampling rate
# Filter settings
filterSettings = {"name": "savgol", "params": {"win": 9, "poly": 2}}
# Gaze parsing settings
parsingSettings = {"name": "I-VT", "params": {"threshold": 120}}
# Dimensions of all output images (Height, Width)
dim = [500, 1000]

sal_maps = []
fix_lists = []
files = glob("{}*.csv".format(PATH_DATA))

# Generate

from Salient360Toolbox import helper
from Salient360Toolbox.generation import saliency as sal_generate
from Salient360Toolbox.generation import scanpath as scanp_generate

for ipath, path in enumerate(files):

	print("\n", path)

	savename = misc.getFileName(path)

	# Print in and out path if there is enough space in the terminal
	inpath = path; outpath = PATH_OUT+savename

	# Load and process raw data
	gaze_data, fix_list = helper.loadRawData(path,
		# If gaze tracking, which eye to extract
		eye=eye,
		# Gaze or Head tracking
		tracking=tracking,
		# Resampling at a different sample rate?
		resample=resample,
		# Filtering algo and parameters if any is selected
		filter=filterSettings,
		# Fixation identifier algo and its parameters
		parser=parsingSettings)

	# Generate saliency map from loaded data
	sal_map = helper.getSaliencyMap(fix_list[:, [2,3,4, 0,1]], dim,
			# Name of binary saliency file created for caching purposes
			name=savename,
			# If a bin file exists at this location we load the saliency data from it, unless force_generate is True. Saliency will be saved if caching is True
			path_save=PATH_OUT,
			# Sigma of the 2D Gaussian drawn at the location of fixations
			gauss_sigma=2,
			# Asks to return saliency data rather than a path to a saliency data file if it exists
			force_return_data=True,
			# Generate data instead of reading from pre-existing file
			force_generate=False,
			# Will save saliency to bin file to fast load at a later time
			caching=True)

	fix_lists.append(fix_list)
	sal_maps.append(sal_map)

	sal_image = sal_generate.toImage(sal_map, cmap="coolwarm")[:,:,::-1]
	
	# Save saliency map as an image
	misc.printNorm("Saliency map to image file.", verbose=0)
	sal_generate.saveImage(sal_image, outpath+"_salmap")
	
	# Save saliency map as an image blended with a stimulus background
	misc.printNorm("Saliency map to blended image file.", verbose=0)
	sal_generate.saveImage(sal_map, outpath+"_bsalmap", blend=PATH_STIM)

	# Save saliency maps to a binary file (read with .utils.readOutFile.readBinarySaliencyMap)
	misc.printNorm("Saliency map to binary file.", verbose=0)
	sal_generate.saveBin(sal_map, outpath)

	# Save saliency maps to a compressed binary file 
	misc.printNorm("Saliency map to compressed binary file.", verbose=0)
	sal_generate.saveBinCompressed(sal_map, outpath)

	# Save a scanpath as a series of fixation points, temporally ordered and labelled
	misc.printNorm("Scanpath data to image.", verbose=0)
	scanp_generate.toImage(fix_list[:, :2], dim, outpath+"_scanpath")

	# Save a scanpath blended with a stimulus background
	misc.printNorm("Scanpath data to blended image.", verbose=0)
	scanp_generate.toImage(fix_list[:, :2], dim, outpath+"_bscanpath", blend=PATH_STIM)

	# Save scanpath data (fixation and saccade features) to file
	misc.printNorm("Scanpath data to csv text file.", verbose=0)
	scanp_generate.toFile(fix_list, outpath+"_fixation.csv",
		# Save all columns (features)
		saveArr=np.arange(fix_list.shape[1]), mode="w")

	# Get a fixation map (2d matrix with number of fixations observed at each pixel location)
	fix_map = helper.getFixationMap(fix_list[:, :2], dim)

	# Save fixation list as a gray scale image
	misc.printNorm("Fixation list to fixation map image", verbose=0)
	fix_map_img = sal_generate.toImage(fix_map, cmap="binary", reverse=True)
	sal_generate.saveImage(fix_map_img, outpath+"_fixmap")

	# Save fixation list as a binary file
	misc.printNorm("Fixation list to fixation map binary", verbose=0)
	sal_generate.saveBin(fix_map, outpath+"_fixmap")

# Compare

WEIGHTS = [1, 0, 1, 1, 1]

from Salient360Toolbox.comparison.saliencyMetrics import metrics
metrics.pop("InfoGain") # This measure is not yet implemented
from Salient360Toolbox.comparison.saliencyCompare import compareSaliency
from Salient360Toolbox.comparison.scanpathCompare import transformScanpath, measureNames, compareScanpath

out_result = open("{0}{1}comparisons.csv".format(PATH_OUT, os.sep),  "a+")

for i1, i2 in [[0, 1], [0, 2], [1, 2]]:

	results = {}

	name1 = misc.getFileName(files[i1])
	name2 = misc.getFileName(files[i2])

	SM1 = sal_maps[i1]
	SM2 = sal_maps[i2]

	FL1 = fix_lists[i1]
	FL2 = fix_lists[i2]

	FM1 = helper.getFixationMap(FL1[:, :2], dim)
	FM2 = helper.getFixationMap(FL2[:, :2], dim)

	misc.printNorm("Computing saliency similarity metrics", verbose=0)
	results = dict(results, **compareSaliency(SM1, SM2, FM1, FM2))

	SP1 = FL1[:, [9, 0,1, 12]] # fix_idx, long, lat (gaze), timestamp
	SP2 = FL2[:, [9, 0,1, 12]]

	SP1[:, 1:3] *= [np.pi*2, np.pi] # Un-normalise
	SP2[:, 1:3] *= [np.pi*2, np.pi]

	SC1 = transformScanpath(SP1, [0, SP1.shape[0]], 0)
	SC2 = transformScanpath(SP2, [0, SP2.shape[0]], 0)

	misc.printNorm("Computing MultiMatch similarity metrics", verbose=0)
	score, scores = compareScanpath(SC1, SC2, weight=WEIGHTS)

	results["multimatch.wavg"] = score
	for i in range(len(scores)):
		results["multimatch."+measureNames[i]] = scores[i]

	for metric, value in results.items():
		out_line = "{},{},{},{:.5e}\n".format( name1, name2,  metric, value)
		out_result.write(out_line)

		misc.printNeutral("{},{},{},{:.3f}".format( name1, name2,  metric, value),
			tab=1, verbose=0)

out_result.close()
