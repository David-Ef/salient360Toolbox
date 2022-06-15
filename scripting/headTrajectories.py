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
#	On my computer the speed up is 1.43, but the first iteration duration can be very long
# 	The speed up gained from numba will be negative with small sample sizes
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
PATH_OUT = "./out_headtraj/"
PATH_STIM = "../data/stimuli/1_PortoRiverside.png"
os.makedirs(PATH_OUT, exist_ok=True)

# Request to process head data
tracking = "H"
dim = [500, 1000]

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
		# Gaze or Head tracking
		tracking=tracking,
		# If gaze tracking, which eye to extract
		eye=tracking,
		# Resampling at a different sample rate?
		resample=None,
		# Filtering algo and parameters if any is selected
		filter=None,
		# Fixation identifier algo and its parameters
		parser=None)

	# Generate saliency map from loaded data
	sal_map = helper.getSaliencyMap(fix_list[:, [2,3,4, 0,1]], dim, savename,
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
