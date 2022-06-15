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

PATH_DATA = "../data/raw_gaze/*Dyn"
PATH_OUT = "./out_dynamic/"
PATH_STIM = "../data/stimuli/1_PortoRiverside.mp4"
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
# Dimensions of all output images
DIM = [500, 1000]
FRAME_COUNT = 0

sal_map_ps = []
fix_lists = []
files = glob("{}*.csv".format(PATH_DATA))

# Generate

from Salient360Toolbox import helper
from Salient360Toolbox.generation import saliency as sal_generate

sal_map = None

for ipath, path in enumerate(files):

	savename = misc.getFileName(path)

	# Print in and out path if there is enough space in the terminal
	inpath = path; outpath = PATH_OUT+savename

	# Load frame data, an array of ints showing the current video frame index for each gaze sample
	frame_data = np.loadtxt(path, delimiter=",", usecols=[0, 2], skiprows=1)
	# Sampling started before the first video frame was displayed
	#	We look for that first frame to determine where to start processing data
	data_range = [np.where(frame_data[:, 1]==1)[0][0], frame_data.shape[0]]
	frame_data = frame_data[slice(*data_range)]
	# Center timestamp and convert to millisecond
	frame_data[:, 0] -= frame_data[0, 0]
	frame_data[:, 0] /= 1e6
	# We extract frame data in order to compute one saliency map per frame
	#	and output a saliency video

	# Load and process raw data
	gaze_data, fix_list, _ = helper.loadRawData(path,
		# Only consider data in this range
		data_range=data_range,
		# Return index of samples that are kept after removing velocity outliers
		#	If True, outlier samples will be removed on the basis of their velocity
		return_keep=True,
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

	# Keep our frame count aligned with processed raw data
	if resample > 0:
		# If interpolated, interpolate the same way
		from scipy import interpolate
		frame_data = interpolate.griddata(frame_data[:, 0], frame_data[:, 1], (gaze_data[:, 9]), method='linear')
	# This will be used to determine what are the starting and ending frames of fixations
	frame_data = frame_data.astype(np.int) # Remove decimal part

	FRAME_COUNT = max(np.max(frame_data), FRAME_COUNT)
	# Update output saliency dimensions with the number of frames
	dim = [FRAME_COUNT]+DIM

	# Keeping, unit vector (x,y,z), long, lat, start/end sample indices
	fix_list = fix_list[:, [2,3,4, 0,1, 10,11]]

	# Generate saliency video from loaded data
	# 	Also pass fixation start/end index of samples (columns 10 and 11)
	# 	getSaliencyMap returns a pointer to the raw saliency video binary data
	sal_map_p = helper.getSaliencyMap(fix_list, dim, savename,
			# If a bin file exists at this location we load the saliency data from it, unless force_generate is True. Saliency will be saved if caching is True
			path_save=PATH_OUT,
			# Sigma of the 2D Gaussian drawn at the location of fixations
			gauss_sigma=2,
			# Array of ordered Int used to compute saliency maps at several intervals
			time_cut=frame_data,
			# Ask instead to return a path to a save bin file
			force_return_data=False,
			# Generate data instead of reading from pre-existing file
			force_generate=False,
			# Will save saliency to bin file (~1GB) to save RAM
			caching=True)

	# Append timecut data to fixation list
	start_frame = frame_data[fix_list[:, 5].astype(int)]
	end_frame = frame_data[fix_list[:, 6].astype(int)]+1
	fix_list = np.hstack([ fix_list, start_frame[:, None], end_frame[:, None] ])

	sal_map_ps.append(sal_map_p)
	fix_lists.append(fix_list)

	# Read one frame from file
	sal_map_ = np.fromfile(sal_map_p, count=np.prod(dim), dtype=np.float32)
	# Reshape flattened data to 3D saliency
	sal_map_ = sal_map_.reshape(dim)

	if sal_map is None:
		sal_map = sal_map_
	else:
		sal_map += sal_map_

sal_map /= sal_map.max()

# Save cumulated saliency map as a video
misc.printNorm("Saliency map to video frames.", verbose=0)
sal_generate.saveImages(sal_map, PATH_OUT+"videosal")

# Save saliency map as an image blended with a stimulus background
misc.printNorm("Saliency map to blended video frames.", verbose=0)
sal_generate.saveImages(sal_map, PATH_OUT+"videosal"+"_blend", blend=PATH_STIM)

# Save saliency maps to a binary file (read with .utils.readOutFile.readBinarySaliencyMap)
misc.printNorm("Saliency map to binary file.", verbose=0)
sal_generate.saveBin(sal_map, outpath)

misc.printWarning("\nRun the following two commands to generate videos from the images in folders:",
	verbose=0, header=None)
misc.printWarning("ffmpeg -r 25 -i \"{0}videosal/\"%d*.png -crf 25 -pix_fmt yuv420p {0}/videosal.mp4".format(PATH_OUT),
	tab=1, verbose=0, header=None)
misc.printWarning("ffmpeg -r 25 -i \"{0}videosal_blend/\"%d*.jpg -crf 25 -pix_fmt yuv420p {0}/videosal_blend.mp4".format(PATH_OUT),
	tab=1, verbose=0, header=None)

# Compare

WEIGHTS = [1,0,1,1,1]

# Compare saliency/fixation videos
from Salient360Toolbox.comparison.dynamic import saliencyCompare as salmetr
salmetr.metrics.pop("InfoGain") # This measure is not yet implemented
from Salient360Toolbox.comparison.dynamic.scanpathCompare import compareScanpath
from Salient360Toolbox.comparison.scanpathCompare import transformScanpath, measureNames

out_result = open("{0}{1}comparisons.csv".format(PATH_OUT, os.sep),  "a+")

from multiprocessing import Pool, cpu_count
from functools import partial

for i1, i2 in [[0, 1], [0, 2], [1, 2]]:

	results = {}

	name1 = misc.getFileName(files[i1])
	name2 = misc.getFileName(files[i2])

	misc.printNorm(name1, name2)

	# Opened BufferReaders
	salmap1_p = sal_map_ps[i1]
	salmap2_p = sal_map_ps[i2]

	FL1 = fix_lists[i1][:, [0,1, 7,8]]
	FL2 = fix_lists[i2][:, [0,1, 7,8]]

	misc.printNorm("Computing saliency similarity metrics", verbose=0)

	# Compare data over windows of N milliseconds
	fPool = salmetr.getFramePoolingIdx(2000, FRAME_COUNT)
	# Run comparisons in parallel
	with Pool(cpu_count()) as p:
		func = partial(salmetr.getWindValue,
				salmap1_p, salmap2_p, FL1, FL2, DIM,fPool)
		result = p.map(func, range(fPool.shape[0]))

	result = sum([], result)

	win_values = {key:[] for key in salmetr.metrics.keys()}
	# Cumulate similarity metrics over frames
	for values in result:
		for key, val in values.items():
			win_values[key].append(val)

	for key, val in win_values.items():
		results[key] = np.nanmean(val)

	SP1 = fix_lists[i1][:, [0, 3,4, 0]] # -fix_idx-, long, lat (gaze), -timestamp-
	SP2 = fix_lists[i2][:, [0, 3,4, 0]]

	# We do not have timestamp information, so we add NaN so that it is not used in calculations
	SP1[:, 3] = np.nan
	SP2[:, 3] = np.nan

	SP1[:, 1:3] *= [np.pi*2, np.pi] # Un-normalise
	SP2[:, 1:3] *= [np.pi*2, np.pi]

	SC1 = transformScanpath(SP1, [0, SP1.shape[0]], 0)
	SC2 = transformScanpath(SP2, [0, SP2.shape[0]], 0)

	# Adding frame data back (become columbs 11 and 12)
	SC1 = np.hstack([SC1, fix_lists[i1][:, 7:9]])
	SC2 = np.hstack([SC2, fix_lists[i2][:, 7:9]])

	misc.printNorm("Computing MultiMatch dissimilarity metrics", verbose=0)
	score, scores = compareScanpath(SC1, SC2,
		# Aggregate fix_list data over 2 seconds (50 * .04s/frame = 2s)
		frame_window_size=50,
		weight=WEIGHTS)

	results["multimatch.wavg"] = score
	for i in range(len(scores)):
		results["multimatch."+measureNames[i]] = scores[i]

	for metric, value in results.items():
		out_line = "{},{},{},{:.5e}\n".format( name1, name2,  metric, value)
		out_result.write(out_line)

		misc.printNeutral("{},{},{},{:.3f}".format( name1, name2,  metric, value),
			tab=1, verbose=0)

out_result.close()
