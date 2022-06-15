#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

"""
python3 -m Salient360Toolbox.compare data/raw_gaze/rawDataStatic1.csv data/raw_gaze/rawDataStatic2.csv --all
python3 -m Salient360Toolbox.compare data/fix_list/fixList1.csv data/fix_list/fixList2.csv --all
"""

import numpy as np

from .CLI_options import compare_opt

args = compare_opt.Options()
args.parse()
opts = args.opts

from .utils.misc import *
from .utils.readOutFile import getBinFilename

from . import helper

file1 = opts.i[0]
file2 = opts.i[1]

# assertC(file1!=file2, "Paths to two different files must be provided.")
# Check that path exists
assertC(os.path.exists(file1), "Path #1 doesn't exist.")
assertC(os.path.exists(file2), "Path #2 doesn't exist.")
# Check that it points to a file
assertC(os.path.isfile(file1), "Path #1 to gaze data is not a file.")
assertC(os.path.isfile(file2), "Path #2 to gaze data is not a file.")

dim = [1000, 2000] # Y, X map dimensions

def getData(path, opts):
	gtype = helper.inferGazeType(path)
	if gtype is None:
		printError("Could not identify file as a raw gaze or fixation list file. File: {}".format(path))
		return None

	# Load as fixation list
	fix_list = None
	if gtype == "raw":
		_, fix_list = helper.loadRawData(path,
			# If gaze tracking, which eye to extract
			eye=opts.eye,
			# Gaze or Head tracking
			tracking=opts.tracking,
			# Resampling at a different sample rate?
			resample=opts.resample,
			# Filtering algo and parameters if any is selected
			filter=helper.filterSettings(opts),
			# Fixation identifier algo and its parameters
			parser=helper.parsingSettings(opts))
	# Load as raw gaze data
	elif gtype == "fixlist":
		fix_list = helper.loadFixlist(path)

	return fix_list

os.makedirs(opts.out, exist_ok=True)
save_file = "{0}{1}comparisons.csv".format(opts.out, os.sep)

data = {}
new_file = not os.path.exists(save_file)
with open(save_file, "a+") as out_result:
	if new_file: out_result.write("Name1, Name2, type, Metric, Val\n")
	# Extract data

	printNeutral("Extracting fixation lists", verbose=1)
	# fixation list
	fix_list1 = getData(file1, opts)
	fix_list2 = getData(file2, opts)

	name1 = getFileName(file1)
	name2 = getFileName(file2)

	results = {}

	# Compare saliency/fixation maps
	if opts.salmap:
		from .comparison.saliencyMetrics import metrics
		metrics.pop("InfoGain") # A baseline needs to be manually added

		salmap1 = helper.getSaliencyMap(fix_list1[:, [2,3,4, 0,1]], dim, name1,
			path_save=opts.out)
		salmap2 = helper.getSaliencyMap(fix_list2[:, [2,3,4, 0,1]], dim, name2,
			path_save=opts.out)

		from .comparison.saliencyCompare import compareSaliency

		fixmap1 = helper.getFixationMap(fix_list1[:, :2], dim)
		fixmap2 = helper.getFixationMap(fix_list2[:, :2], dim)

		printNeutral("Computing saliency similarity metrics", verbose=1)
		results = dict(results, **compareSaliency(salmap1, salmap2, fixmap1, fixmap2,
			basemap=None))

	# Compare scanpaths (fixation lists)
	if opts.scanp:
		from .comparison.scanpathCompare import transformScanpath, measureNames, compareScanpath

		scanp1 = fix_list1[:, [9, 0,1, 12]] # fix_idx, long, lat (gaze), timestamp
		scanp2 = fix_list2[:, [9, 0,1, 12]]

		scanp1[:, 1:3] *= [np.pi*2, np.pi]
		scanp2[:, 1:3] *= [np.pi*2, np.pi]

		SC1 = transformScanpath(scanp1, [0, scanp1.shape[0]], 0)
		SC2 = transformScanpath(scanp2, [0, scanp2.shape[0]], 0)

		printNeutral("Computing MultiMatch dissimilarity metrics", verbose=1)
		score, scores = compareScanpath(SC1, SC2, weight=opts.scanp_weight)

		results["multimatch.wavg"] = score
		for i in range(len(scores)):
			results["multimatch."+measureNames[i]] = scores[i]

	for metric, value in results.items():
		out_result.write("{},{},{},{:.5e}\n".format(
			name1, name2,  metric, value))

if not opts.save:
	if opts.salmap:
		paths = [getBinFilename(opts.out+os.sep+name1, dim, dtype="float32"),
				 getBinFilename(opts.out+os.sep+name2, dim, dtype="float32")]
		for path in paths:
			if os.path.isfile(path):
				os.remove(path)
