#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import os, builtins

os.environ["NUMBA_DISABLE_JIT"] = "1" # Set to 0 to enable numba
# Only print most important messages
builtins.verbose = 0

try:
	import Salient360Toolbox
except:
	import sys
	sys.path.append("../")

PATH_DATA = "../data/raw_gaze/rawDataStatic1.csv"

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
dim = [500, 1000]

from Salient360Toolbox.helper import loadRawData

# Load and process raw data
gaze_data, fix_list, label_list = loadRawData(PATH_DATA,
	# If gaze tracking, which eye to extract
	eye=eye,
	# Gaze or Head tracking
	tracking=tracking,
	# Resampling at a different sample rate?
	resample=resample,
	# Filtering algo and parameters if any is selected
	filter=filterSettings,
	# Fixation identifier algo and its parameters
	parser=parsingSettings,
	# Return fixation/saccade label list
	return_label=True)

from Salient360Toolbox import processing
velocity = processing.identify.commons.getVelocity(gaze_data[:, [0, 1, 2, 9]])

from Salient360Toolbox.utils import plot

# Compare velocity of head, gaze and head+gaze movement signals
plot.displayAllData(gaze_data, labels=label_list)

# Plot raw gaze data on unit sphere
plot.displayGazeSphere(gaze_data[:, :3], label_list)

# Example use of "utils.divergingColorMaps" script
plot.colormap()

# Compare a quasi-uniform sammling of the sphere and a sine 
#     function used to weight equirectangular saliency maps 
plot.equirectangularWeighting()
