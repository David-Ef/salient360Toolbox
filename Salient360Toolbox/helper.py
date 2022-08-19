#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np

from .utils.misc import *

def getCleanHeaderList(path, delimiter=","):
	import re
	with open(path, "r") as file:
		header = file.readline().lower()
		header = re.sub("[^a-zA-Z{}]".format(delimiter), "", header) 
		header = np.array([name.strip() for name in header.split(delimiter)])
	return header

def getColumnIndex(header, targets):
	if type(targets) not in [list, str]:
		printError("[GazeData.getColumnIndex] argument \"targets\" must be a string or a list of string")
		return []
	if type(targets) is str: targets = [targets]
	# Convert all to str, just in case
	else: targets = list(map(str, targets))

	idx = np.where(np.isin(header, targets))[0]

	return idx[0] if len(idx) > 0 else None

def FindRawFeaturesByHeader(filepath, returnValid=False):
	header = getCleanHeaderList(filepath)

	indices = {}

	indices["ts"] = getColumnIndex(header, ["oculots", "oculotimestamp", "ocutimestamp", "etts", "timestamp", "ts"])

	indices["xRayL"] = getColumnIndex(header, ["leftgazex", "leftgazedirx", "lgazex", "xlgaze", "lefteyedirectionx", "leftgazedirectionx"])
	indices["yRayL"] = getColumnIndex(header, ["leftgazey", "leftgazediry", "lgazey", "ylgaze", "lefteyedirectiony", "leftgazedirectiony"])
	indices["zRayL"] = getColumnIndex(header, ["leftgazez", "leftgazedirz", "lgazez", "zlgaze", "lefteyedirectionz", "leftgazedirectionz"])

	indices["xRayR"] = getColumnIndex(header, ["rightgazex", "rightgazedirx", "rgazex", "xrgaze", "righteyedirectionx", "rightgazedirectionx"])
	indices["yRayR"] = getColumnIndex(header, ["rightgazey", "rightgazediry", "rgazey", "yrgaze", "righteyedirectiony", "rightgazedirectiony"])
	indices["zRayR"] = getColumnIndex(header, ["rightgazez", "rightgazedirz", "rgazez", "zrgaze", "righteyedirectionz", "rightgazedirectionz"])

	indices["xRayB"] = getColumnIndex(header, ["bingazex", "bingazedirx", "meangazedirx", "lgazex", "xlgaze", "meangazedirectionx", "meangazedirectionx"])
	indices["yRayB"] = getColumnIndex(header, ["bingazey", "bingazediry", "meangazediry", "lgazey", "ylgaze", "meangazedirectiony", "meangazedirectiony"])
	indices["zRayB"] = getColumnIndex(header, ["bingazez", "bingazedirz", "meangazedirz", "lgazez", "zlgaze", "meangazedirectionz", "meangazedirectionz"])

	indices["xCam"] = getColumnIndex(header, ["xcam", "camx", "headx", "xhead", "camerarotationx", "cameraquaternionx"])
	indices["yCam"] = getColumnIndex(header, ["ycam", "camy", "heady", "yhead", "camerarotationy", "cameraquaterniony"])
	indices["zCam"] = getColumnIndex(header, ["zcam", "camz", "headz", "zhead", "camerarotationz", "cameraquaternionz"])
	indices["wCam"] = getColumnIndex(header, ["wcam", "camw", "headw", "whead", "camerarotationw", "cameraquaternionw"])

	indices["piCam"] = getColumnIndex(header, ["pitch", "campitch", "pitchcam", "pitchead", "headpitch"])
	indices["yaCam"] = getColumnIndex(header, ["yaw", "camyaw", "yawcam", "yawhead", "headyaw"])
	indices["roCam"] = getColumnIndex(header, ["roll", "camroll", "rollcam", "rollhead", "headroll"])

	indices["valR"] = getColumnIndex(header, ["valr", "rval"])
	indices["valL"] = getColumnIndex(header, ["vall", "lval"])

	if returnValid:
		# test if there is enough data to continue

		# Do we have gaze-dir-relative-to-head data?
		dirDataR = (indices["xRayR"] is not None,
				indices["yRayR"] is not None,
				indices["zRayR"] is not None)
		dirDataL = (indices["xRayL"] is not None,
				indices["yRayL"] is not None,
				indices["zRayL"] is not None)
		dirDataB = (indices["xRayB"] is not None,
				indices["yRayB"] is not None,
				indices["zRayB"] is not None)

		# Do we have head rotations as quaternions?
		CamQuat = (indices["xCam"] is not None,
				indices["yCam"] is not None,
				indices["zCam"] is not None,
				indices["wCam"] is not None)

		# Do we have head rotations as Euler angles?
		CamEuler = (indices["piCam"] is not None,
				indices["yaCam"] is not None,
				indices["roCam"] is not None)

		valid =  {"eye": {"L": np.all(dirDataL),
						  "R": np.all(dirDataR),
						  "B": (np.all(dirDataL) and np.all(dirDataR)) or np.all(dirDataB),
						  },
				  "head": {"Q": np.all(CamQuat),
				  		   "E": np.all(CamEuler)
				  		  }
				  }
		
		return indices, valid

	return indices

def FindFixlistFeaturesByHeader(filepath, returnValid=False):
	header = getCleanHeaderList(filepath)

	indices = {}

	# target data column positions in text file
	indices["lon"] = getColumnIndex(header, ["lon", "longitude", "longaze", "longgaze"])
	indices["lat"] = getColumnIndex(header, ["lat", "latitude", "latgaze"])

	indices["x"] = getColumnIndex(header, ["x", "xsph", "xgaze"])
	indices["y"] = getColumnIndex(header, ["y", "ysph", "ygaze"])
	indices["z"] = getColumnIndex(header, ["z", "zsph", "zgaze"])

	indices["ts"] = getColumnIndex(header, ["time", "starttimestamp", "timestamp", "timestart"])
	indices["dur"] = getColumnIndex(header, ["dur", "duration"])
	indices["idx"] = getColumnIndex(header, ["idx", "index", "i"])

	if returnValid:
		# test if there is enough data to continue

		# Do we have long/lat data?
		valid = not (indices["lon"] is None and indices["lat"] is None)
		# Do we have a 3D direction vector?
		valid |= not (indices["x"] is None and indices["y"] is None and indices["z"] is None)
	
		return indices, valid

	return indices

def filterSettings(opts):
	filterName = opts.filter
	filterParam = {}

	if filterName is None: return {"name": "None", "params": None}

	filter_opt = {}
	for filt_opt in opts.filter_opt:
		filt_opt = filt_opt.split("=")
		if len(filt_opt) == 2:
			filter_opt[filt_opt[0]] = float(filt_opt[1])

	name = filterName[0].lower()
	if name == "g":
		filterParam["sigma"] = filter_opt["sigma"] if "sigma" in filter_opt.keys() else 4
	elif name == "s":
		filterParam["win"] = filter_opt["win"] if "win" in filter_opt.keys() else 9
		filterParam["poly"] = filter_opt["poly"] if "poly" in filter_opt.keys() else 2

	return {"name": filterName, "params": filterParam}

def parsingSettings(opts):
	pars_opts = ""
	if opts.IVT is not None:
		parserName = "I-VT"
		pars_opts = opts.IVT
	elif opts.IHMM is not None:
		parserName = "I-HMM"
		pars_opts = opts.IHMM
	elif opts.ICT is not None:
		parserName = "I-CT"
		pars_opts = opts.ICT
	elif opts.IDT is not None:
		parserName = "I-DT"
		pars_opts = opts.IDT
	else:
		return {"name": "I-VT", "params": {"threshold": 120}}

	parserParam = {}
	for pars_opt in pars_opts:
		pars_opt = pars_opt.split("=")
		if len(pars_opt) == 2:
			parserParam[pars_opt[0]] = float(pars_opt[1])

	return {"name": parserName, "params": parserParam}

def loadRawData(path, eye="R", return_fixlist=True, **kwargs):
	idx, valid = FindRawFeaturesByHeader(path, returnValid=True)

	idx_load = []
	# 0: timestamp
	idx_load += [idx["ts"]]
	# 1,2,3,4: head rotations
	Euler2Quat = False
	if not valid["head"]["Q"]:
		# Will transform Euler angles to quaternion in next phase
		Euler2Quat = True
		idx_load += [idx["piCam"], idx["yaCam"], idx["roCam"], idx["roCam"]]
	else:
		idx_load += [idx["wCam"], idx["xCam"], idx["yCam"], idx["zCam"]]
	# 5,6,7: unit gaze direction in 3D world (x,y,z)
	if eye=="R":
		idx_load += [idx["xRayR"], idx["yRayR"], idx["zRayR"]]
	elif eye=="L":
		idx_load += [idx["xRayL"], idx["yRayL"], idx["zRayL"]]
	elif eye=="B":
		if valid["eye"]["B"]:
			idx_load += [idx["xRayB"], idx["yRayB"], idx["zRayB"], idx["xRayB"], idx["yRayB"], idx["zRayB"]]
		elif valid["eye"]["R"] and valid["eye"]["L"]:
			idx_load += [idx["xRayR"], idx["yRayR"], idx["zRayR"], idx["xRayL"], idx["yRayL"], idx["zRayL"]]
		elif valid["eye"]["R"]:
			idx_load += [idx["xRayR"], idx["yRayR"], idx["zRayR"], idx["xRayR"], idx["yRayR"], idx["zRayR"]]
		elif valid["eye"]["L"]:
			idx_load += [idx["xRayL"], idx["yRayL"], idx["zRayL"], idx["xRayL"], idx["yRayL"], idx["zRayL"]]
	if eye=="H":
		idx_load += [0, 0, 0] # Will not be used

	kwargs["Euler2Quat"] = Euler2Quat

	"""
	Feature order:
		0: timestamp
		1,2,3,4: camera rotation as quaternion (or Euler) (W,X,Y,Z)
		5,6,7: gaze direction vector relative to camera rotation (X,Y,Z)
	"""
	with open(path, "r") as f:
		l = f.readline().strip()
		l = l.split(",")
		printNeutral("Loading the following columns from raw data files:\n{}".format([l[i] for i in idx_load]), header="Func loadRawData")

	raw_data = np.loadtxt(path,
		# Columns are separated by a comma
		delimiter=",",
		# Extract columns identified previously
		usecols=idx_load,
		# We suppose that the first row is the header, we skip it
		skiprows=1)
		#.astype(np.float32) # Will cut timestamp precision and produce duplicate samples

	if return_fixlist:
		return getFixationList(raw_data, eye=eye, **kwargs)
		
	return raw_data

def loadFixlist(path):
	idx = FindFixlistFeaturesByHeader(path)

	if None in [idx["lon"], idx["lat"]]:
		return -1

	# Re-organize fix_list file to contain lon, lat, x, y, z, timestamp
	fix_list = np.loadtxt(path, delimiter=",", skiprows=1)
	# Support fixation lists with only one fixation
	if len(fix_list.shape) == 1:
		fix_list = fix_list[None, :]

	# Rearrange columns
	fix_list_tmp = fix_list[:, [idx["lon"], idx["lat"]] ]

	if None in [idx["x"], idx["y"], idx["z"]]: # Project lat/long on unit sphere
		X = np.sin(fix_list[:, idx["lat"]]*np.pi) * np.cos( (1-fix_list[:, idx["lon"]])*(2*np.pi))
		Y = np.sin(fix_list[:, idx["lat"]]*np.pi) * np.sin( (1-fix_list[:, idx["lon"]])*(2*np.pi))
		Z = np.cos(fix_list[:, idx["lat"]]*np.pi)

		X = X[:, None]
		Y = Y[:, None]
		Z = Z[:, None]
	else:
		X = fix_list[:, idx["x"], None]
		Y = fix_list[:, idx["y"], None]
		Z = fix_list[:, idx["z"], None]

	if idx["ts"] is None:
		ts = np.zeros(fix_list.shape[0])[:, None]
	else:
		ts = fix_list[:, idx["ts"], None]

	if idx["idx"] is None:
		IDX = np.arange(fix_list.shape[0])[:, None]
	else:
		IDX = fix_list[:, idx["idx"], None]

	nrows = fix_list_tmp.shape[0]
	# Put all info together
	fix_list_tmp = np.concatenate([
		fix_list_tmp,
		X, Y, Z,
		np.zeros([nrows, 4]),
		IDX,
		np.zeros([nrows, 2]),
		ts,
		np.zeros([nrows, 16])
		], axis=1)

	return fix_list_tmp

def getFixationList(raw_data,
	# Head trajectory parameter
	tempWindowSize=100,
	# Gaze or Head tracking
	tracking="HE",
	# If gaze tracking, which eye to extract
	eye=None,
	# Resampling at a different sample rate?
	resample=None,
	# Filtering algo and parameters if any is selected
	filter=None,
	# Fixation identifier algo and its parameters
	parser=None,
	# Are rotation data expressed as Euler angle (would require converting)
	Euler2Quat=False,
	# Only process data in this range
	data_range=None,
	# Progress bar callback
	callback=lambda *a: None,
	# Return fixation/saccade label list
	return_label=False,
	# Return sample indices kept after velocity pruning
	return_keep=False,
	# Return velocity signal
	return_velocity=False,
	**kwargs):

	from . import processing
	from .processing.identify import I_algos as parsers

	if data_range is None:
		data_range = [0, np.inf]
	if filter is None:
		filter = {"name": "None"}
	if parser is None:
		parser = {"name": "I-VT", "params": {"threshold": 120}}

	if type(raw_data) != np.ndarray:
		printError("Argument \"raw_data\" must be of type str or numpy.ndarray. Got \"{}\"".format(type(raw_data)), verbose=0)

		ret = [None, None]
		if return_label: ret.append(None)
		if return_keep: ret.append(None)
		if return_velocity: ret.append(None)
		return ret

	if Euler2Quat:
		callback(0, "Converting Euler angles to Quaternions.")
		printNeutral("Converting Euler angles to Quaternion. Assuming data comes from unity!")
		processing.preprocess.convertEulerToQuat(raw_data[:, 1:5])

	# By default we process the entire file, but you can restrict to a certain data range
	if data_range[0] < 0 or data_range[0] > (raw_data.shape[0]-1):
		data_range[0] = 0
	if data_range[1] < 0 or data_range[1] > (raw_data.shape[0]):
		data_range[1] = raw_data.shape[0]

	# Raw gaze data to gaze samples projected on unit sphere (among other things)
	gaze_data = processing.preproc(raw_data,
			# Start and Nsample in raw_data
			data_range,
			resample=resample,
			callback=callback,
			Euler2Quat=Euler2Quat,
			# Binocular data requires taking the average of left and right data samples
			eye=eye,
			**kwargs)

	if gaze_data is None:
		ret = [None, None]
		if return_label: ret.append(None)
		if return_keep: ret.append(None)
		if return_velocity: ret.append(None)
		return ret

	velocity = keep = None
	if tracking == "H":
		# Get head exploration trajectory by downsampling gaze samples with a moving window
		label_list = processing.preprocess.downsample2Centroid(gaze_data,
			tempWindowSize=tempWindowSize)
		# Assign head position as gaze
		gaze_data[:, :3] = gaze_data[:, 6:9]
		# Zero out eye direction vectors
		gaze_data[:, 3:6] = 0
	else:
		if callback is not None:
			callback(0, "Identifying saccades\nMay take a minute or two.")
		printNeutral("Identifying saccades. It may take a minute or two.", tab=1, verbose=1)

		# Will compute velocity and remove outlier samples
		keep, velocity = processing.identify.commons.getVelocity(gaze_data[:, [0,1,2, 9]],
			return_keep=return_keep)
		# Remove nan and outlier samples

		if keep is not None:
			gaze_data = gaze_data[keep]
			velocity = velocity[keep]

		# Smooth velocity
		if filter["name"][0].lower() == "g":
			from scipy.ndimage.filters import gaussian_filter1d
			velocity = gaussian_filter1d(velocity, filter["params"]["sigma"])
		elif filter["name"][0].lower() == "s":
			from scipy.signal import savgol_filter
			velocity = savgol_filter(velocity, int(filter["params"]["win"]), int(filter["params"]["poly"]))

		# Label each gaze points as fixation or saccade (or blink)
		if parser["name"] == "I-VT":
			label_list = parsers[parser["name"]](gaze_data[:, 9], velocity,
				**parser["params"], callback=callback)
		elif parser["name"] == "I-HMM":
			label_list = parsers[parser["name"]](velocity,
				**parser["params"], callback=callback)
		elif parser["name"] == "I-CT":
			label_list = parsers[parser["name"]](gaze_data[:, [0,1,2, 9]],
				**parser["params"], callback=callback)

	# Get scanpath information as a list of saccade/fixation features
	#	Velocity is necessary for saccade peak velocity feature
	fix_list = processing.extract.getGazeFeatures(gaze_data, label_list,
		velocity=velocity)

	ret = [gaze_data, fix_list]
	if return_label: ret.append(label_list)
	if return_keep: ret.append(keep)
	if return_velocity: ret.append(velocity)
	return ret

def getSaliencyMap(fix_list, dim,
	# Name of binary saliency file saved for caching purposes
	name="tmp",
	# Sigma of the 2D Gaussian drawn at the location of fixations
	gauss_sigma=2,
	# Path to save binary data
	path_save=None,
	# Array of ordered Int used to compute saliency maps at several intervals
	time_cut=None,
	# Return data on top of potentially saving data to a file
	force_return_data=False,
	# If a binary file with the same name already exists, will generate it again
	force_generate=False,
	# Should we cache to file or generate saliency everytime?
	caching=False):
	"""
	DOC
	"""
	# If path to a file (salmap) is provided we return its content or a pointer to it
	import numpy as np
	import os

	if not force_generate and path_save is not None:
		from .utils.readOutFile import getBinFilename
		path_file = getBinFilename(path_save+os.sep+name, dim, dtype="float32")

		if os.path.exists(path_file):
			if time_cut is None and force_return_data:
				# return data
				from .utils.readOutFile import readBinarySaliencyMap
				return readBinarySaliencyMap(path_file)
			else:
				return path_file

	dim = np.array(dim, dtype=int)

	# Otherwise, we create a new array and compute saliency map(s)
	sal_map = np.zeros(dim, dtype=np.float32) # Y, X

	if time_cut is None:
		from .generation.saliency import getSaliency
	else:
		from .generation.saliency import getSaliencyDyn as getSaliency

	getSaliency(sal_map, fix_list, gauss_sigma=gauss_sigma, time_cut=time_cut)

	if time_cut is not None or (path_save is not None and caching):
		from .generation.saliency import saveBin
		path_file = path_save+os.sep+name
		path_file = saveBin(sal_map, path_file)

	if time_cut is not None and not force_return_data:
		return path_file

	return sal_map

def getFixationMap(fix_list, dim):
	from .generation.scanpath import toFixationMap

	return toFixationMap(fix_list, dim)
