#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np
import os, re

def getBinFilename(name, shape, type_="salmap", dtype="float32"):
	"""
	DOC
	"""
	# if n_frame is None: # static
	# 	return "{}_{}x{}_{}b_{}.bin".format(name, *dim[::-1], dtype[-2:], type_)
	# else: # dynamic
	# 	return "{}_{}x{}x{}_{}b_{}.bin".format(name, *dim[::-1], n_frame, dtype[-2:], type_)

	return "{}_{}_{}b_{}.bin".format(
			name,
			"x".join([str(dim) for dim in shape[::-1]]),
			dtype[-2:], type_)

get_binsalmap_infoRE = re.compile("([\w\d_]+)_(\d+)x(\d+)_(\d+)b(?:_(\w+))?.bin")
get_vbinsalmap_infoRE = re.compile("([\w\d_]+)_(\d+)x(\d+)x(\d+)_(\d+)b(?:_(\w+))?.bin")

def extractFileInfo(file):
	"""
	DOC
	"""
	from .misc import printError

	filename = os.path.basename(file)

	# Static stimulus file name
	info = list(get_binsalmap_infoRE.findall(filename.split(os.sep)[-1]))
	if len(info) > 0 and len(info[0]) == 5:
		info = list(info[0])
		info.insert(3, 1)
		return [info[0], *list(map(int, info[1:-1])), info[-1]]
	
	# Dynamic stimulus file name
	info = list(get_vbinsalmap_infoRE.findall(filename.split(os.sep)[-1]))
	if len(info) > 0 and len(info[0]) == 6:
		info = info[0]
		return [info[0], *list(map(int, info[1:-1])), info[-1]]

	printError("""File [\"{}\"] information couldn't be parsed.
We expect the following formats:
  static stimuli: NAME_01x02_03b.bin where 01: width, 02: height, 03: bit precision
  dynamic stimuli: NAME_01x02x03_04b.bin where 01: width, 02: height, 03: frame count, 04: bit precision
""")
	return None

def readBinarySaliencyMap(path_file, i_frame=0):
	"""
	DOC
	"""
	dtypes = {16: np.float16,
			  32: np.float32,
			  64: np.float64}

	name, width, height, _, dtype, type_ = extractFileInfo(path_file)

	with open(path_file, "rb") as f:
		# Position read pointer right before target frame
		f.seek(width*height * i_frame * (dtype//8))

		# Read one frame from file
		data = np.fromfile(f, count=width*height, dtype=dtypes[dtype])
		# Reshape flattened data to 2D image
		data = data.reshape([height, width])

	return data

def readBinaryFixMap(path_file, i_frame=0):
	"""
	DOC
	"""
	pass
