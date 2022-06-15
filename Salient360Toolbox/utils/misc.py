#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import os, shutil, builtins

if not hasattr(builtins, "verbose"): builtins.verbose = 2

def getVideoFrame(path_video, frame=0, crash=True):
	"""
	DOC
	"""
	import cv2
	# Check mimeType of blended content - expects "video"
	cont_type = getMimeType(path_video)
	assertC(cont_type.split("/")[0] == "video",
		"You need to pass a path to a video file to function \"blendFrames\". Got type \"{}\".".format(cont_type),
		crash=crash)

	# Open video file
	vid = cv2.VideoCapture()
	vid.open(path_video)

	assertC(vid.isOpened(), "Couldn't open video: \"{}\"".format(path_video), printIfFail=True, crash=crash)

	# Extract video specs.
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

	if type(frame) == float:
		if frame > 1: frame = int(frame)
		else: frame = int(frame * length)

	# Clip frame index
	if frame < 0: frame = 0
	if frame >= length: frame = length-1

	vid.set(cv2.CAP_PROP_POS_FRAMES, frame);
	ret, frame = vid.read()
	vid.release()

	return frame

def getMimeType(path):
	"""
	Return file Media type at path
	"""
	from mimetypes import MimeTypes
	return MimeTypes().guess_type(path)[0]

def inferGazeType(path):
	"""
	Read first line in file (header) and try to infer data type from column names
	"""
	import numpy as np
	from ..helper import FindRawFeaturesByHeader, FindFixlistFeaturesByHeader

	rawTags, rawValids = FindRawFeaturesByHeader(path, returnValid=True)
	if sum(rawValids["eye"].values()) > 0 and sum(rawValids["head"].values()) > 0:
		return "raw"

	fixTags, fixvalid = FindFixlistFeaturesByHeader(path, returnValid=True)
	if fixvalid:
		return "fixlist"

	return None

def getTerminalWidth():
	"""
	Return the current wdith of the terminal/shell
	"""
	return shutil.get_terminal_size()[0]

def getFileName(file):
	"""
	Return file name without path or extension
	"""
	# get filename
	name = os.path.basename(file)
	# remove file extension
	return os.path.splitext(name)[0]

def getDummyFixData(N=11):
	"""
	Return dummy a list of fixation positions traversing the sphere from top to bottom
	For testing purposes.
	"""
	import numpy as np

	fixList = np.zeros([0, 13])
	lon = 0
	lat = -np.pi/2
	for _ in np.arange(N+1):

		fixationPt = np.zeros(13)
		fixationPt[0] = np.cos(lat) * np.cos(lon)
		fixationPt[1] = np.cos(lat) * np.sin(lon)
		fixationPt[2] = np.sin(lat)

		fixList = np.concatenate([ fixList, fixationPt.reshape([1, fixationPt.shape[0]]) ])
		lon += np.pi*2 / N
		lat += np.pi / N

	return fixList

def clearline(ngoup=0):
	"""Clear last line with spaces
	*ngoup*: relocate cursor ngoup lines before erasing. Useful when last stdout ended with a line return."""
	print("{}\r{}\r".format("\033[F"*ngoup, " "*getTerminalWidth()), end="")

def clearlines(nlines):
	"""Clear n lines and set cursor at the start of the n-th line"""
	for i in range(nlines):
		print("\r{}{}".format(" "*getTerminalWidth(), "\033[F"), end="")
	print("\r{}\r".format(" "*getTerminalWidth()), end="")

def getEmptyObject():
	"""
	DOC
	"""
	return type('', (), {})()

def printP(*args):
	"""
	DOC
	"""
	return os.sep.join(args)

def printNorm(*args, bold=False, verbose=2, **kwargs):
	"""
	DOC
	"""
	printC(*args, color="21", bold=bold, verbose=verbose, **kwargs)

def printNeutral(*args, header=None, bold=True, verbose=2, **kwargs):
	"""
	DOC
	"""
	if header is not None and os.name != "nt" and header != "":
		header = "\033[1;4m{}:\033[m\033[94m".format(header)
		args = header, *args
	printC(*args, color="94", bold=bold, verbose=verbose, **kwargs)

def printWarning(*args, header="Warning", bold=True, verbose=1, **kwargs):
	"""
	DOC
	"""
	if header is not None and os.name != "nt" and header != "":
		header = "\033[1;4m{}:\033[m\033[33m".format(header)
		args = header, *args
	printC(*args, color="33", bold=bold, verbose=verbose, **kwargs)

def printError(*args, header="Error", bold=True, verbose=0, exit_=False, **kwargs):
	"""
	DOC
	"""
	if header is not None and os.name != "nt" and header != "":
		header = "\033[1;4m{}:\033[m\033[91m".format(header)
		args = header, *args
	if exit_: verbose = -1
	printC(*args, color="91", verbose=verbose, bold=bold, exit_=exit_, **kwargs)

def printSuccess(*args, bold=True, verbose=1, **kwargs):
	"""
	DOC
	"""
	printC(*args, color="92", bold=bold, verbose=verbose, **kwargs)

def printC(*args, color="21", bold=False, tab=0, clear=False, verbose=1, exit_=False, sep=" ", **kwargs):
	"""
	DOC
	"""
	# Verbose level
	if verbose != -1 and hasattr(builtins, "verbose") and verbose > builtins.verbose:
		return
	# Clear line before?
	if clear:
		cl = "\r" + " "*getTerminalWidth() + "\r"
	else:
		cl = ""

	if type(tab) == float:
		tab = " "*int(tab)
	else:
		tab = " "*(tab*2)

	if os.name == "nt":
		print("{}{}{}".format(
				cl,
				" "*(tab*2),
				sep.join(map(str, args))
				),
			**kwargs)
	else:
		print("{}{}\033[{}m\033[{}m{}\033[m".format(
				cl,
				tab,
				int(bold),
				color,
				sep.join(map(str, args))
				),
			**kwargs)

	if exit_: exit()

def assertC(bool_, message, printIfFail=True, crash=True):
	"""
	DOC
	"""
	if bool_:
		if not printIfFail:
			printSuccess("Assert passed: ", end="\n\t", verbose=-1)
			printSuccess(message, bold=False, verbose=-1)
	else:
		printError("Assert failed: ", verbose=-1, header="")
		printError(message, tab=1, bold=False, verbose=-1, header="")
		if crash: exit()
