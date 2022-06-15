#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

import numpy as np
import cv2

from ..utils.misc import *

scanpath_header = "long_Gaze,lat_Gaze,X_gaze,Y_gaze,Z_gaze,long_Eye,lat_Eye,long_Head,lat_Head,Fix_index,Fix_idx_start,Fix_idx_end,Fix_time_start,Fix_time_end,Fix_duration,Fix_Dispersion,Fix_peak_vel,Fix_peak_acc,Sacc_peak_vel,Sacc_peak_acc,Sacc_ampl_Gaze,Sacc_ampl_Eye,Sacc_ampl_Head,Sacc_absAngle_Gaze,Sacc_absAngle_Eye,Sacc_absAngle_Head,Sacc_relAngle_Gaze,Sacc_relAngle_Eye,Sacc_relAngle_Head"
scanpath_fmt = "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%i,%i,%i,%i,%i,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e"
def toFile(fix_list, output_name, mode="w", saveArr=None, header=None, fmt=None):
	"""
	Save fixation list to text file.
	Parameter `saveArr` specifies the fixation/saccade features to save.
	Full list of fixation/saccade data:
		0: mean longitude during fixation: Gaze
		1: mean latitude during fixation: Gaze
		2: mean gaze position on Gaze as a unit vector (X)
		3: mean gaze position on Gaze as a unit vector (Y)
		4: mean gaze position on Gaze as a unit vector (Z)
		5: mean longitude during fixation: Eye
		6: mean latitude during fixation: Eye
		7: mean longitude during fixation: Head
		8: mean latitude during fixation: Head
		9: fixation index
		10: index of first sample in this fixation
		11: index of last sample in this fixation
		12: start timestamp (msec)
		13: end timestamp (msec)
		14: fixation duration (msec)
		15: mean fixation dispersion (rad)
		16: peak fixation velocity (rad/sec)
		17: peak fixation acceleration (rad/sec/sec)
		18: peak saccade velocity (rad/sec)
		19: peak saccade acceleration (rad/sec/sec)
		20: saccade amplitude: Gaze (rad.)
		21: saccade amplitude: Eye  (rad.)
		22: saccade amplitude: Head (rad.)
		23: absolute angle between a saccade and the longitudinal axis: Gaze (rad.)
		24: absolute angle between a saccade and the longitudinal axis: Eye (rad.)
		25: absolute angle between a saccade and the longitudinal axis: Head (rad.)
		26: relative angle between two consecutive saccades: Gaze (rad.)
		27: relative angle between two consecutive saccades: Eye (rad.)
		28: relative angle between two consecutive saccades: Head (rad.)
	"""
	if saveArr is None: saveArr = [9, 0, 1, 12]
	if header is None: header = scanpath_header.split(",")
	if fmt is None: fmt = scanpath_fmt.split(",")

	if len(saveArr) == 0:
		printError("The array of feature index (saveArr) passed to this function cannot be empty.")
		return

	header = ", ".join([header[i].strip() for i in saveArr])
	fmt = ",".join([fmt[i] for i in saveArr])

	if mode[0] == "a":
		header = ""

	with open(output_name, mode) as f:
		np.savetxt(f,
			fix_list[:, saveArr],
			header=header,
			fmt=fmt)

def toFixationMap(fix_list, map_res):
	"""
	Expects: latitudes (Y), longitudes (X)
	A fixation map counts fixations per pixels
	Returns a fixation map (dtype: int)
	"""
	map_res = np.array(map_res).astype(int)

	# lat, long provided by default are normalized [0, 1]
	fix_list = fix_list.copy() * np.array(map_res[::-1])-1
	# Create new fix_map
	fix_map = np.zeros(map_res)
	# Get unique fixation position and their individual hit count
	pos, val = np.unique(fix_list.astype(int), return_counts=True, axis=0)
	# Add hit count to fixation position
	fix_map[pos[:, 1], pos[:, 0]] = val

	return fix_map

def toImage(fix_list, img_res, output_name,
	text=True, link=False, shadow=False,
	ptsSize=3,
	extension="png",
	blend=None):
	"""
	Draw coloured fixation map.
	"""

	img_res = np.array(img_res, dtype=int)

	if blend is None:
		img = np.zeros([*img_res, 3])
	else:
		assertC(type(blend) in [np.ndarray, str], "Argument \"blend\" must be either a numpy array or a path to a media (image, video). Got \"{}\"".format(type(blend)))
		if type(blend) != np.ndarray:
			if type(blend) == str:
				mimetype = getMimeType(blend).split("/")[0]
				if mimetype == "image":
					img = cv2.imread(blend, cv2.IMREAD_COLOR)
				elif mimetype == "video":
					img = getVideoFrame(blend, .1)
				else:
					printError("Could not open media at location [\"{}\"]".format(blend))
					return
		else:
			img = np.ascontiguousarray(blend)

		if np.any(np.array(img_res) != np.array(img.shape[:-1])):
			img = cv2.resize(img, tuple(img_res)[::-1])

	posS = (fix_list * [*img.shape[1::-1]]).astype(int)

	for iPos in range(posS.shape[0]):
		if shadow:
			cv2.circle(img=img,
				center=(posS[iPos, 0]-2, posS[iPos, 1]+2),
				radius=int(ptsSize*4.3),
				color=(0,0,0),
				thickness=-1,
				lineType=16)
		cv2.circle(img=img,
			center=(posS[iPos, 0], posS[iPos, 1]),
			radius=ptsSize*4,
			color=(255 - iPos/posS.shape[0]*255, 58, iPos/posS.shape[0]*255),
			thickness=-1,
			lineType=16)

		if text:

			img = cv2.putText(img=img,
				text=str(iPos+1),
				org=(posS[iPos, 0], posS[iPos, 1]),
				fontFace=cv2.LINE_AA,
				fontScale=.7,
				color=(0,0,0),
				thickness=3)

			img = cv2.putText(img=img,
				text=str(iPos+1),
				org=(posS[iPos, 0], posS[iPos, 1]),
				fontFace=cv2.LINE_AA,
				fontScale=.7,
				color=(255 - iPos/posS.shape[0]*255, 58, iPos/posS.shape[0]*255),
				thickness=2)

		if link and iPos > 0:
			if shadow:
				cv2.line(img,
					(posS[iPos-1, 0]-2, posS[iPos-1, 1]+2),
					(posS[iPos, 0]-2, posS[iPos, 1]+2),
					color=(0,0,0),
					thickness=4,
					lineType=16)

			cv2.line(img,
				(posS[iPos-1, 0], posS[iPos-1, 1]),
				(posS[iPos, 0], posS[iPos, 1]),
				color=(255 - iPos/posS.shape[0]*255, 58, iPos/posS.shape[0]*255),
				thickness=3,
				lineType=16)

	cv2.imwrite('{}.{}'.format(output_name, extension), img)
	
scanpath_info = toFile.__doc__.split("\n")[4:-1]
