#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2020
# Lab: SGL, Goethe University, Frankfurt
# Comment: 
# ---------------------------------

import numpy as np
import numba

try:
	import quaternion as quat
except:
	print("Quaternion module not found")
	print("\trun `pip install numpy-quaternion`")
	exit()

from ..utils.misc import *

class VPdata():
	def __init__(self, angleSpan, VPmap):
		self.angleSpan = angleSpan
		self.VPmap = VPmap

	@staticmethod
	def htcvive16():
		# HTC Vive headset dimensions (2016 release)
		return VPdata([110, 105], [2160, 1200])
	@staticmethod
	def oculus16():
		# Oculus Rift headset dimensions (2016 customer release)
		return VPdata([80, 90], [2160, 1200])
		# return VPdata([95, 106], [1920, 1080]) # Oculus Dev1?

@numba.njit
def convertEulerToQuat(data):
	# In Unity Euler angles are reported in degrees by default
	if np.any(np.abs(data[:, :3]) > (2*np.pi)):
		# Transform to radians
		data[:, :4] = np.deg2rad(data[:, :4])

		# TODO: error here
		# deg2rad would copy and original data object would not be modified
		# Where is deg2rad applied then?

	# from: github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
 	# pitch
	X = data[:, 0]
	# yaw
	Y = data[:, 1]
	# roll
	Z = data[:, 2]

	c1 = np.cos(X/2)
	c2 = np.cos(Y/2)
	c3 = np.cos(Z/2)

	s1 = np.sin(X/2)
	s2 = np.sin(Y/2)
	s3 = np.sin(Z/2)

	#YXZ
	data[:, 0] = c1 * c2 * c3 + s1 * s2 * s3 # W
	data[:, 1] = s1 * c2 * c3 + c1 * s2 * s3 # X
	data[:, 2] = c1 * s2 * c3 - s1 * c2 * s3 # Y
	data[:, 3] = c1 * c2 * s3 - s1 * s2 * c3 # Z

def interpolate_raw(gaze_data, validity, samplingRate=120):
	from scipy import interpolate

	ET  = slice(5, 8, None)
	HMD = slice(1, 5, None)

	repeatET = np.all(gaze_data[1:, ET] != gaze_data[:-1, ET], axis=1)
	repeatET = np.append(repeatET, True)

	nET = np.where(np.logical_and(validity, repeatET))[0]
	nHMD = np.where(np.logical_and(
						validity,
						np.append(
							np.all(gaze_data[1:, HMD] != gaze_data[:-1, HMD], axis=1),
							True)
						)
					)[0]-1

	gaze_data[:, 0] = (gaze_data[:, 0] - gaze_data[0, 0])

	# New data at N (=resample) samples per millisecond
	new_data = np.zeros([int(gaze_data[-1, 0]/1e3 * samplingRate), gaze_data.shape[1]])
	# Interpolation space
	interpR = np.linspace(0, gaze_data[-1, 0], new_data.shape[0], endpoint=True)
	new_data[:, 0] = interpR

	# Interpolate gaze unit direction
	new_data[:, ET] = interpolate.griddata(gaze_data[nET, 0], gaze_data[nET, ET], (interpR), method='cubic', fill_value="extrapolate")
	# Interpolate camera rotation (quaternion)
	new_data[:, HMD] = quat.as_float_array(quat.squad(quat.as_quat_array(gaze_data[nHMD, HMD]), gaze_data[nHMD, 0], (interpR)))

	new_data[:, ET] /= np.linalg.norm(new_data[:, ET], axis=1)[:, None]

	return new_data

def getDataOnSphere(data, Euler2Quat=False, callback=lambda *a: None):

	HMD_rot = quat.as_quat_array(data[:, 1:5])

	head = quat.rotate_vectors(HMD_rot, [0, 0, 1])
	head = quat.rotate_vectors(np.quaternion(1, 1, 0, 0), head)

	# This solution is faster for large datafiles, but uses a lot of memory
	# gaze = quat.rotate_vectors(HMD_rot, data[:, 5:8])[np.eye(data.shape[0], dtype=bool)]
	gaze = np.empty([data.shape[0], 3])
	for i in range(data.shape[0]):
		gaze[i] = quat.rotate_vectors(HMD_rot[i], data[i, 5:8])
		callback((i+1)/data.shape[0], "Calculating eye-in-space data")
	gaze = quat.rotate_vectors(np.quaternion(1, 1, 0, 0), gaze)

	if Euler2Quat:
		# Reverse because of Unity Euler angle system
		gaze[:, 2] = -gaze[:, 2]

	return head, gaze

def prepareGaze(data, head, gaze):
	"""
	Outputs array like:
		0,1,2: gaze direction vector
		3,4,5: eye direction vector
		6,7,8: head direction vector
		9: timestamp
		10: sample idx
	"""

	gaze_in = np.zeros( (gaze.shape[0], 11))

	# combined gaze vector
	gaze_in[:, :3] = gaze
	# eye direction vector
	gaze_in[:, 3:6] = data[:, [5,7,6]]
	# head direction vector
	gaze_in[:, 6:9] = head
	# timestamp
	gaze_in[:, 9] = data[:, 0]
	# sample idx
	gaze_in[:, 10] = np.arange(gaze_in.shape[0])

	# Renormalise vectors to be unit vectors
	# Can deviate from unit vectors due to floating-point precision
	gaze_in[:, :3] /= np.linalg.norm(gaze_in[:, :3] , axis=1)[:, None]
	gaze_in[:, 3:6] /= np.linalg.norm(gaze_in[:, :3] , axis=1)[:, None]
	gaze_in[:, 6:9] /= np.linalg.norm(gaze_in[:, :3] , axis=1)[:, None]
	# Without this correction outputs of
	#	  ..utils.distance.dist_orthodromic
	# and ..utils.distance.dist_angle_arrays_unsigned
	# will slightly diverge

	return gaze_in

def preprocess(data, data_range, resample=None, Euler2Quat=False, eye=None,
	callback=lambda *a: None):
	"""
	Input: raw head and eye tracking data
	Output: list of fixation and saccade features

	Expected feature order in data matrix:
		0: timestamp
		1,2,3,4: camera rotation as quaternion (or Euler) (X,Y,Z,W)
		5,6,7: gaze direction vector relative to camera rotation (X,Y,Z)
	"""

	data = data[slice(*data_range)]

	if eye == "B":
		data[:, 5:8] = (data[:, 5:8] + data[:, 8:11])/2
		# Reproject on sphere
		data[:, 5:8] /= np.linalg.norm(data[:, 5:8], axis=1)[:, None]

	data[:, 0] = data[:, 0] - data[0, 0]
	# We expect timestamps in milliseconds, we transform them if this is not the case
	#	Get log10 of average timestamp differences between samples (good approximation for one over sampling rate)
	logframerate = np.log10(np.nanmean(data[1:, 0] - data[:-1, 0]))
	#	Round to the nearest multiple of 3 (because we transform from nanosec (SMI), microsec (Tobii) to millisec)
	rem = int(3 * (logframerate//3))
	if rem != 0:
		data[:, 0] /= 10**rem
		printWarning("Timestamps were divided by 1e{} to be in milliseconds".format(rem), header="preprocess", verbose=0)

	callback(0, "Processing valid data samples.")

	if eye != "H":
		# Exclude NaNs found in eye or head data
		#	SMI reports NaN were gaze is lost
		validity = ~np.any( np.isnan(data[:, 1:]), axis=1 )
		#	Tobii reports all 0 or -1 when data is missing
		validity &= ~( data[:, 5:8].sum(axis=1) == -3 )
		validity &= ~( data[:, 5:8].sum(axis=1) == 0 )
	else:
		validity = np.ones([data.shape[0]], dtype=bool)

	if type(resample) in [int, float] and resample > 0:
		callback(.1, "Interpolating data to {} Hz.".format(resample))
		data = interpolate_raw(data, validity,
			samplingRate=resample)
	else:
		data = data[validity]
		
	callback(0, "Calculating eye-in-space data")
	head, gaze = getDataOnSphere(data, Euler2Quat=Euler2Quat, callback=callback)

	callback(0, "Preparing final gaze matrix")
	gaze = prepareGaze(data, head, gaze)

	return gaze

def downsample2Centroid(gp, tempWindowSize=90):
	"""
	Return a segmentation of data samples as a trajectory of points based on an uniform resampling of camera rotations.
	parameter `tempWindowSize` determines the size of the windows used to downsample signal.
	"""

	sampleN = int(gp[-1, 9]//tempWindowSize)

	label_list = np.zeros([gp.shape[0]])

	for iSS in range(sampleN):

		iStart = iSS * tempWindowSize
		iEnd = (iSS+1) * tempWindowSize

		data_range = np.logical_and(gp[:, 9] >= iStart, gp[:, 9] < iEnd)
		if data_range.sum() == 0:
			continue
		iSs, iEs = np.array(np.where(data_range))[0][[0, -1]]

		label_list[iSs:iEs] = 0
		label_list[iEs] = 1

	return label_list
