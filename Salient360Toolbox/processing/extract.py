#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

from ..utils.misc import *
from ..utils.distances import *
from ..utils.conversion import *

def getGazeFeatures(gaze_point, fixationMarkers, velocity=None):
	# Aggregate fixation samples into fixation points and compute fixation/saccade features

	if velocity is None: velocity = np.zeros([gaze_point.shape[0]])

	acceleration = (velocity[1:]-velocity[:-1])/(gaze_point[1:, 9] - gaze_point[:-1, 9])

	# Get fixation start and end sample indices
	fixationMarkers = fixationMarkers != 0
	# Case where no saccades were identified
	if fixationMarkers.sum() < 5 or fixationMarkers.sum() > (fixationMarkers.shape[0]-2):
		printWarning("Zero saccades were identified. Your parsing algorithm's parameters may be wrong.",
			header="[getGazeFeatures]")
		return np.empty([0, 29])

	trans = np.where(fixationMarkers[:-1] != fixationMarkers[1:])[0]
	starts = trans.copy()
	# Starts with a fixation
	if fixationMarkers[0] == True: starts = np.append([-1], starts)
	# Ends on a fixation
	if fixationMarkers[-1] == True: starts = np.append(starts, fixationMarkers.shape[0]-1)

	fixationPts = np.empty([starts.shape[0]//2, 29])
	fixationPts[:] = np.nan

	# Compute fixational features

	eyeAvgPos = np.empty([fixationPts.shape[0], 3])
	camAvgPos = np.empty([fixationPts.shape[0], 3])

	for i in range(0, len(starts), 2):

		startMarker = starts[i]+1
		endMarker = min(starts[i+1]+1, fixationMarkers.shape[0]-1)

		fixationPt = fixationPts[i//2, :] # view (not a copy)

		# Fixation position (unit vector)
		fixationPt[2:5] = gaze_point[startMarker: endMarker, :3].mean(axis=0)
		fixationPt[2:5] /= vec_magnitude(fixationPt[2:5])

		# Gaze position on sphere (long, lat)
		fixationPt[:2] = UnitVector2EquirectArrCheck(fixationPt[2:5])
		# longitude
		fixationPt[0] = fixationPt[0] / (2*np.pi) - .25
		# latitude
		fixationPt[1] = 1 - (fixationPt[1] / np.pi + .5)

		# print(gaze_point[startMarker: endMarker, :3])
		# print(fixationPt[:2])

		# Mean eye pos on sphere(unit vector)
		eyeAvgPos[i//2] = gaze_point[startMarker: endMarker, 3:6].mean(axis=0)
		eyeAvgPos[i//2] /= vec_magnitude(eyeAvgPos[i//2])

		# Eye position on sphere (long, lat)
		fixationPt[5:7] = UnitVector2EquirectArrCheck(eyeAvgPos[i//2])
		# longitude
		fixationPt[5] = fixationPt[5] / (2*np.pi) - .25
		# latitude
		fixationPt[6] = 1 - (fixationPt[6] / np.pi + .5)

		# print(gaze_point[startMarker: endMarker, 3:6])
		# print(fixationPt[5:7])
		# print("="*20)

		# Mean cam pos on sphere(unit vector)
		camAvgPos[i//2] = gaze_point[startMarker: endMarker, 6:9].mean(axis=0)
		camAvgPos[i//2] /= vec_magnitude(camAvgPos[i//2])

		# Camera position on sphere (long, lat)
		fixationPt[7:9] = UnitVector2EquirectArrCheck(camAvgPos[i//2])
		# longitude
		fixationPt[7] = fixationPt[7] / (2*np.pi) - .25
		# latitude
		fixationPt[8] = 1 - (fixationPt[8] / np.pi + .5)

		# Fixation index
		fixationPt[9] = i//2

		# Fixation sample idx start
		fixationPt[10] = startMarker
		# Fixation sample idx end
		fixationPt[11] = endMarker-1

		# Start timestamp
		fixationPt[12] = gaze_point[startMarker, 9]
		# End timestamp
		fixationPt[13] = gaze_point[endMarker, 9]

		# Fixation duration
		fixationPt[14] = gaze_point[endMarker, 9] - gaze_point[startMarker, 9]

		# Mean fixation dispersion (rad)
		fixationPt[15] = mean_dist_angle_topoint(gaze_point[startMarker: endMarker, :3], fixationPt[2:5])

		# Peak fixation velocity (rad/sec)
		fixationPt[16] = np.max(velocity[startMarker: endMarker])
		# Peak fixation acceleration (rad/sec)
		fixationPt[17] = np.max(acceleration[startMarker: endMarker])

	# Expects output data to be
	#	longitudes: [0, 2*np.pi]
	#	latitudes: [0, np.pi]
	#		Origin: top-left
	fixationPts[:, 0][fixationPts[:, 0] < 0] += 1
	fixationPts[:, 1][fixationPts[:, 1] < 0] += 1
	fixationPts[:, 5][fixationPts[:, 5] < 0] += 1
	fixationPts[:, 6][fixationPts[:, 6] < 0] += 1
	fixationPts[:, 7][fixationPts[:, 7] < 0] += 1
	fixationPts[:, 8][fixationPts[:, 8] < 0] += 1

	# Compute saccadic features
	def getSaccFeatures(vecs1, vecs2, equirect1, equirect2):
		# Return  saccade amplitude, horizontal and relative angles
		feat = np.empty([vecs1.shape[0], 3])

		unnorm = np.array([[2*np.pi, np.pi]])

		merc1 = Equirect2Mercator(equirect1 * unnorm)
		merc2 = Equirect2Mercator(equirect2 * unnorm)

		saccs = merc2 - merc1

		rec = np.array( ((2*np.pi, 0)) )

		# looping saccades - assumption: the shortest saccade vector is the right one
		# 	looping R
		loopR = np.where(saccs[:-1,0] > np.pi/2)[0]
		saccs[loopR,:] = merc2[loopR+1,:] - (rec + merc1[loopR,:])
		# 	looping L
		loopL = np.where(saccs[:-1,0] < -np.pi/2)[0]
		saccs[loopL,:] = (rec + merc2[loopL+1, :]) - merc1[loopL, :]

		# Saccade amplitude
		feat[:, 0] = dist_angle_arrays_unsigned(vecs1, vecs2)
		# Saccade horizontal angle
		feat[:, 1] = dist_angle_topoint_signed(saccs, [1, 0])
		# Saccade relative angle
		# 	Use the fact that a relative angle is equal to
		#		the difference of two absolute angles
		# feat[1:, 2] = feat[1:, 1] - feat[:-1, 1]
		# 	Calculate relative angle directly
		feat[1:, 2] = dist_angle_topoint_signed(saccs[1:], saccs[:-1])
		feat[0, 2] = np.nan

		return feat

	if fixationPts.shape[0] > 1:
		# print("_-"*25)
		# print(fixationPts[0, 2:5],fixationPts[1, 2:5],
		# 	  fixationPts[0, :2],fixationPts[1, :2])
		# print(eyeAvgPos[0, :], eyeAvgPos[1, :],
		# 	  fixationPts[0, 5:7],fixationPts[1, 5:7])
		# print('int')
		# print(getSaccFeatures(fixationPts[:-1, 2:5],fixationPts[1:, 2:5], fixationPts[:-1, :2],fixationPts[1:, :2])[2])
		# print(getSaccFeatures(eyeAvgPos[:-1, :], eyeAvgPos[1:, :],
		# 												fixationPts[:-1, 5:7],fixationPts[1:, 5:7])[2])
		# exit()
		# saccade features: Gaze (rad.)
		fixationPts[1:, [20, 23, 26]] = getSaccFeatures(fixationPts[:-1, 2:5],fixationPts[1:, 2:5],
														fixationPts[:-1, :2],fixationPts[1:, :2])
		# saccade features: Eye (rad.)
		fixationPts[1:, [21, 24, 27]] = getSaccFeatures(eyeAvgPos[:-1, :], eyeAvgPos[1:, :],
														fixationPts[:-1, 5:7],fixationPts[1:, 5:7])
		# saccade features: Head (rad.)
		fixationPts[1:, [22, 25, 28]] = getSaccFeatures(camAvgPos[:-1, :], camAvgPos[1:, :],
														fixationPts[:-1, 7:9],fixationPts[1:, 7:9])

		# Peak vel & acc are based on samples rather than data computed in the previous loop
		for ifix in range(1, fixationPts.shape[0]):
			# Peak sacc vel
			fixationPts[ifix, 18] = np.max(velocity[int(fixationPts[ifix-1, 11]): int(fixationPts[ifix, 10])])
			# Peak sacc accel
			fixationPts[ifix, 19] = np.max(acceleration[int(fixationPts[ifix-1, 11]): int(fixationPts[ifix, 10])])

	return fixationPts
