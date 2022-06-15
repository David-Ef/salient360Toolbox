#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: plotting function made to check or demonstrate data
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Colormap (def.: coolwarm)
def colormap(mapname="coolwarm"):
	from .divergingColorMaps import getColormapByName

	cmap = getColormapByName(mapname)

	# random picture
	img = np.random.randint(0, 255, size=[768, 1024], dtype=int)
	# convert to 3-channel heatmap
	img = cmap[img] # Dim: [768, 1024, 3]

	plt.subplot(121)
	plt.imshow(img)
	plt.title("Random sampling")

	# Top-bottom gradient
	img = np.repeat(np.arange(255)[:, None], 25, axis=1)
	# convert to 3-channel heatmap
	img = cmap[img] # Dim: [255, 25, 3]

	plt.subplot(122)
	plt.imshow(img)
	plt.title("Linear gradient")

	plt.suptitle("Showing color map \"{}\"".format(mapname))

	plt.show()

# Equirectangular saliency map comparison
def equirectangularWeighting():
	"""
	Compare latitudinal sin weighting with a quasi-uniform sphere sampling.
	Sampling method approximates a sin function (0, pi).
	The Sine function is a simpler weighting method.
	"""
	from ..comparison.commons import quasiUniformSphereSampling

	sinWeight = np.sin(np.linspace(0, np.pi, 180))

	for i, x in enumerate([1e3, 1e4, 1e5, 1256637]):
		sSampl = quasiUniformSphereSampling(x)
		sSampl = np.rad2deg(sSampl[:, 0]).astype(int)
		# sinWeight = np.rad2deg(sinWeight).astype(int)

		# Count number of points per longitude
		count = np.zeros([180])
		for lat in range(count.shape[0]):
			count[lat] = (sSampl==lat).sum()

		plt.subplot(2, 2, i+1)
		plt.title("{:.2e} sampling points".format(sSampl.shape[0]))
		plt.plot(np.arange(count.shape[0]), count/count.sum(), "r",
			label="quasi-uniform sphere sampling")
		plt.plot(np.arange(sinWeight.shape[0]), sinWeight/sinWeight.sum(), "g",
			label="sin weighting")
		plt.legend()
	plt.show()

def displayGazeSphere(gp, colorMat=None):
	from mpl_toolkits.mplot3d import Axes3D

	if colorMat is None:
		colorMat = np.arange(gp.shape[0])
	fig = plt.figure(figsize=(10, 10), dpi=100)

	# plot gaze data on sphere
	x = gp[:, 0]
	y = gp[:, 1]
	z = gp[:, 2]
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x, y, z, zdir="z", depthshade=True, c=colorMat)

	# Plot a 3D sphere (vertices)
	lat, lon = np.mgrid[0:np.pi:10j, 0:2*np.pi:20j]
	# (lat,long) to (x, y, z)
	x = np.sin(lat)*np.cos(lon)
	y = np.sin(lat)*np.sin(lon)
	z = np.cos(lat)

	# Vertices colour is a function of gaze points' colours
	from matplotlib import cm
	color = 1 - cm.viridis(np.unique(colorMat)).mean(axis=0) - .2
	color[color<0] = 0
	color[color>1] = 1

	colors = np.ones(list(z.shape[:2])+[4])
	colors[:, :, :3] = color[None, None, :3]

	rcount, ccount, _ = colors.shape
	surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
	surf.set_facecolor((0, 0, 0, 0))

	ax.set_xlabel('\nX')
	ax.set_ylabel('\nY')
	ax.set_zlabel('\nZ')
	
	plt.show()

# Needed by displayAllData
def _getFixBoundaries(markers):
	start = []
	end = []
	iMarker = 0
	# Find _end_ and _start_ of fixation
	while iMarker < len(markers)-1:
		if len(end) == 0:
			start.append(iMarker)
			iMarker+=1
			while iMarker < len(markers) and markers[iMarker] == 0:
				iMarker+=1
			end.append(iMarker)
		# if iMarker < len(markers)-1 and markers[iMarker] > 0 and markers[iMarker+1] == 0:
		elif markers[iMarker] > 0 and markers[iMarker+1] == 0:
			start.append(iMarker)
			iMarker+=1
			while iMarker < len(markers) and markers[iMarker] == 0:
				iMarker+=1
			end.append(iMarker)
		else:
			iMarker+=1
	return start, end

# Gaze and head data display
def displayAllData(gp,
	labels=None,
	vel_thresh=100):
	"""
	Display angle velocity of camera, eye and camera+eye signals. Useful to check data and show compensatory behaviour of head and eyes.
	Note that camera/head rotation is actually full-body rotation (subjects were sitting in a rolling chair).
	"""
	from .distances import dist_angle_vectors_unsigned

	from scipy.ndimage.filters import gaussian_filter1d
	from scipy.signal import savgol_filter

	gp = gp.copy()

	diffT = gp[1:, 9] - gp[:-1, 9]
	timeS = gp[:-1, 9]

	distanceHE = np.zeros([gp.shape[0]-1])
	distanceH = np.zeros([gp.shape[0]-1])
	distanceE = np.zeros([gp.shape[0]-1])
	for i in range(gp.shape[0]-1):
		vec1 = gp[i, :3]; vec2 = gp[i+1, :3]
		distanceHE[i] = dist_angle_vectors_unsigned(vec1, vec2)

		vec1 = gp[i, 6:9]; vec2 = gp[i+1, 6:9]
		distanceH[i] = dist_angle_vectors_unsigned(vec1, vec2)

		vec1 = gp[i, 3:6]; vec2 = gp[i+1, 3:6]
		distanceE[i] = dist_angle_vectors_unsigned(vec1, vec2)

	# Remove loop above

	velHE = np.rad2deg(distanceHE/diffT)*1e3
	velH = np.rad2deg(distanceH/diffT)*1e3
	velE = np.rad2deg(distanceE/diffT)*1e3

	miss_HE = np.logical_not(np.isnan(velHE) | np.isinf(velHE))
	miss_H = np.logical_not(np.isnan(velH) | np.isinf(velH))
	miss_E = np.logical_not(np.isnan(velE) | np.isinf(velE))

	velHE = velHE[miss_HE]
	velH = velH[miss_H]
	velE = velE[miss_E]

	# velHE = savgol_filter(velHE, 9, 2)
	# velH = savgol_filter(velH, 9, 2)
	# velE = savgol_filter(velE, 9, 2)

	colorMat = "b"
	if labels is None:
		labels = np.ones([velE.shape[0]], dtype=bool)

	labels = labels[:timeS.shape[0]]
	colorMat = np.zeros(labels.shape, dtype="|S7")
	colorMat[labels==0] = 'g' # Fixations
	colorMat[labels==1] = 'b' # Saccades
	colorMat[labels==2] = 'r' # Blinks
	colorMat = np.array([i.decode() for i in colorMat])

	start, end = _getFixBoundaries(labels)

	fig = plt.figure(figsize=(10, 5), dpi=100)

	ax1 = fig.add_subplot(311)
	ax1.scatter(timeS[miss_E], velE, marker='o', s=5, color=colorMat[miss_E])
	ax1.set_title("Eye only velocity")
	plt.setp(ax1.get_xticklabels(), visible=False)

	end_= 0
	for i in range(len(start)):
		ax1.axvspan(gp[end_, 9], gp[start[i], 9], facecolor='b', alpha=0.2) # Saccade
		ax1.axvspan(gp[start[i], 9], gp[end[i], 9], facecolor='g', alpha=0.2) # Fixation
		end_ = end[i]

	ax2 = fig.add_subplot(312, sharex=ax1)
	ax2.scatter(timeS[miss_H], velH, marker='o', s=5, color="b")
	ax2.set_title("Head only velocity")
	plt.setp(ax2.get_xticklabels(), visible=False)

	end_= 0
	for i in range(len(start)):
		ax2.axvspan(gp[end_, 9], gp[start[i], 9], facecolor='b', alpha=0.2) # Saccade
		ax2.axvspan(gp[start[i], 9], gp[end[i], 9], facecolor='g', alpha=0.2) # Fixation
		end_ = end[i]

	ax3 = fig.add_subplot(313, sharex=ax1)
	ax3.scatter(timeS[miss_HE], velHE, marker='o', s=5, c=colorMat[miss_HE])
	ax3.set_title("Head + Eye velocity")

	end_= 0
	for i in range(len(start)):
		ax3.axvspan(gp[end_, 9], gp[start[i], 9], facecolor='b', alpha=0.2) # Saccade
		ax3.axvspan(gp[start[i], 9], gp[end[i], 9], facecolor='g', alpha=0.2) # Fixation
		end_ = end[i]

	ax1.set_xlim([-100, int(gp[-1, 9] + 100)])

	ax1.axhline(y=vel_thresh,xmin=0,xmax=20000,c="r",linewidth=2,zorder=0)
	# ax2.axhline(y=25,xmin=0,xmax=20000,c="g",linewidth=2,zorder=0)
	ax3.axhline(y=vel_thresh,xmin=0,xmax=20000,c="r",linewidth=2,zorder=0)

	plt.xlabel('Time (sec)')
	ax2.set_ylabel('Velocity (deg/sec)')

	plt.tight_layout()
	plt.show()
