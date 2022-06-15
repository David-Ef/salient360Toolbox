#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: Output polar distribution (saccade angle and amplitude)
# Note: optimized version (no loop, full numpy)
# ---------------------------------

import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.stats as stats
import scipy.io as scio
import numpy as np

from scipy.interpolate import griddata

plt.rcParams["figure.figsize"] = [10.8, 10.8]

global RES, MAX_AMPL

RES = [1/4, 1/2]
MAX_AMPL = 50

def createPDF(saccades,  bandwithBias=2):

	if saccades.shape[1] > 2:
		# Calculate amplitude if necessary
		saccades[:, 0] = np.abs(saccades[:, 0] - saccades[:, 1])
		# saccades[:, 2] -= saccades[:, 3]
		Dangle = np.min(
						np.concatenate(
							[(saccades[:, 2] - saccades[:, 3])[:, None],
							 (saccades[:, 2] - (2*np.pi - saccades[:, 3]))[:, None]],
							axis=1),
						axis=1)

		saccades[:, 2] = Dangle

		saccades = saccades[:, [0, 2]]

	saccades = saccades[np.logical_not(np.any(np.isnan(saccades), axis=1))]
	saccades[:, 1] = np.rad2deg(saccades[:, 1])

	print("  {} fixations".format(saccades.shape[0]))
	if saccades.shape[0] == 0:
		print("[createPDF] out: no saccade data")
		return None

	rho =  np.arange(0, MAX_AMPL+RES[0], RES[0])
	theta = np.arange(0, 361+RES[1], RES[1])

	proba = np.zeros([rho.shape[0], theta.shape[0]])

	loss = saccades[:, 0] > MAX_AMPL
	saccades = saccades[np.logical_not(loss), :]
	print("[createPDF] Lost {} samples".format(loss.sum()))

	# These two values are the "bandwidths" for the amplitude (r:rho) and angle (t:theta).
	#	Below is a rule of thumb method taken from numpy
	hr = saccades[:, 0].std() * np.power(4/3/saccades.shape[0], 1/5) /  bandwithBias
	ht = saccades[:, 1].std() * np.power(4/3/saccades.shape[0], 1/5) /  bandwithBias

	print("\r", " "*40, "\rGenerating rho and theta grids", end="")
	r_ = np.repeat(np.arange(rho.shape[0])[None, :], saccades.shape[0], axis=0)
	t_ = np.repeat(np.arange(theta.shape[0])[None, :], saccades.shape[0], axis=0)

	print("\r", " "*40, "\rComputing rho and theta differences to samples", end="")
	diff_theta = np.abs(t_*RES[1] - saccades[:, 1][:, None])
	diff_theta[diff_theta>=180] = 360-diff_theta[diff_theta>=180] # Data are circular
	diff_rho = r_*RES[0] - saccades[:, 0][:, None]

	print("\r", " "*40, "\rDividing rho and theta by their bandwidth", end="")
	diff_theta = diff_theta / ht
	diff_rho = diff_rho / hr

	print("\r", " "*40, "\rComputing proba of all rho points", end="")
	norm_pdf_rho = stats.norm.pdf(diff_rho)
	print("\r", " "*40, "\rComputing proba of all theta points", end="")
	norm_pdf_theta = stats.norm.pdf(diff_theta)

	print("\r", " "*40, "\rComputing dot product of rho and theta", end="")
	proba = np.dot(norm_pdf_rho.T, norm_pdf_theta) 
	# proba /= saccades.shape[0]*hr*ht

	return proba

def createPolarPlot(proba, filename=None, maskRadius=0, ext=".png"):
	rho =  np.arange(0, MAX_AMPL+RES[0], RES[0])
	theta = np.arange(0, 361+RES[1], RES[1])

	print("\r", " "*40, "\rBuilding and saving polar plot", end="")

	# Upsample data 
	XX, YY = np.meshgrid(rho, theta)
	points = np.concatenate( [XX.flatten()[:, None], YY.flatten()[:, None]], axis=1)
	values = proba.flatten("F")

	RES_ = [1/16, 1/4]
	rhoP =  np.arange(0, MAX_AMPL+RES_[0], RES_[0])
	thetaP = np.arange(0, 360+RES_[1], RES_[1])
	grid_r, grid_theta = np.meshgrid(rhoP, thetaP)
	proba = griddata(points, values, (grid_r, grid_theta), method='cubic',fill_value=0).T

	ax = plt.subplot(polar=True)

	theta_, rad_ = np.meshgrid(np.deg2rad(thetaP), rhoP)
	ax.pcolormesh(theta_, rad_, proba, cmap="viridis")

	# ax.set_theta_zero_location('E')  # Set zero to North 
	# ax.set_xticklabels(['' * 8], color='#000000', fontsize=18)  # Customize the xtick labels
	# ax.spines['polar'].set_visible(False)  # Show or hide the plot spine
	# ax.set_face_color('#FFFFFF')  # Color the background of the plot area
	
	# ax.yaxis.set_ticks(np.arange(5, MAX_AMPL, 5))
	ax.set_rgrids(np.arange(5, MAX_AMPL, 10), fmt="-%i-", angle=90, color='#E0E0E0', horizontalalignment='center', verticalalignment='center', fontsize=23) 
	ax.set_thetagrids(angles=np.arange(0, 360, 45), labels=np.arange(0, 360, 45), color='#000000', horizontalalignment='center', verticalalignment='center', fontsize=23) 

	if maskRadius > 0:
		fig = plt.gcf()
		mask_radius = plt.Circle((0, 0), maskRadius, transform=ax.transData._b, fill=False, edgecolor='#F81111', linewidth=3)#, fill=True, alpha=.5)
		fig.gca().add_artist(mask_radius)

	if filename is not None:
		plt.savefig(filename+ext, transparent=True)
	else:
		plt.show()

	plt.clf()

def saveMat(mat, path, type_="npz"):
	if type_ == "npz":
		np.savez(path, mat)
	elif type_ == "np":
		np.save(path, mat)
	elif type_ == "mat":
		scio.savemat(path, {"proba": mat})
	else:
		print("Wrong file type: {}, no file created".format(type))

if __name__ == "__main__":

	data = np.loadtxt("data_example.csv", delimiter=",", skiprows=1)
	# Samples in sxample data are, by line:
	#	At 5° of amplitude at 0° angle (East)
	#	At 10° of amplitude at 45° angle
	#	At 20° of amplitude at 90° angle (North)
	#	At 30° of amplitude at 180° angle (West)
	#	At 40° of amplitude at 270° angle (South)

	proba = createPDF(data,  bandwithBias=10)

	if proba is not None:
		createPolarPlot(proba, filename="example", maskRadius=10)
