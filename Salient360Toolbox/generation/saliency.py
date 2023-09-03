#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# Note: Some functions have option to output graphical representations of the gaze analyses process. Set "if False:" to "if True:" to enalbe them. Script will exit after graphical output.
# ---------------------------------

import numpy as np
import sys

from ..utils.misc import *
import numba

@numba.njit(parallel=True)
def getGaussian_(map_, fix, gauss_sigma):
	# # Orthodromic distance - not compatible with numba
	# a = np.einsum("k,ijk->ij", fix, map_)
	# c = np.arccos(a)

	# Simple distance in 3D space - faster approximation of the orthodromic distance output
	c = np.sqrt(np.power(map_ - fix, 2).sum(-1))

	return np.exp(
		 -(
			(
				c**2
			) /
			(2 * gauss_sigma**2)
		  )
		)

@numba.njit(parallel=True)
def getGaussianSupport(dim, pos, gauss_sigma):
	# Gaze position on equirectangular map

	rect = [int(pos[0] * dim[1]), int(pos[1] * dim[0])]

	# Dimension of the Gaussian support window
	# The window is a square at the equator (Sy==Sx)
	#	As we go further from the equator latitudinally Sx increases faster
	#	to account for the latitudinal deformation of the equirectangular projection
	Sy = int(dim[0] * np.sin(gauss_sigma*2.5))
	Sx = min(dim[1],
			int(dim[1] *
					(
						(1+np.tan( np.abs( pos[1] * np.pi - np.pi/2 ) ) ) *
						(gauss_sigma*1.5)
					)
				)
			)

	# Get corners of the Gaussian support window
	X = [rect[0]-Sx//2, rect[0]+Sx//2]
	Y = [rect[1]-Sy//2, rect[1]+Sy//2]
	# Interpolate linearly between points
	X = np.arange(X[0], X[1])
	Y = np.arange(Y[0], Y[1])
	# Wrap around points that would hit outside the map
	Y[Y>dim[0]-1] -= dim[0]
	X[X>dim[1]-1] -= dim[1]

	return Y, X

# @numba.jit
def saliencyOp_(saliencymap, fix, pvi, gauss_sigma):
	"""
	Takes as input a matrix, a fixation position and a Gaussian sigma
	Draw Gaussian at fixation location in the matrix
	Optimized with Gaussian support as a function of latitude
	"""

	SalMapRes = saliencymap.shape[-2:]

	posC = fix[3:5] # long, lat
	Y, X = getGaussianSupport(SalMapRes, posC, gauss_sigma)

	# Create a meshgrid to reference all points within the window
	Y, X = np.meshgrid(Y, X)

	# Uncomment to draw Gaussian support rectangle
	# saliencymap[Y, X] += 1

	# The ellipsis has no effect on a 2D array, but will hit all "frames" when creating a "saliency video" (3D array where first dim is a set of saliency maps)
	saliencymap[..., Y, X] += getGaussian_(pvi[Y, X, :], fix[:3], gauss_sigma)

def getSphereGridPoints3D_(height, width):
	"""
	Return an array containing 3D unit vectors at the location of each pixel in the saliency map to output
	"""

	pvi = np.empty( (height, width, 3) )

	yy, xx = np.meshgrid(np.arange(height), np.arange(width))

	Ex = (np.pi*2) - xx.T / (width-1) * (np.pi*2)
	Ey = (yy.T / (height-1)) * np.pi

	pvi[:, :, 0] = np.sin(Ey) * np.cos(Ex)
	pvi[:, :, 1] = np.sin(Ey) * np.sin(Ex)
	pvi[:, :, 2] = np.cos(Ey)

	return pvi

def getSaliency(saliencymap, fix_list, gauss_sigma=2, callback=None, **kwargs):
	# Pass fix_list of a frame or whole stimulus
	# toImage and toFrames call this function
	# Returns a matrix (W, H) or (W, H, nFrames)
	#	Send returned data to
	#		toImage, toFrames, toBin, toBinFrames

	printNeutral("Computing saliency data", verbose=1)

	SalMapRes = saliencymap.shape

	pvi = getSphereGridPoints3D_(SalMapRes[0], SalMapRes[1])

	gauss_sigma = np.deg2rad(gauss_sigma)

	progressStep = max(1, fix_list.shape[0]//25) # 50 steps max to show progression
	for iFix in range(fix_list.shape[0]):

		# Non-optimized method
		# saliencymap += np.exp(-( ((pvi[:, :, :] - fix_list[iFix][:3])**2) / (2*gaussSigma**2)).sum(axis=2)); continue

		saliencyOp_(saliencymap, fix_list[iFix, :5], pvi, gauss_sigma)

		if callback is not None and iFix % progressStep == 0:
			continue_ = callback((iFix+1)/fix_list.shape[0])
			if not continue_: clearline(); return None

		printNorm("{:>6.2%}%".format((iFix+1)/fix_list.shape[0]), clear=True, end="", verbose=0)
	clearline()

def getSaliencyDyn(saliencymap, fix_list, gauss_sigma=2, time_cut=None, callback=None):

	length = saliencymap.shape[0]
	SalMapRes = saliencymap.shape[1:]

	pvi = getSphereGridPoints3D_(SalMapRes[0], SalMapRes[1])

	gauss_sigma = np.deg2rad(gauss_sigma)

	progressStep = max(1, fix_list.shape[0]//50) # 50 steps max to show progression
	for iFix in range(fix_list.shape[0]):

		# Start cut
		Scut = time_cut[int(fix_list[iFix, 5])]-1
		# End cut
		Ecut = min(time_cut[int(fix_list[iFix, 6])], length)

		saliencyOp_(saliencymap[Scut:Ecut, :, :], fix_list[iFix, :5], pvi, gauss_sigma)

		if callback is not None and iFix % progressStep == 0:
			continue_ = callback((iFix+1)/fix_list.shape[0])
			if not continue_: clearline(); return None

		printNorm("iFix:", iFix+1, end="", clear=True, verbose=0)
	clearline()

def toImage(sal_map, cmap=None, reverse=False):
	"""
	DOC
	"""
	from ..utils import divergingColorMaps
	if type(cmap) is int:
		colormap = divergingColorMaps.getColormapByIndex(cmap)
	elif type(cmap) is str:
		colormap = divergingColorMaps.getColormapByName(cmap)
	else:
		colormap = divergingColorMaps.cmap

	colormap = colormap[::-1] if reverse else colormap

	sal_image = np.empty([*sal_map.shape, 3], dtype=np.uint8)
	if sal_map.sum() == 0:
		sal_image[:] = colormap[0]
	else:
		# Select in colours according to values
		sal_image[:] = colormap[ (sal_map / sal_map.max() * 255).astype(np.uint8) ]

	return sal_image

def saveImage(mat, path_file, extension="png", blend=None):
	"""
	DOC
	"""
	if blend is not None:
		blendImage(mat, path_file, blend,
			extension=extension)
	else:
		import cv2
		cv2.imwrite(path_file+"."+extension, mat[:, :])

def saveBin(mat, path_file, type_="salmap", force=False, callback=None):
	"""
	DOC
	"""
	final_path = path_file+"_{}_{}b_{}.bin".format(
					"x".join([str(dim) for dim in mat.shape[::-1]]),
					mat.dtype.alignment*8, type_)

	if force or not os.path.exists(final_path):
		mat.tofile(final_path)

	return final_path

def saveImages(mat, path_folder, extension="png", blend=None, force=False, ignore_prompt=False):
	assertC(len(mat.shape) == 3, "function \"saveImages\" expects a 3D tensor (n_frames, px_height, px_width). Got {}.".format(mat.shape), printIfFail=True)

	if blend is not None:
		blendImages(mat, path_folder, blend, force=force, ignore_prompt=ignore_prompt)
		return

	os.makedirs(path_folder, exist_ok=True)
	if path_folder[-1] == os.sep: path_folder = path_folder[:-1]
	stim_name = path_folder.split(os.sep)[-1]

	continue_ = False
	N_files = len(os.listdir(path_folder))
	if not force and\
		os.path.exists(path_folder) and\
		N_files > 0:
		ask = " " if not ignore_prompt else "C"
		while ask[0] not in ["C", "F", "S"]:
			printWarning("Output folder [{}] already contains files [{}]".format(path_folder, N_files), verbose=0)
			printWarning("What do you want to do?", tab=1, header="", verbose=0)
			printWarning("\"C\" [Continue] to generate missing images,", bold=False, tab=2, header="", verbose=0)
			printWarning("\"F\" [Force] to overwrite existing files,", bold=False, tab=2, header="", verbose=0)
			printWarning("\"S\" [Stop] to exit this function", bold=False, tab=2, header="", verbose=0)
			ask = input("\t\t_ ")
			if len(ask) == 0: ask = " "

			clearlines(6)

			ask = ask[0].upper()

		if ask == "C":
			continue_ = True
		elif ask == "F":
			continue_ = False
		else:
			return False

	for i_frame in range(mat.shape[0]):
		printNeutral("frame {}/{}".format(i_frame, mat.shape[0]), sep="", end="", clear=True, verbose=0)
		
		file_path = "{}{}{{:0{}}}".format(path_folder, os.sep, int(np.log10(mat.shape[0]))+1).format(i_frame)
		if os.path.exists(file_path+"."+extension) and continue_:
			continue

		img = toImage(mat[i_frame, :, :])
		saveImage(img[:,:,::-1], file_path)
	clearlines(1)

	printSuccess("Use the following command in the terminal to create a video out of the frames (requires ffmpeg).", clear=True)
	printNeutral("\tffmpeg -i {0}{2}%03d.{3} -c:v libx264 -vf fps={{FPS_COUNT}} -pix_fmt yuv420p {0}{2}{1}.mp4".format(path_folder, stim_name, os.sep, extension), bold=False)

def saveBinCompressed(mat, path_file, tmp_path=None, force=False, callback=lambda *a: None):
	if len(mat.shape) == 2:
		mat = mat[None, ...]
	n_frames, height, width = mat.shape
	dtype = mat.dtype

	if tmp_path is None:
		tmp_path = os.path.basename(path_file)

	path = path_file.split(os.sep)
	output_dir = os.sep.join(path[:-1])
	filename = path[-1]

	final_path = "{}/{}_{}x{}x{}_{}b.tar.gz".format(output_dir, filename, width, height, n_frames, mat.dtype.alignment*8)

	if not(force or not os.path.exists(final_path)):
		return final_path

	os.makedirs(output_dir, exist_ok=True)

	path_rawbin = tmp_path+"numpyBin_save.bin"

	if not os.path.isfile(tmp_path):
		# Beware data is saved un-normalized!
		printNorm("Saving saliency maps as binary file -- {:.2f}GB uncompressed".format(mat.nbytes/1e9),
			end="", clear=True, verbose=2)
		# Create on same disk then copy compressed version to destination
		mat.tofile(path_rawbin)
	else:
		printNorm("Using saliency binary file in [\"{}\"]".format(tmp_path),
			end="", clear=True, verbose=2)
		path_rawbin = tmp_path

	import tarfile
	printNeutral(" - Compressing...", end="", verbose=1); sys.stdout.flush()
	targz = tarfile.open(tmp_path+"numpyBin_save.tar.gz", "w:gz")
	callback(.1)
	targz.add(path_rawbin, arcname="{}_{}x{}x{}_{}b.bin".format(filename, width, height, n_frames, dtype.name[-2:]))
	callback(.9)
	targz.close()
	printNeutral(" {:.2f}GB -- Done".format(os.path.getsize(tmp_path+"numpyBin_save.tar.gz")/1e9), verbose=1)
	printNeutral("  Moving out of [{}] to [{}].".format(os.path.realpath(tmp_path), os.path.realpath(output_dir)), verbose=2)
	os.rename(tmp_path+"numpyBin_save.tar.gz", final_path)
	if not os.path.isfile(tmp_path): os.remove(path_rawbin)

	return final_path

def blendImage(mat, path_file, blend, extension="jpg"):
	"""
	DOC
	"""
	import cv2
	# if blend is not a numpy array, check type and load image if necessary
	assertC(type(blend) in [np.ndarray, str], "Argument \"blend\" must be either a numpy array or a path to a media (image, video). Got \"{}\"".format(type(blend)))
	img = None
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
		if img is None:
			printError("Failed to open media at location [\"{}\"]".format(blend))
			return
	else:
		img = np.ascontiguousarray(blend)

	if np.any(np.array(mat.shape) != np.array(img.shape[:-1])):
		img = cv2.resize(img, mat.shape[::-1]) 

	max_ = mat.flatten().max()
	if max_ != 0:
		mat /= max_
	else:
		mat[:] = 0

	img[:, :, :] = img[:,:, :] * .3 + (mat[:, :, None]*255).astype(np.uint8) * .7

	if path_file is None:
		return img

	cv2.imwrite(path_file+"."+extension, img)

def blendImages(sal_map, path_folder, path_video,
	force=False, ignore_prompt=False):
	import cv2
	assertC(len(sal_map.shape) == 3, "function \"saveImages\" expects a 3D tensor (n_frames, px_height, px_width). Got {}.".format(sal_map.shape), printIfFail=True)

	# Check mimeType of blended content - expects "video"
	cont_type = getMimeType(path_video)
	if cont_type.split("/")[0] != "video":
		printError("You need to pass a path to a video file to function \"blendImages\". Got type \"{}\".".format(cont_type))
		return
	
	# Get filename without extension
	vidname = "_".join(path_video.split(os.sep)[-1].split(".")[:-1])
	# Open video file
	vid = cv2.VideoCapture()
	vid.open(path_video)

	assertC(vid.isOpened(), "Couldn't open video: \"{}\"".format(path_video), printIfFail=True)

	# Extract video specs.
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps    = vid.get(cv2.CAP_PROP_FPS)
	printSuccess("Video info [", vidname, "]: length=", length, ", width=", width, ", heigh=", height, ", fps=", fps, bold=False, sep="", verbose=2)

	out_path = "{0}{1}".format(path_folder, os.sep)

	os.makedirs(out_path,
		exist_ok=True)

	if length != sal_map.shape[0]:
		printWarning("Video frame count [{}] is different from saliency maps count [{}]".format(length, sal_map.shape[0]), bold=False, verbose=2)

	resize = False
	if (width*height) != np.prod(sal_map.shape[1:]):
		printWarning("Video dimensions [{}] are different from saliency maps [{}]. Output will be downsampled.".format([height, width], sal_map.shape[1:]), bold=False, verbose=2)
		resize = True
		newDims = (height, width) if (width*height) < np.prod(sal_map.shape[1:]) else (sal_map.shape[2], sal_map.shape[1])

	continue_ = False
	N_files = len(os.listdir(out_path))
	if not force and\
		os.path.exists(out_path) and\
		N_files > 0:
		ask = " " if not ignore_prompt else "C"
		while ask[0].upper() not in ["C", "F", "S"]:
			printWarning("Output folder [{}] already contains files [{}]".format(out_path, N_files), verbose=0)
			printWarning("What do you want to do?", tab=1, header="", verbose=0)
			printWarning("\"C\" [Continue] to generate missing images,", bold=False, tab=2, header="", verbose=0)
			printWarning("\"F\" [Force] to overwrite existing files,", bold=False, tab=2, header="", verbose=0)
			printWarning("\"S\" [Stop] to exit this function", bold=False, tab=2, header="", verbose=0)
			ask = input("\t\t_ ")
			if len(ask) == 0: ask = " "

			clearlines(6)

		if ask == "C": continue_ = True
		elif ask == "F": continue_ = False
		else: return False

	n_frames = min(length, sal_map.shape[0])
	for i_frame in range(n_frames):
		if i_frame%5==0:
			printNeutral("frame {}/{}".format(i_frame, length), sep="", end="", clear=True, verbose=0)
			sys.stdout.flush()

		path_file = "{}{}{{:0{}}}".format(path_folder, os.sep, int(np.log10(n_frames))+1).format(i_frame)
		if os.path.exists(path_file+".jpg") and continue_:
			continue

		ret, frame = vid.read()
		if resize:
			frame = cv2.resize(frame, newDims, interpolation=cv2.INTER_AREA)

		blendImage(sal_map[i_frame, :,:], path_file, frame)
		i_frame += 1
	clearlines(1)

	vid.release()

	printSuccess("Use the following command in the terminal to create a video from the frames (requires ffmpeg).", clear=True, verbose=1)
	printNeutral("ffmpeg -i {0}{2}%03d.jpg -c:v libx264 -vf fps={3} -pix_fmt yuv420p {0}{2}{1}_blendmap.mp4".format(path_folder, vidname, os.sep, int(fps)), bold=False, verbose=1, tab=1)
