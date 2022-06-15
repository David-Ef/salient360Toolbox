#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

"""
python -m Salient360Toolbox.generate ./data/raw_gaze/rawDataStatic1.csv --blendfile ./data/stimuli/1_PortoRiverside.png --all
"""

from glob import glob
import numpy as np
import os

from .CLI_options import generate_opt

args = generate_opt.Options()
args.parse()
opts = args.opts

from .utils import misc

from . import helper
from .generation import saliency as sal_generate
from .generation import scanpath as scanp_generate

os.makedirs(opts.out, exist_ok=True)

misc.printNeutral("Generating data for {} file{}".format(len(opts.i), "s" if len(opts.i) > 1 else ""))

for ipath, path in enumerate(opts.i):

	savename = opts.filenames[ipath]

	gtype = helper.inferGazeType(path)
	if gtype is None:
		misc.printError("Could not identify file as a raw gaze or fixation list file. File: {}".format(savename))
		continue

	# Print in and out path if there is enough space in the terminal
	inpath = path; outpath = opts.out+savename
	max_path_len = len(inpath+outpath); diff_term_width = misc.getTerminalWidth() - (27+4)
	if diff_term_width > 0:
		# Truncate paths and add ellipses if they are too long
		if max_path_len > diff_term_width:
			inpath = os.sep+"…"+inpath[-int(len(inpath)/max_path_len * diff_term_width):]
			# outpath = os.sep+"…"+outpath[-int(len(outpath)/max_path_len * diff_term_width):]
			outpath  = outpath

		misc.printNeutral("Processing file [\"{}\"] to [\"{}\"] as {}".format(inpath, outpath, gtype),
			verbose=0)

	# Load as fixation list
	gaze_data = None
	if gtype == "raw":
		gaze_data, fix_list = helper.loadRawData(path,
			# If gaze tracking, which eye to extract
			eye=opts.eye,
			# Gaze or Head tracking
			tracking=opts.tracking,
			# Resampling at a different sample rate?
			resample=opts.resample,
			# Filtering algo and parameters if any is selected
			filter=helper.filterSettings(opts),
			# Fixation identifier algo and its parameters
			parser=helper.parsingSettings(opts))
	# Load as raw gaze data
	elif gtype == "fixlist":
		fix_list = helper.loadFixlist(path)

	if opts.proc_raw:
		if gaze_data is not None:
			misc.printNeutral("Processing raw gaze data as fixation")

			gaze_data[:, 3] = np.arctan2(gaze_data[:, 0], gaze_data[:, 1])/(2*np.pi) - .25 # longitude
			gaze_data[:, 4] = 1 - (np.arcsin(gaze_data[:, 2]) / np.pi + .5) # latitude
			gaze_data = gaze_data[:, [3,4, 0,1,2]]
			gaze_data[:, :2][gaze_data[:, :2] < 0] += 1

			fix_list = gaze_data
			savename += "_raw"
			
			opts.scanp_file = False
		else:
			misc.printWarning("\"proc_raw\" parameter is not admissible because no raw data is available",
				verbose=0)

	# Operation necessitating to compute a saliency map
	if opts.sal_img or opts.sal_bin or opts.sal_bin_comp:
		sal_map = helper.getSaliencyMap(fix_list[:, [2,3,4, 0,1]], opts.img_dim.copy(), savename,
				# If a bin file exists at this location we load the saliency data from it, unless force_generate is True. Saliency will be saved if caching is True
				path_save=opts.out,
				# Sigma of the 2D Gaussian drawn at the location of fixations
				gauss_sigma=opts.sal_gauss if opts.sal_gauss is not None else 2,
				# Asks to return saliency data rather than a path to a saliency data file if it exists
				force_return_data=True,
				# Generate data instead of reading from pre-existing file
				force_generate=opts.force,
				# Will save saliency to bin file
				caching=False)

		if opts.sal_img:
			sal_image = sal_generate.toImage(sal_map, cmap="coolwarm")[:,:,::-1]
			# Save saliency map as an image
			misc.printNorm("Saliency map to{} image file.".format(" blended" if opts.blend else ""), verbose=0)

			sal_generate.saveImage(sal_map if opts.blend else sal_image,
					outpath+"_{}salmap".format("b" if opts.blend else ""),
				blend=opts.blendfile)

		if opts.sal_bin:
			misc.printNorm("Saliency map to binary file.", verbose=0)
			sal_generate.saveBin(sal_map, outpath)

		if opts.sal_bin_comp:
			misc.printNorm("Saliency map to compressed binary file.", verbose=0)
			sal_generate.saveBinCompressed(sal_map, outpath)

	if opts.scanp_img:
		# Represent a scanpath as a series of fixation points, temporally ordered and labelled
		misc.printNorm("Scanpath data to{} image.".format(" blended" if opts.blend else ""), verbose=0)
		scanp_generate.toImage(fix_list[:, :2], opts.img_dim.copy(),
							   outpath+"_{}scanpath".format("b" if opts.blend else ""),
			blend=opts.blendfile)

	if opts.scanp_file:
		# Save scanpath data to file
		misc.printNorm("Scanpath data to csv text file.", verbose=0)
		scanp_generate.toFile(fix_list, outpath+"_fixation.csv",
			saveArr=opts.scanp_feat,
			mode="w")

	if opts.fix_img or opts.fix_bin:
		fix_map = helper.getFixationMap(fix_list[:, :2], opts.img_dim)

		if opts.fix_img:
			misc.printNorm("Fixation list to fixation map image", verbose=0)
			fix_map_img = sal_generate.toImage(fix_map, cmap="binary", reverse=True)
			sal_generate.saveImage(fix_map_img, outpath+"_fixmap")

		if opts.fix_bin:
			misc.printNorm("Fixation list to fixation map binary", verbose=0)
			sal_generate.saveBin(fix_map, outpath+"_fixmap")
