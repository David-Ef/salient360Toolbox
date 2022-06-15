#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Source: inspired from junyanz implementation of CycleGAN and pix2pix (github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options).
# ---------------------------------

from .base_options import *

class Options(BaseOptions):
	def __init__(self):
		super(Options, self).__init__()

		# IN --------------------------------------------------------------------------
		self.parser.add_argument("i",
							help="File(s) or folders (will try to load all files in folders)",
							nargs="+",
							type=str)
		self.parser.add_argument("--blendfile",
							help="Image or video content to blend with saliency map and scanpath",
							type=str)
		# GENERATE --------------------------------------------------------------------
		self.parser.add_argument("--img-dim",
							help="Image output dimensions. Expects \"0x0\" (width x height)",
							default="2000x1000",
							type=BaseOptions.to_limit)
		self.parser.add_argument("--proc-raw",
							help="If true, will use fixation data to produce saliency and fixation list files. If false, will use raw data as if each sample was a fixation (useful to draw saliency of raw data).",
		                    action="store_true")
		#		Saliency map
		self.parser.add_argument("--sal-gauss",
							help="Sigma in degrees of FoV of the gaussian drawn on saliency maps (def.: 2°)",
							default=2.,
							type=float)
		self.parser.add_argument("--sal-img",
							help="Create saliency map and save as a png.",
		                    action="store_true")
		self.parser.add_argument("--sal_bin",
							help="Create saliency map and save as a binary file.",
		                    action="store_true")
		self.parser.add_argument("--sal-bin-comp",
							help="Create saliency map and save as a compressed binary file.",
		                    action="store_true")
		#		Scanpath (fixation list of features)
		self.parser.add_argument("--scanp-img",
							help="Save scanpath image as png.",
		                    action="store_true")
		self.parser.add_argument("--scanp-file",
							help="Save a scanpath text file with gaze features set in --scanp_feat.",
		                    action="store_true")
		self.parser.add_argument("--scanp-feat",
							nargs="+", default=[9,0,1,12],
							help="Scanpath features to save to file. Will determine scanpath header. See documentation on scanpath for a list of available features.",
							type=int)
		#		Fixation map
		self.parser.add_argument("--fix-img",
							help="Save fixation map as png.",
		                    action="store_true")
		self.parser.add_argument("--fix-bin",
							help="Save fixation map as binary file.",
		                    action="store_true")

	def parse(self):
		super(Options, self).parse()

		# if self.opts.all:
		# 	printNeutral("Flag \"all\" is set. Will compare saliency maps and scanpaths.", verbose=1)
		# 	self.opts.salmap = self.opts.scanp = True
		# elif not self.opts.salmap and not self.opts.scanp:
		# 	printError("You need to select a type of data to compare (\"--all\", \"--salmap\", \"--scanp\")")
		# else:
		# 	printNeutral("Compare: {}.".format("saliency maps" if self.opts.salmap else "scanpaths"), verbose=1)

		from glob import glob
		import numpy as np

		import os
		from ..utils.misc import printError, printWarning, printNeutral

		# Operations planned
		if self.opts.all:
			printNeutral("Flag \"all\" is set. Will generate everything.", verbose=1)
			self.opts.sal_img = self.opts.sal_bin = self.opts.sal_bin_comp = self.opts.scanp_file = self.opts.scanp_img = self.opts.fix_img = self.opts.fix_bin = True
		elif not (self.opts.sal_img or self.opts.sal_bin or self.opts.sal_bin_comp or self.opts.scanp_file or self.opts.scanp_img or self.opts.fix_img or self.opts.fix_bin):
			printError("You need to select a type of data to generate (\"--all\", \"--sal-img\", \"--sal-bin\", \"--sal_bin-comp\", \"--scanp-file\", \"--scanp-img\", \"--fix-img\", \"--fix-bin\")")
			self.opts.i = []
			return

		# Reverse image dimension order (WxH -> HxW)
		self.opts.img_dim = np.array(self.opts.img_dim)[::-1]

		# If an image or video is provided, we supposed that the user wants to use it to blend with gaze data
		self.opts.blend = self.opts.blendfile is not None and os.path.isfile(self.opts.blendfile)

		# Add files and retrieve list of text files from folders
		#	Only add new files
		file_list = []
		for path in self.opts.i:
			path = os.path.realpath(path)
			if os.path.exists(path):
				if os.path.isdir(path):
					file_list.extend([os.path.realpath(file) for file in glob(path+os.sep+"*.csv") if
						os.path.realpath(file) not in file_list])
					file_list.extend([os.path.realpath(file) for file in glob(path+os.sep+"*.txt") if
						os.path.realpath(file) not in file_list])
				elif path not in file_list:
					file_list.append(path)
			else:
				printError("Couldn't find file/folder [\"{}\"]. Ignoring.".format(path))

		# If N files share the same name, we disambiguate them by looking for something unique up in their path
		#	Necessary, since all processed data are saved under the same folder

		#	List all filenames
		file_name_list = [os.path.splitext(os.path.basename(file))[0] for file in file_list]
		#	Unique names and number of times they appear in the list
		names, count = np.unique(file_name_list, return_counts=True)

		#	count>1 means we have duplicate filenames
		names = names[count>1]
		dupl = {name: [] for name in names}
		for iname, name in enumerate(file_name_list):
			if name in dupl.keys():
				dupl[name].append(iname)
			# Remove from list (verbose=2) if same file

		for name, inames in dupl.items():
			# list of split paths
			path_e = [np.array(os.path.dirname(file_list[iname]).split(os.sep)[1:]) for iname in inames]
			# max path length (max number of folders in paths)
			max_l = int(np.max([len(path) for path in path_e]))

			for i in range(1, max_l+1):
				if len(path_e) == 0: break
				# Folder names, starting from the end
				dir_name = [path[-i] for path in path_e]
				# Uniques, indices and counts of folders at this degree across all paths for one unique filename
				uni, idx, count = np.unique(dir_name, return_counts=True, return_index=True)
				# only one unique name means that all paths share the same folder name here
				if len(uni) == 1: continue
				unique = np.equal(count, 1)
				if not np.any(unique): continue
				# Locations of folder names appearing only once
				uni_loc = idx[unique]

				# Loop over unique folder names and prepend it to the filename
				for loc in uni_loc:
					printWarning("File [\"{}\"] in folder [\"{}\"] renamed [\"{}\"]".format(
							file_name_list[inames[loc]], os.sep.join(path_e[loc]), dir_name[loc] + "_" + name),
						header="", verbose=2)
					file_name_list[inames[loc]] = dir_name[loc] + "_" + name
					# remove path at index "loc" as it's been taken care of
					path_e.pop(loc)

		# Update list of raw gaze file to process
		self.opts.i = file_list
		# Add list of filename that will be used when saving processed data
		self.opts.filenames = file_name_list
