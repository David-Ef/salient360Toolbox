#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# Source: inspired from junyanz implementation of CycleGAN and pix2pix (github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options).
# ---------------------------------

import argparse
import numpy as np

class BaseOptions():
	to_limit = lambda x: list(map(float, x.split("x")))
	# is_range = lambda x: type(x) in [list, tuple, np.ndarray] and len(is_range) > 1

	def __init__(self):
		parser = argparse.ArgumentParser(conflict_handler="resolve")

		# FLAGS -----------------------------------------------------------------------
		# parser.add_argument("-c", "--content",
		# 					help="Type of content to process (def.: auto).",
		# 					default="auto", choices=["auto", "static", "dynamic"],
		# 					type=str)
		parser.add_argument("-t", "--tracking",
							help="Produce Head [\"H\"] or Head+Eye [\"HE\"] data (def.: HE; i.e. head+eye data).",
							default="HE", choices=["HE", "H"],
							type=str)
		parser.add_argument("-a", "--all",
							help="Do everything if applicable (in generate or compare).",
		                    action="store_true")
		parser.add_argument("-f", "--force",
							help="If used, do not produce any file that already exists (def.: False).",
		                    action="store_true")
		parser.add_argument("--nonumba",
							help="If used, do not let numba compile supported functions (def.: False). Numba makes functions run faster, but their first call takes longer to evaluate.",
		                    action="store_true")
		# parser.add_argument("--ignore_prompt",
		# 					help="If true, all question prompting will be silenced and default choices applied.",
		#                     action="store_true")
		parser.add_argument("-v", "--verbose",
							help="Verbosity levels. \"0\": general ouputs; \"1\": process details; \"2\": everything (def.: 1).",
							default=1,
							type=int)
		# OUT -------------------------------------------------------------------------
		parser.add_argument("-o", "--out",
							help="All output files will be saved to this folder.",
							default="./out/",
							type=str)
		# EXTRACT ---------------------------------------------------------------------
		parser.add_argument("-e", "--eye",
							help="Eye data selected to be processed: [L]eft, [R]ight or [B]inocular (def.: R).",
							default="R", choices="LRB",
							type=str)
		#		Preprocessing
		parser.add_argument("-rs", "--resample", help="Resampling rate in Hertz (def.: 0, no resampling).",
							default=0,
							type=int)
		parser.add_argument("--headwindow", help="Temporal window size in msec uesd to compute head rotation trajectory (def.: 100 msec).",
							default=100.,
							type=float)
		parseGp = parser.add_argument_group()
		parseGp.add_argument("--filter",
							help="What filter to use (\"gauss\" for Gaussian or \"savgol\" Savitzky-Golay). No filtering by default.",
							choices=["gauss", "savgol"],
							type=str)
		parseGp.add_argument('--filter-opt',
							help="Filter parameters. Gaussian parameter: \"sigma\"; Savgol parameters: \"win\" for window size and \"poly\" for polynomial order.",
							default=["sigma=4", "win=9", "poly=2"],
							nargs="*")
		#		Fixation analysis
		parseGp = parser.add_mutually_exclusive_group(required=False)
		parseGp.add_argument('--IVT',
							help="Identify fixation with a velocity-based algorithm. Specify velocity threshold this way: `--IVT threshold=120` (def. = 120)",
							# default=["threshold=120"],
							nargs="*")
		parseGp.add_argument('--IHMM',
							help="Identify fixation with an HMM-based algorithm. Specify the number of hidden state this way: `--IHMM nStates=2` (def. and min. state = 2)",
							# default=["nStates=2"],
							nargs="*")
		parseGp.add_argument('--ICT',
							help="Identify fixation with a clustering-based algorithm. Specify the eps and minpts parameters this way: `--ICT eps=.005 minpts=5` (def.: eps = .005, minpts = 5)",
							# default=["eps=.005", "minpts=5"],
							nargs="*")
		parseGp.add_argument('--IDT',
							help="Identify fixation with a dispersion-based algorithm (not implemented)",
							# default=[""],
							nargs="*")

		self.parser = parser
		self.opts = None

	def parse(self):
		import builtins
		import os

		self.opts = self.parser.parse_args()

		# disable/enable use of numba
		os.environ["NUMBA_DISABLE_JIT"] = str(int(self.opts.nonumba))
		builtins.nonumba = self.opts.nonumba

		# clip verbosity level
		self.opts.verbose = np.clip(self.opts.verbose, 0,2)
		builtins.verbose = self.opts.verbose

		# End path with dir separator if there isn't one
		self.opts.out = self.opts.out+os.sep if self.opts.out[-1] != os.sep else self.opts.out
