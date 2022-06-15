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
							help="Two files to compare",
							nargs=2,
							type=str)
		# COMPARE ---------------------------------------------------------------------
		self.parser.add_argument("--salmap",
							help="Compare saliency maps.",
		                    action="store_true")
		self.parser.add_argument("--scanp",
							help="Compare scanpaths.",
		                    action="store_true")
		# PARAMETERS ------------------------------------------------------------------
		self.parser.add_argument("--scanp_weight",
							help="Scanpath comparison weighting (1: distance between fixations; 2: angle between saccade vectors).",
							nargs=5, default=[1.,0.,1.,1.,1.],
							type=float)
		# OUT -------------------------------------------------------------------------
		self.parser.add_argument("--save",
							help="Save data produced (saliency maps and scanpaths) along with comparison results.",
		                    action="store_true")

	def parse(self):
		super(Options, self).parse()
		
		from ..utils.misc import printNeutral, printError

		self.opts.all = (self.opts.salmap and self.opts.scanp) or self.opts.all

		self.opts.salmap = True if self.opts.all else self.opts.salmap
		self.opts.scanp = True if self.opts.all else self.opts.scanp

		if self.opts.all:
			printNeutral("Flag \"all\" is set. Will compare saliency maps and scanpaths.", verbose=1)
			self.opts.salmap = self.opts.scanp = True
		elif not self.opts.salmap and not self.opts.scanp:
			printError("You need to select a type of data to compare (\"--all\", \"--salmap\", \"--scanp\")")
			exit()
		else:
			printNeutral("Compare: {}.".format("saliency maps" if self.opts.salmap else "scanpaths"), verbose=1)
