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
		self.parser.add_argument("paths",
							help="Path to gaze file or folder containing gaze files. Or path to image or video content to set as background. You can provide multiple paths.",
							nargs="*",
							type=str)
		# DISPLAY OPTIONS -------------------------------------------------------------
		self.parser.add_argument('--bg',
							help="Hide equirectangular image.",
		                    action="store_false")
		self.parser.add_argument('--sm',
							help="Hide saliency heatmap.",
		                    action="store_false")
		self.parser.add_argument('--sp',
							help="Display fixation points.",
		                    action="store_true")
		self.parser.add_argument('--gp',
							help="Display raw gaze points.",
		                    action="store_true")
		# SCENE SETTINGS --------------------------------------------------------------
		self.parser.add_argument('--settings',
							help="Scene setting options. E.g.: SM.Gauss=5 Tog.idle=0",
							nargs="+")
		self.parser.add_argument('--show-settings',
							help="Displays GUI setting documentation",
		                    action="store_true")
		self.parser.add_argument('--load-settings',
							help="Path to a file containing pairs of key-value settings to load",
							type=str)
		# OpenGL SETTINGS --------------------------------------------------------------
		self.parser.add_argument("--opengl",
							help="OpenGL shader version to use (default: 130). Change in cases where 130 is too old for your computer (as seen with some Macs, 400 worked)",
							default=130,
							type=int)

	def parse(self):
		super(Options, self).parse()
