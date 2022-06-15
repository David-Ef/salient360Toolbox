#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018-2022
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

"""
Example:
	`python -m Salient360Toolbox.visualise data/raw_gaze/rawDataStatic2.csv data/stimuli/1_PortoRiverside.png -v 2 --gp --bg --settings Tog.idle=0 Tog.loop=0 GP.type=Solid VP.mult=4 SM.cm=viridis`

	Read and plot data from ./tests/data/rawData2.csv
	Set ./tests/data/1_PortoRiverside.png as background stimulus
	Set verbose level 2 (-v 2). I.e., output everything
	Display raw gaze data (--gp)
	Don't show background stimuli (--bg)
	Set scene settings
		Tog.idle=0 -> turn idle animation off
		Tog.loop=0 -> turn off mouse looping left/right
		GP.type=Solid -> display gaze point as solid colour
		SM.cm=viridis -> select viridis colourmap to draw saliency heatmap
		VP.mult=4 -> enlarge viewport by a factor of 4
"""

from .CLI_options import visualise_opt

args = visualise_opt.Options()
args.parse()
opts = args.opts

settings = {}
if opts.settings is not None:
	for val in opts.settings:
		val = val.split("=")
		if len(val) != 2: val.append("")
		settings[val[0]] = val[1]
del opts.settings

display = {
	"bg": opts.bg,
	"sm": opts.sm,
	"sp": opts.sp,
	"gp": opts.gp
}

from .visualisation import QT_GL
QT_GL.startApplication(paths=opts.paths, args=opts, settings=settings, display=display)
