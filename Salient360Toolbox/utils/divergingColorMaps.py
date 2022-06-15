#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: all colormaps made available by matplotlib.
# Note: By default "cmap" is a blue to red colormap "coolwarm".
# ---------------------------------

from matplotlib.pyplot import colormaps, get_cmap
from numpy import linspace, uint8

# Relative import
try: from .misc import printWarning
except: printWarning = print

# We remove colormaps that are simply the reversed version of others
colormaps = [cmap for cmap in colormaps() if cmap[-2:] != "_r"]
n_levels = 256

def getColormapByName(name, levels=n_levels):
	if name not in colormaps:
		printWarning("There is no colourmap named {}. Fallback: \"coolwarm\"".format(name))
		return cmap

	colormap = get_cmap(name)
	colormap = colormap(linspace(0, 1, levels))
	colormap = colormap[:, :3] * 255
	colormap = colormap.astype(uint8)

	return colormap

def getColormapByIndex(index, levels=n_levels):
	if index < 0 or index >= len(colormaps):
		printWarning("There is no colourmap with index [{}]. Fallback: 45 (\"coolwarm\")".format(index))
		return cmap
	
	cmap_name = colormaps[index]
	return getColormapByName(cmap_name, levels)

cmap = getColormapByName("coolwarm")
