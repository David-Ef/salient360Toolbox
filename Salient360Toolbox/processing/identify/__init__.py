#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------
from ...utils.misc import printError

I_algos = {}

try:
	from .I_VT import parse as IVT_parse
	I_algos["I-VT"] = IVT_parse
except:
	printError("Could not import velocity based saccade/fixation parsing algorithm.")

try:
	from .I_HMM import parse as IHMM_parse
	I_algos["I-HMM"] = IHMM_parse
except:
	printError("Could not import HMM saccade/fixation parsing algorithm.\n\trun `pip install pomegranate`")

try:
	from .I_CT import parse as ICT_parse
	I_algos["I-CT"] = ICT_parse
except:
	printError("Could not import cluster based saccade/fixation parsing algorithm.\n\trun `pip install sklearn`")

try:
	# Not implemented
	from .I_DT import parse as IDT_parse
	I_algos["I-DT"] = IDT_parse
except:
	printError("Could not import dispersion based saccade/fixation parsing algorithm.")
