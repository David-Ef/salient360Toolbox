#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: display CLI usages
# ---------------------------------

from .utils.misc import printNeutral
import sys

printNeutral("Usages")

printNeutral("Generate scanpath or saliency data.", tab=1)
from .CLI_options import generate_opt
sys.argv[0] = "python -m Salient360Toolbox.generate"
generate_opt.Options().parser.print_usage()

printNeutral("Compare fixation or saliency data.", tab=1)
from .CLI_options import compare_opt
sys.argv[0] = "python -m Salient360Toolbox.compare"
compare_opt.Options().parser.print_usage()

printNeutral("Visualize gaze data.", tab=1)
from .CLI_options import visualise_opt
sys.argv[0] = "python -m Salient360Toolbox.visualise"
visualise_opt.Options().parser.print_usage()
