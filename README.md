# salient360Toolbox
Toolbox for processing, visualising, comparing and generating data related to gaze data: as head and eye rotations (VR notably).

The development of this toolbox started with the [Salient360 challenge](https://salient360.ls2n.fr/).
It has since been improved several times.

<img src="/documentation/example.png" alt="Example view" height="300"/>

## ToC

* [Functionalities](#functionalities)
* [Usage](#usage)
* [Install](#install)
* [Limits](#limits)
* [Extra scripts](#extra-scripts)
* [Cite](#cite)
* [Acknowledgements](#acknowledgements)

## Functionalities

<img src="/documentation/schema.png" alt="Flowchart" width="100%"/>

The toolbox contains a significant [helper](Salient360Toolbox/helper.py) script to simplify your task.

TODO: list Outputs, list/define computed scanpath features, list/describe comparison algo (saliency & scanpath)

## Usage

There are three way to use the toolbox: via a GUI, scripting, or the command line.
Scripting offers more features than the other methods.

### Scripting

The root folder named [scripting](scripting) contains examples showing how to use all functionalities of the toolbox.

### GUI

`python -m Salient360Toolbox.visualise`

<img src="/documentation/example.png" alt="Example view" height="300"/>

Implemented with pyQT, the GUI lets you drag-and-drop gaze data files and stimuli (image or video) to visualise gaze.
It displays scanpath and saliency maps, fixation list or raw data.

You can use it to export data as csv files or images.

### CLI

Output of `python -m Salient360Toolbox`

```
Usages
  Generate scanpath or saliency data.
usage: python -m Salient360Toolbox.generate [-h] [-t {HE,H}] [-a] [-f]
                                            [--nonumba] [-v VERBOSE] [-o OUT]
                                            [-e {L,R,B}] [-rs RESAMPLE]
                                            [--headwindow HEADWINDOW]
                                            [--filter {gauss,savgol}]
                                            [--filter-opt [FILTER_OPT [FILTER_OPT ...]]]
                                            [--IVT [IVT [IVT ...]] | --IHMM
                                            [IHMM [IHMM ...]] | --ICT
                                            [ICT [ICT ...]] | --IDT
                                            [IDT [IDT ...]]]
                                            [--blendfile BLENDFILE]
                                            [--img-dim IMG_DIM] [--proc-raw]
                                            [--sal-gauss SAL_GAUSS]
                                            [--sal-img] [--sal_bin]
                                            [--sal-bin-comp] [--scanp-img]
                                            [--scanp-file]
                                            [--scanp-feat SCANP_FEAT [SCANP_FEAT ...]]
                                            [--fix-img] [--fix-bin]
                                            i [i ...]
  Compare fixation or saliency data.
usage: python -m Salient360Toolbox.compare [-h] [-t {HE,H}] [-a] [-f]
                                           [--nonumba] [-v VERBOSE] [-o OUT]
                                           [-e {L,R,B}] [-rs RESAMPLE]
                                           [--headwindow HEADWINDOW]
                                           [--filter {gauss,savgol}]
                                           [--filter-opt [FILTER_OPT [FILTER_OPT ...]]]
                                           [--IVT [IVT [IVT ...]] | --IHMM
                                           [IHMM [IHMM ...]] | --ICT
                                           [ICT [ICT ...]] | --IDT
                                           [IDT [IDT ...]]] [--salmap]
                                           [--scanp]
                                           [--scanp_weight SCANP_WEIGHT SCANP_WEIGHT SCANP_WEIGHT SCANP_WEIGHT SCANP_WEIGHT]
                                           [--save]
                                           i i
  Visualize gaze data.
usage: python -m Salient360Toolbox.visualise [-h] [-t {HE,H}] [-a] [-f]
                                             [--nonumba] [-v VERBOSE] [-o OUT]
                                             [-e {L,R,B}] [-rs RESAMPLE]
                                             [--headwindow HEADWINDOW]
                                             [--filter {gauss,savgol}]
                                             [--filter-opt [FILTER_OPT [FILTER_OPT ...]]]
                                             [--IVT [IVT [IVT ...]] | --IHMM
                                             [IHMM [IHMM ...]] | --ICT
                                             [ICT [ICT ...]] | --IDT
                                             [IDT [IDT ...]]] [--bg] [--sm]
                                             [--sp] [--gp]
                                             [--settings SETTINGS [SETTINGS ...]]
                                             [--show-settings]
                                             [--load-settings LOAD_SETTINGS]
                                             [--opengl OPENGL]
                                             [paths [paths ...]]
```

## Install

The toolbox needs the following modules to operate.
I usually use a Conda environment like so:

```bash
conda create -n salient360 python=3.8
conda activate salient360

pip install scipy==1.5.2 numpy==1.19.2 matplotlib==3.3.2 pyopengl==3.1.1a1 numba==0.51.2 scikit-image==0.17.2 statsmodels==0.12.1

pip install opencv-python==4.2.0.32 numpy-quaternion==2020.11.2.17.0.49 PyQt5==5.11.3
```

## Limits

### Unfinished parts

### Known issues

### TODOs

## Extra scripts

### [Polar plot generator](extra/Polar_plots_generator)

This is the script I use in my papers to generate polar plots of saccade angle data.
It takes a simple csv file as input with, per line: amplitude and angle.

<img src="/extra/Polar_plots_generator/example.png" alt="polar plot output example" width="200"/>

### [Viewport extractor](extra/Viewport_extractor)

This set of script relies on openGL to quickly extract a viewport (the portion of a 360 stimulus that would be shown inside a VR headset, for example).
One script extracts, the other places the extracted view back into an equirectangular view.

* Original equirectangular stimulus:
<img src="/data/stimuli/P41_2000x1000.jpg" alt="Equirectangular stimulus" width="300"/>

* Extracted view:
<img src="/extra/Viewport_extractor/Viewport.png" alt="Extracted viewport" width="100"/>

* Extracted view back in equirectangular:
<img src="/extra/Viewport_extractor/VPinEquirect.png" alt="Viewport in equirectangular" width="300"/>

*The extracted view is not perfect, it shows slight deformations.*

As an example, I used this pair of scripts to generate videos visualising user behaviour.
You can download an example of such a video [here](/documentation/UserBehaviourExample.mp4).

## Cite

* David, E., Gutiérrez, J., Võ, M. L. H., Coutrot, A., Perreira Da Silva, M., & Le Callet, P. (2024). The Salient360! toolbox: Handling gaze data in 3D made easy. Computers & Graphics, 103890. [10.1145/3588015.3588406](https://doi.org/10.1016/j.cag.2024.103890)

```
@article{david2024salient360,
  title={The Salient360! toolbox: Handling gaze data in 3D made easy},
  author={David, Erwan and Guti{\'e}rrez, Jes{\'u}s and V{\~o}, Melissa L{\`e}-Hoa and Coutrot, Antoine and Perreira Da Silva, Matthieu and Le Callet, Patrick},
  journal={Computers \& Graphics},
  pages={103890},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgements

Although, I wrote the toolbox alone, it is conceptually the product of discussions and interactions between the following fine folks:

* Antoine Coutrot
* Erwan David
* Jesús Gutiérrez
* Melissa Le-Hoa Võ
* Patrick Le callet
* Matthieu Perreira Da Silva

I worked on this project as part of the LS2N (*Laboratory of Digital Sciences of Nantes*, France) and while at the Scene Grammar lab (Frankfurt, Germany) with the following funding by :

* RFI Atlanstic2020.
* Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) project number 222641018 SFB/TRR 135, subproject C7 to MLV.

Part of the [Multimatch](https://link.springer.com/article/10.3758/s13428-012-0212-2) implementation was taken from the implementation by [Adina Wagner](https://www.adina-wagner.com/) ([repo](https://github.com/adswa/multimatch_gaze)).
Thank you to her for her work.
