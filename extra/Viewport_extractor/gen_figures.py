#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID <David@psych.uni-frankfurt.de>
# Year: 2021
# Lab: SGL, Goethe University, Frankfurt
# Comment: fast way to extract viewport using openGL
# ---------------------------------

import numpy as np
import cv2

from ViewportGenerator.generator import ViewportView
from EquirectGenerator.generator import EquirectView

IN_img = "../../data/stimuli/P41_2000x1000.jpg"
OUT_img = "./"

POV_angle = [110, 110]
POV_pix = [512, 512]
Equi_pix = [512, 1024]

camera_angle = [np.pi*.6, 0, np.pi]

viewport = ViewportView(path_img=IN_img,
						angle_dim=POV_angle,
						pixel_dim=POV_pix)

equirect = EquirectView(BGimg_data=np.zeros([*Equi_pix, 3], dtype=np.float32),
						VPimg_data=np.random.rand(*[*POV_pix, 3]).astype(np.float32),
						angle_dim=POV_angle,
						pixel_dim=Equi_pix)

viewport.camAngle = camera_angle
img = viewport.get_frame()
cv2.imwrite(OUT_img+"Viewport.png", img[:,:, ::-1])

# VP content alone in equi
equirect.camAngle = camera_angle
equirect.update_textures(VPdata=np.flipud(img).copy())

VPinEqui = equirect.get_frame()

cv2.imwrite(OUT_img+"VPinEquirect.png", VPinEqui[:,:,::-1])
