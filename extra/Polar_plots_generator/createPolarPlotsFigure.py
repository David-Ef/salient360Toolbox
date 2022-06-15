#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2, os

PATH = "./plots/"
FILENAMEM = "Nfixations_eR_c{}_{}_{}.png".format # C, Type
DOMAIN = ["VP", "Sp", "HMD"]#, "Subs"]
TYPE = ["Abs", "Rel"]

font_path = "/home/sgl/.fonts/Helvetica-Normal.ttf"
font_pathB = "/home/sgl/.fonts/Helvetica-bold.ttf"
font = ImageFont.truetype(font_path, 100)
fontB = ImageFont.truetype(font_pathB, 100)
fontTicks = ImageFont.truetype(font_path, 60)

gap = 200
margin = 100
mat_dim = [3, 1]
img_dim = [920, 920]
H = mat_dim[0] * img_dim[0] + gap * (mat_dim[0]+1)
W = mat_dim[1] * img_dim[1] + gap * (mat_dim[1]+1)

def drawTextRotated(txt, font, orientation=0):
	# Image for text to be rotated
	img_txt = Image.new('RGBA', font.getsize(txt))
	draw_txt = ImageDraw.Draw(img_txt)
	draw_txt.text((0, 0), txt, font=font, fill=(0, 0, 0, 255))
	return img_txt.rotate(90, expand=1)

for domain in DOMAIN:
	print(domain)
	for type_ in TYPE:
		outname = "{}_{}_polarPlots.png".format(domain, type_)
		# if os.path.exists(outname): continue
		print(type_)
		dispMat = np.zeros([H, W, 4], dtype=np.uint8)
		for iC in range(0, 3):
			filename = PATH+FILENAMEM(iC, domain, type_)
			img = cv2.imread(filename, -1) # Load with alpha channel
			assert img is not None, "Couldn't find image \"{}\"".format(filename)
			print(filename, img is not None)

			iS = 0

			h = iC - 1
			w = iS

			if iC == 0:
				h = 2

			img = img[80:1000, 80:1000]

			h = h * img_dim[0] + gap * (h+1)
			w = w * img_dim[1] + gap * (w+1)

			# print(dispMat.shape, ":", h, h+img_dim[0], ",", w, w+img_dim[1])

			dispMat[h:h+img_dim[0], w:w+img_dim[1], :] = img

		print("Add margings")
		dispMat = np.concatenate([np.zeros([dispMat.shape[0], margin, 4]), dispMat], axis=1)
		# dispMat = np.concatenate([np.zeros([margin, dispMat.shape[1], 4]), dispMat], axis=0)

		length = img_dim[0] - 48*2

		print("Draw ticks")
		MAX_PLOT = 50
		tickLabels = np.array([*np.linspace(MAX_PLOT, 0, 6), *np.linspace(10, MAX_PLOT, 5)], dtype=int)
		nticks = tickLabels.shape[0]
		dist = 104
		xStart = margin
		yStart = 48 # margin in polar plots
		for iy_plot in range(3):
			for ix_plot in range(1):

				Vx1 = xStart+20 + (img_dim[0]+gap)*ix_plot + gap
				Vy1 = yStart + (img_dim[0]+gap)*iy_plot + gap
				Vx2 = Vx1+20
				Vy2 = Vy1

				Hx1 = xStart+58 + (img_dim[0]+gap)*ix_plot + gap
				Hy1 = yStart-30 + (img_dim[0]+gap)*iy_plot + gap
				Hx2 = Hx1
				Hy2 = Hy1+20

				for iTick in range(nticks):
					# Vert
					dispMat = cv2.line(
								dispMat,
								tuple([Vx1, Vy1 + int(length*(iTick/(nticks-1) ))]),
								tuple([Vx2, Vy2 + int(length*(iTick/(nticks-1) )) ]),
								(0, 0, 0, 255),
								thickness=4, lineType=cv2.LINE_AA
								)

					# Horiz
					dispMat = cv2.line(
								dispMat,
								tuple([Hx1 + int(length*(iTick/(nticks-1))), Hy1 ]),
								tuple([Hx2 + int(length*(iTick/(nticks-1))), Hy2 ]),
								(0, 0, 0, 255),
								thickness=4, lineType=cv2.LINE_AA
								)

		print("Add margings")
		dispMat = np.concatenate([np.zeros([dispMat.shape[0], margin, 4]), dispMat], axis=1)
		# dispMat = np.concatenate([np.zeros([margin, dispMat.shape[1], 4]), dispMat], axis=0)

		PIL_Img = Image.fromarray(dispMat.astype(np.uint8))	
		draw = ImageDraw.Draw(PIL_Img)

		print("Write tick labels")
		for iy_plot in range(3):
			for ix_plot in range(1):

				Vx1 = xStart + margin + (img_dim[0]+gap)*ix_plot + gap
				Vy1 = yStart + (img_dim[0]+gap)*iy_plot + gap

				Hx1 = xStart + margin*2 + (img_dim[0]+gap)*ix_plot + gap
				Hy1 = yStart - margin + (img_dim[0]+gap)*iy_plot + gap

				for iTick in range(nticks):
					text = str(tickLabels[iTick])
					width_text, heigth_text = fontTicks.getsize(text)
					# Vert
					draw.text( (Vx1 - width_text, Vy1 - heigth_text//2 + int(length*(iTick/(nticks-1) )) ),
							text, font=fontTicks, fill=(0, 0, 0, 255))

					# Horiz
					draw.text((Hx1 - width_text + int(length*(iTick/(nticks-1))), Hy1 ),
							text, font=fontTicks, fill=(0, 0, 0, 255))

		# for iText, text, in enumerate(["6 deg."]):
		# 	width_text = font.getsize(text)[0]
		# 	draw.text((margin*2 + img_dim[0]*iText + img_dim[0]/2 - width_text/2 + gap*(iText+1), 150),
		# 			text, font=font, fill=(0, 0, 0, 255))

		for iText, text, in enumerate(["Central", "Peripheral", "Control"]):
			t = drawTextRotated(text, font, orientation=90)
			height_text = font.getsize(text)[0]
			PIL_Img.paste(t, (100, int(margin*2 + img_dim[1]*iText + img_dim[1]/2 - height_text/2 + gap*(iText+1))))

		dispMat = np.array(PIL_Img)

		print("Add margings")
		dispMat = np.concatenate([np.zeros([dispMat.shape[0], margin, 4]), dispMat], axis=1)
		dispMat = np.concatenate([np.zeros([margin, dispMat.shape[1], 4]), dispMat], axis=0)

		PIL_Img = Image.fromarray(dispMat.astype(np.uint8))

		if domain == "HMD":
			draw = ImageDraw.Draw(PIL_Img)
			
			# # X axis title
			# text = "Mask radius"
			# width_text = fontB.getsize(text)[0]
			# draw.text((margin*5 + img_dim[0]/2 - width_text/2, 0), text, font=fontB, fill=(0, 0, 0, 255))
			# Y axis title

			text = "Mask type"
			t = drawTextRotated(text, fontB, orientation=90)
			height_text = fontB.getsize(text)[0]
			PIL_Img.paste(t, (0, int(margin*3 + (img_dim[1]*3)/2 - height_text/2+ (gap*4)/2)))
			PIL_Img.thumbnail(( PIL_Img.size[0]//2, PIL_Img.size[1]//2 ), Image.ANTIALIAS)

		dispMat = np.array(PIL_Img)
		cv2.imwrite(outname, dispMat)
