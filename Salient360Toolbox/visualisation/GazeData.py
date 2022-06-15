#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import numpy as np
import os, sys

from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from ..utils.misc import *
from .. import helper
from . import ModalWindows

class GazeData(object):
	iconCallback = None

	def filterSettings(self):
		filterName = self.parent.sceneOption["GP.preproc.filter"]
		filterParam = {}

		name = filterName[0].lower()
		if name == "g":
			filterParam["sigma"] = self.parent.sceneOption["GP.preproc.filter.gauss.sigma"]
		elif name == "s":
			filterParam["win"] = self.parent.sceneOption["GP.preproc.filter.savgol.win"]
			filterParam["poly"] = self.parent.sceneOption["GP.preproc.filter.savgol.poly"]

		return {"name": filterName, "params": filterParam}

	def parsingSettings(self):
		parserName = self.parent.sceneOption["GP.parse"]
		parserParam = {}

		for name, _ in self.parent.sceneOption:
			if name.split(".")[1] == parserName.replace("-", ""):
				parserParam[name.split(".")[-1]] = self.parent.sceneOption[name]

		return {"name": parserName, "params": parserParam}

	def __init__(self, parent):

		self.parent = parent
		self.MainWindow = self.parent.parent
		self.path_raw = []
		self.path_fix = []

		self.Ifilter = self.filterSettings
		self.Ialgo = self.parsingSettings
		self.tracking = lambda: ["H", "HE"][self.parent.sceneOption["GP.tracking"]<3]
		self.eye = lambda: ["L", "R", "B", "H"][self.parent.sceneOption["GP.tracking"]]
		self.resample = lambda: self.parent.sceneOption["GP.preproc.resample"]

		self.processing = False

		self.reset_()

		self.dim = lambda: [self.parent.sceneOption["SM.mapY"],
							self.parent.sceneOption["SM.mapX"]]

		# None: Will show warning
		# True: Won't show and will apply default behaviour
		# False: Won't show and won't apply default behaviour
		self.showWarnings = {
			"E2E_DataFallback": True if self.MainWindow.args.force else None,
			"E2H_DataFallback": True if self.MainWindow.args.force else None,
			"NoDataAvailable": True if self.MainWindow.args.force else None
		}

	def process(self, path, concatenate=False):
		from ..utils.misc import inferGazeType

		if not concatenate:
			self.path_raw = []
			self.path_fix = []

		self.processing = True

		count = {"raw": 0, "fixlist": 0}

		if not concatenate:
			self.reset_()
			if len(path) > 1: concatenate = True

		for file in path:
			gtype = inferGazeType(file)
			if gtype is None: continue

			self.MainWindow.setStatusBar("Parsing as {} file: {}".format(
					"raw gaze" if gtype == "raw" else "fixation list", file),
				perma=False)

			# Load as fixation list
			if gtype == "raw":
				if file not in self.path_raw: self.path_raw.append(file)
				out = self.loadRawData(file, add=concatenate)
			# Load as raw gaze data
			elif gtype == "fixlist":
				if file not in self.path_fix: self.path_fix.append(file)
				out = self.loadFixlist(file, add=concatenate)

			self.MainWindow.setStatusBar("Parsed as {} file: {}".format(
					"raw gaze" if gtype == "raw" else "fixation list", file),
				perma=False)
			count[gtype] += int(out)

		nGazeSamples = self.raw_gaze.shape[0] if self.raw_gaze is not None else 0
		nFixations = self.fix_list.shape[0] if self.fix_list is not None else 0

		self.MainWindow.setStatusBar("{} raw sample{} | {} fixation{}".format(
				nGazeSamples, "s" if nGazeSamples>1 else "",
				nFixations, "s" if nFixations>1 else ""),
			perma=True)

		self.processing = False

		dataOpt = self.parent.SettingInterface.SMdataChoice
		dataOpt.model().item(0,0).setEnabled(nGazeSamples != 0)
		dataOpt.model().item(1,0).setEnabled(nFixations != 0)

		if nGazeSamples == 0 and self.parent.sceneOption["SM.fromfix"] == 0:
			self.parent.sceneOption["SM.fromfix"] = 1

		if (count["raw"]+count["fixlist"]) > 0:
			self.parent.SaveInterface.SetSaveButtonsDisabled(False)
			self.generate()
		else:
			self.parent.SaveInterface.SetSaveButtonsDisabled(True)

		return count["raw"], count["fixlist"]

	def reset_(self):
		self.raw_gaze = None
		self.gaze_marker = None # saccade/fixation identifier
		self.fix_list = None # fixation feature
		self.sal_map = None # grayscale
		self.sal_image = None # color map coded
		self.fix_map = None # Display as icon in options
		self.blend_sal_map = None # Saliency map blended with background content

		self.path_raw = []
		self.path_fix = []

		if hasattr(self.MainWindow, "container"):
			self.parent.SaveInterface.SetSaveButtonsDisabled(True)

	def reset(self):
		self.reset_()

		empty = np.zeros([1, 1])
		GazeData.iconCallback("sm", empty, [1, 1])
		GazeData.iconCallback("gp", empty, [1, 1])
		GazeData.iconCallback("sp", empty, [1, 1])
		self.parent.Equirect.setBackgroundTexture(np.zeros([1, 1, 3]))

		self.parent.Equirect.set_GazePointData()
		self.parent.Equirect.update_texture()
		self.parent.Equirect.update()

		self.MainWindow.setStatusBar("", perma=True)

	def generate(self, parse=False, saliency=True, salmap=True, rawmap=True, fixmap=True, blend=True,
		precision=np.float32):

		self.processing = True

		if parse:
			self.process(self.path_raw + self.path_fix, False)
			self.parent.Equirect.set_GazePointData()
			return
		
		if saliency:
			self.computeSaliency(precision)
		if salmap:
			self.genSalMap()
		if blend:
			self.genBlendSalmap()
		if rawmap:
			self.genGazeMap()
		if fixmap:
			self.genFixMap()

		self.processing = False

		self.parent.Equirect.update_texture()
		self.parent.Equirect.update()

	def computeSaliency(self, precision=np.float32):
		if self.fix_list is None: return
		from ..generation import saliency

		data = None
		if self.parent.sceneOption["SM.fromfix"] == 0 and self.raw_gaze is not None:
			# Rearrange columns: x,y,z, lon,lat
			data = self.raw_gaze[:, [2,3,4, 0,1]].copy()

		elif self.fix_list is not None:
			# Rearrange columns: x,y,z, lon,lat
			data = self.fix_list[:, [2,3,4, 0,1]].copy()
		if data is None:
			printWarning("Could not generate saliency data from {} because there is no data.".format(
				"fixations" if self.parent.sceneOption["SM.fromfix"] == 1 else "raw gaze"))
			return

		pDiag = ModalWindows.progressBar("Computing saliency map", "stop", self.MainWindow)
		pDiag.show()
		def callback(frac):
			# Uncomment to see the saliency image as it is calculated
			# self.genBlendSalmap()
			# self.genSalMap()
			# self.parent.Equirect.update_texture()
			# self.parent.Equirect.update()

			return pDiag.setValue(frac)

		# SalMap
		self.sal_map = np.zeros(self.dim(), dtype=precision) # H, W
		# try:
		saliency.getSaliency(self.sal_map, data, # x,y,z, lon,lat
				gauss_sigma=self.parent.sceneOption["SM.Gauss"],
				callback=callback)
		# except Exception as e:
		# 	print(e)
		# 	printWarning("Error when generating saliency data from {}.".format(
		# 		"fixations" if self.parent.sceneOption["SM.fromfix"] == 1 else "raw gaze"))
		# 	self.sal_map = None
		pDiag.close()

	def genSalMap(self):
		if self.sal_map is None: return
		from ..generation.saliency import toImage

		self.sal_image = toImage(self.sal_map,
			cmap=self.parent.sceneOption["SM.cm"],
			reverse=self.parent.sceneOption["SM.rev"])
		
		self.sal_image = self.sal_image.reshape([self.dim()[1], self.dim()[0], 3])
		GazeData.iconCallback("sm", self.sal_image, self.dim())

	def genBlendSalmap(self):
		if self.parent.Equirect.backgroundImage is None or self.sal_map is None:
			self.parent.SaveInterface.SMbimg.setDisabled(True)
			self.parent.SaveInterface.SPbimg.setDisabled(True)
			return
		from ..generation.saliency import blendImage

		dim = self.parent.Equirect.backgroundImage.shape
		self.blend_sal_map = blendImage(self.sal_map.copy(),
			None, self.parent.Equirect.backgroundImage.copy().reshape([dim[1], dim[0], dim[2]]))
		self.blend_sal_map = self.blend_sal_map.reshape([self.dim()[1], self.dim()[0], 3])

		self.parent.SaveInterface.SMbimg.setDisabled(False)
		self.parent.SaveInterface.SPbimg.setDisabled(False)

	def genGazeMap(self):
		if self.raw_gaze is None or len(self.raw_gaze) == 0:
			self.parent.DisplayInterface.GPImg.toggleSelected(False)
			return
		from ..generation.scanpath import toFixationMap

		dim = [100, 200]
		self.gaze_map = toFixationMap(self.raw_gaze[:, :2], dim)
		self.gaze_map = np.repeat(self.gaze_map[:,:,None], 3, axis=2)
		self.gaze_map = (self.gaze_map/self.gaze_map.max() * 255).astype(np.uint8)

		self.parent.DisplayInterface.GPImg.toggleSelected(self.parent.Equirect.showGP)
		GazeData.iconCallback("gp", self.gaze_map, dim)

	def genFixMap(self):
		if self.fix_list is None or len(self.fix_list) == 0:
			self.parent.DisplayInterface.SPImg.toggleSelected(False)
			return
		from ..generation.scanpath import toFixationMap

		dim = [100, 200]
		self.fix_map = toFixationMap(self.fix_list[:, :2], dim)
		self.fix_map = np.repeat(self.fix_map[:,:,None], 3, axis=2)
		self.fix_map = (self.fix_map/self.fix_map.max() * 255).astype(np.uint8)

		self.parent.DisplayInterface.SPImg.toggleSelected(self.parent.Equirect.showSP if self.raw_gaze is not None else True)
		GazeData.iconCallback("sp", self.fix_map, dim)

	# Data
	#	Load
	def loadFixlist(self, path, add=False):

		fix_list = helper.loadFixlist(path)

		if type(fix_list) != np.ndarray and fix_list == -1:
			QtWidgets.QMessageBox.warning(None, "Parsing error", "File does not contain fixation data.\nPlease check data format.\nFile: \"{}\"".format(path))
			return False

		if add and self.fix_list is not None:
			self.fix_list = np.concatenate([self.fix_list, fix_list], axis=0)
		else:
			self.fix_list = fix_list

		return True

	def loadRawData(self, path, add=False):

		pDiag = ModalWindows.progressBar("Parsing raw gaze data", "stop", self.MainWindow)
		pDiag.show()

		def warningMissingData(missingEye, fallback):
			showWarningsEl = "E2E_DataFallback" if fallback[0] != "H" else "E2H_DataFallback"

			if self.showWarnings[showWarningsEl] is None:

				askWarningCheck = ModalWindows.warningCheck(
					title="Missing data: {}".format(missingEye),
					message="{} eye data are missing{}.<br />Do you want to fall back on <b>{} data</b>?".format(
							missingEye.title(),
						 	" but <u>no alternative eye data are available</u>" if fallback[0] == "H" else "",
							fallback.lower()),
					parent=self.MainWindow)

				askWarningCheck.exec()

				result, future = askWarningCheck.getResults()
				result = result == 1

				self.showWarnings[showWarningsEl] = result if future else None

				return result, future

			return self.showWarnings[showWarningsEl], True

		idx, valid = helper.FindRawFeaturesByHeader(path, returnValid=True)

		gazeOpt = self.parent.SettingInterface.GPtrackChoice
		gazeOpt.model().item(0,0).setEnabled(
			valid["eye"]["L"] and gazeOpt.model().item(0,0).isEnabled()
			)
		gazeOpt.model().item(1,0).setEnabled(
			valid["eye"]["R"] and gazeOpt.model().item(1,0).isEnabled()
			)
		gazeOpt.model().item(2,0).setEnabled(
			valid["eye"]["B"] and gazeOpt.model().item(2,0).isEnabled()
			)
		gazeOpt.model().item(3,0).setEnabled(
			np.any(list(valid["head"].values())) and gazeOpt.model().item(3,0).isEnabled()
			)

		# Assert that we have the minimum necessary data
		if not np.any(list(valid["head"].values())):
			# No data fallback: if you want to be able to ignore camera data, add columns with camera rotation data [1,0,0,0] (no rotation).
			pDiag.close()
			printError("Head rotation data are missing in file: {}".format(path), verbose=2)
			
			return False

		eyeOpts = [gazeOpt.itemText(i)for i in range(gazeOpt.count())]
		eyeOpts0 = [eye[0] for eye in eyeOpts]
		eyes = eyeOpts0.copy()

		settingIdx = self.parent.sceneOption["GP.tracking"]
		eye = eyes[settingIdx] # Choice made in settings

		# Checking that we have the data we need according to the "Gaze points" settings 
		#	Will suggest to fallback to other data if needed
		fallback = None
		if eye in eyes[:-1]: # Head data left out
			if not valid["eye"][eye]:
				printError("Missing {} data in file: {}".format(eye, path), header="loadRawData")

				# Fallback 1: other eye data + binocular
				eyes.remove(eye)
				if valid["eye"][eyes[0]]:
					eye = eyes[0]
					# Warning - fell back on other eye data (R or L)
				elif valid["eye"][eyes[1]]:
					eye = eyes[1]
					# Warning - fell back on other eye data (Binocular)
				elif np.any(list(valid["head"].values())):
					# Not possible due to condition above
					# Fallback 2: head data
					eye = eyes[2]
				else:
					# Not possible due to condition above
					pass

				fallback = warningMissingData(eyeOpts[settingIdx].split(" ")[0], eyeOpts[eyeOpts0.index(eye)])

				printError("Eye data missing in file: {}. {}".format(path,
					"Falling back." if fallback[0] else "Not processing further."
					), verbose=2)

				if fallback[0] is False:
					pDiag.close()
					return False

				self.parent.sceneOption["GP.tracking"] = eyeOpts0.index(eye)

		raw_gaze, fix_list, gaze_marker = helper.loadRawData(path,
			# If gaze tracking, which eye to extract
			eye=self.eye(),
			# Gaze or Head tracking
			tracking=self.tracking(),
			# Resampling at a different sample rate?
			resample=self.resample(),
			# Filtering algo and parameters if any is selected
			filter=self.Ifilter(),
			# Fixation identifier algo and its parameters
			parser=self.Ialgo(),
			# Progress bar callback
			callback=pDiag.setValue,
			return_label=True
			)

		# Reset previous setting if fallback
		if fallback is not None and not fallback[1]:
			self.parent.sceneOption["GP.tracking"] = settingIdx
		if pDiag.wasCanceled(): return False
		
		if fix_list is None:
			pDiag.close()
			QtWidgets.QMessageBox.warning(None, "Parsing error", "Found 0 fixation\nin file: \"{}\"".format(path))
			return False

		pDiag.close()
		# Without a call to "destroy" approx 1 window out of 30 will fail to disappear (when loading a lot of files in a row)
		pDiag.destroy()
		
		# Unit sphere to equirectangular transformation
		# raw_gaze = raw_gaze[:, :3]
		raw_gaze[:, 3] = np.arctan2(raw_gaze[:, 0], raw_gaze[:, 1])/(2*np.pi) -.25 # longitude
		raw_gaze[:, 4] = 1 - (np.arcsin(raw_gaze[:, 2]) / np.pi + .5) # latitude
		raw_gaze = raw_gaze[:, [3,4, 0,1,2]]
		raw_gaze[:, :2][raw_gaze[:, :2]<0] += 1

		# Reorganize feature columns
		reorder = [0,1, 2,3,4, 9,12]
		fix_idx_arr = np.arange(fix_list.shape[1]).tolist()
		fix_idx_arr = reorder + [idx for idx in fix_idx_arr if idx not in reorder]
		fix_list = fix_list[:, fix_idx_arr]

		if add:
			self.raw_gaze = raw_gaze if self.raw_gaze is None else np.concatenate([self.raw_gaze, raw_gaze], axis=0)
			self.gaze_marker = gaze_marker if self.gaze_marker is None else np.concatenate([self.gaze_marker, gaze_marker], axis=0)
			self.fix_list = fix_list if self.fix_list is None else np.concatenate([self.fix_list, fix_list], axis=0)
		else:
			self.raw_gaze = raw_gaze
			self.gaze_marker = gaze_marker
			self.fix_list = fix_list

		return True

	# Export methods
	def saveSalmapImg (self, file_type="*", blend=False):
		# Save dialog window
		parent = self.MainWindow.container
		qDiag = QtWidgets.QDialog(parent)
		qDiag.setWindowTitle("Save {}saliency map".format("blended " if blend else ""))

		check = {"savefile": False}
		global data; data = ["./blend_salmap.png" if blend else "./salmap.png", ""]

		def updateSaveFile():
			global data
			data.clear()
			parent.spawnFileDialog(file_name="salmap.png", file_type=file_type, out=data)

			data = [data[0][0], data[0][1]]
			if len(data[0])>0:
				if data[0][-4:] not in [".jpg", ".png"]:
					data[0] += ".png"

				saveFileLab.setText(data[0] if len(data[0])< 50 else "..."+data[0][-50:])
				check["savefile"] = True
			else:
				saveFileLab.setText("[no file selected]")
				check["savefile"] = False

			acceptBtn.setDisabled(not check["savefile"])

		def closeEvent(event): qDiag.reject(); qDiag.close()
		qDiag.closeEvent = closeEvent

		# Get save file location
		saveFileBtn = QtWidgets.QPushButton("File location")
		saveFileLab = QtWidgets.QLabel(data[0])#"[no output file selected]")
		saveFileLab.setStyleSheet("border: 2px solid rgb(200, 200, 200);border-radius: 5px;")
		acceptBtn = QtWidgets.QPushButton("Save")
		acceptBtn.setDisabled(False)

		dimWid = QtWidgets.QWidget()
		MapSizeX = QtWidgets.QSpinBox()
		MapSizeX.setRange(100, 10000)
		MapSizeX.setSingleStep(100)
		MapSizeX.setValue(self.parent.sceneOption["SM.mapX"])
		MapSizeY = QtWidgets.QSpinBox()
		MapSizeY.setRange(100, 10000)
		MapSizeY.setSingleStep(100)
		MapSizeY.setValue(self.parent.sceneOption["SM.mapY"])
		dimBox = QtWidgets.QHBoxLayout(parent)
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("Map dimensions"))
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("[W:"))
		dimBox.addWidget(MapSizeX)
		dimBox.addWidget(QtWidgets.QLabel(", H:"))
		dimBox.addWidget(MapSizeY)
		dimBox.addWidget(QtWidgets.QLabel("]"))
		dimWid.setLayout(dimBox)

		saveFileBtn.clicked.connect(updateSaveFile)
		acceptBtn.clicked.connect(lambda x: qDiag.accept())

		layout = QtWidgets.QGridLayout(parent)
		layout.addWidget(saveFileBtn, 0, 0, 1, 1)
		layout.addWidget(saveFileLab, 0, 1, 1, 2)
		layout.addWidget(dimWid, 1, 0, 1, 3)
		layout.addWidget(QtWidgets.QLabel("If you modify the dimensions above, saliency will be recomputed."), 2, 0, 1, 3)
		layout.addWidget(acceptBtn, 3, 0, 1, 3)
		qDiag.setLayout(layout)

		qDiag.setModal(True)
		qDiag.exec()

		saveFile = data
		if qDiag.result() == 0 or len(saveFile) != 2:
			# printWarning("Dialog cancelled.")
			return

		dim = [MapSizeY.value(), MapSizeX.value()]
		if dim != self.dim():
			self.parent.sceneOption["SM.mapX"] = MapSizeX.value()
			self.parent.sceneOption["SM.mapY"] = MapSizeY.value()

			self.generate(rawmap=False, fixmap=False)
		
		path = os.path.dirname(saveFile[0])
		basename, extension = os.path.splitext(os.path.basename(saveFile[0]))

		from ..generation.saliency import saveImage
		if blend:
			saveImage(self.blend_sal_map.reshape([self.dim()[0], self.dim()[1], 3])[:,:,::-1],
				path+os.sep+basename, extension=extension[1:])
		else:
			saveImage(self.sal_image.reshape([self.dim()[0], self.dim()[1], 3])[:,:,::-1],
				path+os.sep+basename, extension=extension[1:])

		self.MainWindow.setStatusBar("Output: {}{}".format(
			"…" if len(saveFile[0]) > 100 else "",
			saveFile[0][-100:]),
				perma=False)

	def saveSalmapBin (self, file_type="*", compress=False):
		# Save dialog window
		parent = self.MainWindow.container
		qDiag = QtWidgets.QDialog(parent)
		qDiag.setWindowTitle("Save {}binary saliency map".format("compressed " if compress else ""))

		check = {"savefile": False}
		global data; data = ["./salmap.tar.gz" if compress else "./salmap.bin", ""]
		def updateSaveFile():
			global data
			data.clear()
			parent.spawnFileDialog(file_name="./salmap.tar.gz" if compress else "./salmap.bin",
								   file_type=file_type, out=data)

			data = [data[0][0], data[0][1]]
			if len(data[0])>0:

				saveFileLab.setText(data[0] if len(data[0])< 50 else "..."+data[0][-50:])
				check["savefile"] = True
			else:
				saveFileLab.setText("[no file selected]")
				check["savefile"] = False

			acceptBtn.setDisabled(not check["savefile"])
		def closeEvent(event): qDiag.reject(); qDiag.close()
		qDiag.closeEvent = closeEvent

		# Get save file location
		saveFileBtn = QtWidgets.QPushButton("File location")
		saveFileLab = QtWidgets.QLabel(data[0])#"[no output file selected]")
		saveFileLab.setStyleSheet("border: 2px solid rgb(200, 200, 200);border-radius: 5px;")
		acceptBtn = QtWidgets.QPushButton("Save")
		acceptBtn.setDisabled(False)

		dimWid = QtWidgets.QWidget()
		MapSizeX = QtWidgets.QSpinBox()
		MapSizeX.setRange(100, 10000)
		MapSizeX.setSingleStep(100)
		MapSizeX.setValue(self.parent.sceneOption["SM.mapX"])
		MapSizeY = QtWidgets.QSpinBox()
		MapSizeY.setRange(100, 10000)
		MapSizeY.setSingleStep(100)
		MapSizeY.setValue(self.parent.sceneOption["SM.mapY"])
		dimBox = QtWidgets.QHBoxLayout(parent)
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("Map dimensions"))
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("[W:"))
		dimBox.addWidget(MapSizeX)
		dimBox.addWidget(QtWidgets.QLabel(", H:"))
		dimBox.addWidget(MapSizeY)
		dimBox.addWidget(QtWidgets.QLabel("]"))
		dimWid.setLayout(dimBox)

		floatChoiceList = ["16 bits", "32 bits", "64 bits"]
		floatChoiceDict = {"16 bits": np.float16, "32 bits": np.float32, "64 bits": np.float64}
		# Float precision
		flChoice = QtWidgets.QComboBox(parent)
		[flChoice.addItem(el) for el in floatChoiceList]
		flChoice.setCurrentIndex(np.log2(int(self.sal_map.dtype.name[-2:]))-4)

		saveFileBtn.clicked.connect(updateSaveFile)
		acceptBtn.clicked.connect(lambda x: qDiag.accept())

		layout = QtWidgets.QGridLayout(parent)
		layout.addWidget(saveFileBtn, 0, 0, 1, 1)
		layout.addWidget(saveFileLab, 0, 1, 1, 2)
		layout.addWidget(dimWid, 1, 0, 1, 3)
		layout.addWidget(QtWidgets.QLabel("Float precision"), 2, 0, 1, 1)
		layout.addWidget(flChoice, 2, 1, 1, 1)
		layout.addWidget(QtWidgets.QLabel("If you modify the information above, saliency will be recomputed."), 3, 0, 1, 3)
		layout.addWidget(acceptBtn, 4, 0, 1, 3)
		qDiag.setLayout(layout)

		qDiag.setModal(True)
		qDiag.exec()

		saveFile = data
		if qDiag.result() == 0 or len(saveFile) != 2:
			# printWarning("Dialog cancelled.")
			return

		dim = [MapSizeY.value(), MapSizeX.value()]
		precision = floatChoiceDict[floatChoiceList[flChoice.currentIndex()]]

		if dim != self.dim() or self.sal_map.dtype != precision:
			self.parent.sceneOption["SM.mapX"] = MapSizeX.value()
			self.parent.sceneOption["SM.mapY"] = MapSizeY.value()

			self.generate(rawmap=False, fixmap=False, precision=precision)
		
		path = os.path.dirname(saveFile[0])
		basename, extension = os.path.splitext(os.path.basename(saveFile[0]))
		basename = basename[:-4] if compress else basename

		out_path = ""
		if compress:
			pDiag = ModalWindows.progressBar("Compressing and saving binary data", "stop", self.MainWindow)
			pDiag.show()
			from ..generation.saliency import saveBinCompressed
			out_path = saveBinCompressed(self.sal_map.reshape([self.dim()[0], self.dim()[1]])[None, :, :],
				path+os.sep+basename, callback=pDiag.setValue)
			pDiag.close()
		else:
			pDiag = ModalWindows.progressBar("Saving binary data", "stop", self.MainWindow)
			pDiag.show()
			from ..generation.saliency import saveBin
			out_path = saveBin(self.sal_map.reshape([self.dim()[0], self.dim()[1]]),
				path+os.sep+basename, callback=pDiag.setValue)
			pDiag.close()

		self.MainWindow.setStatusBar("Output: {}{}".format(
			"…" if len(out_path) > 100 else "",
			out_path[-100:]),
				perma=False)

	def saveGazeFixlist (self, file_type="*"):
		from .DraggableContentWidget import DragFeatureWidget

		reorder = [0,1, 2,3,4, 9,12]

		# DraggableContentWidget
		parent = self.MainWindow.container
		qDiag = QtWidgets.QDialog(parent)
		qDiag.setWindowTitle("Save fixation list")

		check = {"savefile": True, "features": True}
		global data; data = ["./fix_list.csv", ""]
		def updateSaveFile():
			global data
			data.clear()
			parent.spawnFileDialog(file_name="./fix_list.csv", file_type=file_type, out=data)

			data = [data[0][0], data[0][1]]
			if len(data[0]) > 0:
				if data[0][-4:] not in [".csv", ".txt"]: data[0] += ".csv"

				saveFileLab.setText(data[0] if len(data[0]) < 50 else "..."+data[0][-50:])
				check["savefile"] = True
			else:
				saveFileLab.setText("[no file selected]")
				check["savefile"] = False

			acceptBtn.setDisabled(not sum(check.values()) == 2)

		def closeEvent(event): qDiag.reject(); qDiag.close()
		qDiag.closeEvent = closeEvent

		# Get save file location
		saveFileBtn = QtWidgets.QPushButton("File location")
		saveFileLab = QtWidgets.QLabel(data[0])
		saveFileLab.setStyleSheet("border: 2px solid rgb(200, 200, 200);border-radius: 5px;")
		acceptBtn = QtWidgets.QPushButton("Save")

		InfoLab = QtWidgets.QLabel("Move gaze features that you want to save to the bottom container by drag-and-dropping or right-cliking them.\nMove them around to order them as you wish.")
		InfoLab.setStyleSheet("border: 2px solid rgb(200, 150, 150);border-radius: 5px;background-color: rgb(220, 160, 145);")

		saveFileBtn.clicked.connect(updateSaveFile)
		acceptBtn.clicked.connect(lambda x: qDiag.accept())
		acceptBtn.setDisabled(False)

		# Pick gaze features
		qDiag.DragWin = DragFeatureWidget()
		# Show header string
		headerLab = QtWidgets.QLabel("")
		headerLab.setWordWrap(True)
		headerLab.setStyleSheet("border: 1px solid rgb(200, 200, 200);border-radius: 2px;")
		peekLab = QtWidgets.QLabel("")
		peekLab.setWordWrap(True)
		peekLab.setStyleSheet("border: 1px solid rgb(200, 200, 200);border-radius: 2px;")

		layout = QtWidgets.QGridLayout(parent)
		layout.addWidget(saveFileBtn, 0, 0, 1, 1)
		layout.addWidget(saveFileLab, 0, 1, 1, 2)
		layout.addWidget(InfoLab, 1, 0, 1, 3)
		layout.addWidget(qDiag.DragWin, 2, 0, 1, 3)
		layout.addWidget(headerLab, 3, 0, 1, 3)
		layout.addWidget(peekLab, 4, 0, 1, 3)
		layout.addWidget(acceptBtn, 5, 2, 1, 1)
		qDiag.setLayout(layout)

		def checkFeature(out):
			headerLab.setText("<b>Header</b>: {}{}".format(
								"#" if len(out)>0 else "empty",
								", ".join([a[1] for a in out])
							)
						)

			from ..generation.scanpath import scanpath_fmt
			ifeat = [feat[0] for feat in out]
			fmt = scanpath_fmt.split(",")
			fmt = ", ".join([fmt[i] for i in ifeat])
			fix = self.fix_list[-1, :].copy()

			fix_idx_arr = np.arange(7, fix.shape[0]).tolist()
			[fix_idx_arr.insert(i, ii) for ii, i in enumerate(reorder)]
			fix = fix[fix_idx_arr]

			peekLab.setText("<b>Last line</b>: " + fmt % (*fix[ifeat], ))
			check["features"] = len(out)>0
			acceptBtn.setDisabled(not sum(check.values())==2)

		qDiag.DragWin.changed.connect(checkFeature)
		checkFeature(qDiag.DragWin.getOutFeaturesSorted())

		qDiag.setModal(True)
		qDiag.exec()

		features = qDiag.DragWin.getOutFeaturesSorted()
		saveFile = data

		if qDiag.result() == 0 or len(saveFile) != 2 or len(features) == 0:
			# printWarning("Dialog cancelled.")
			return

		fix_idx_arr = np.arange(7, self.fix_list.shape[1]).tolist()
		[fix_idx_arr.insert(i, ii) for ii, i in enumerate(reorder)]

		from ..generation.scanpath import toFile
		toFile(self.fix_list[:, fix_idx_arr],
			saveFile[0],
			saveArr=[feat[0] for feat in features])

	def saveGazeFixmap (self, file_type="*"):
		# Save dialog window
		parent = self.MainWindow.container
		qDiag = QtWidgets.QDialog(parent)
		qDiag.setWindowTitle("Save fixation map")

		check = {"savefile": False}
		global data; data = ["./fixmap.png", ""]
		def updateSaveFile():
			global data
			data.clear()
			parent.spawnFileDialog(file_name="./fixmap.png",
								   file_type=file_type, out=data)

			data = [data[0][0], data[0][1]]
			if len(data[0]) > 0:
				if data[0][-4:] not in [".png", ".jpg"]: data[0] += ".png"

				saveFileLab.setText(data[0] if len(data[0]) < 50 else "..."+data[0][-50:])
				check["savefile"] = True
			else:
				saveFileLab.setText("[no file selected]")
				check["savefile"] = False

			acceptBtn.setDisabled(not check["savefile"])

		def closeEvent(event): qDiag.reject(); qDiag.close()
		qDiag.closeEvent = closeEvent

		# Get save file location
		saveFileBtn = QtWidgets.QPushButton("File location")
		saveFileLab = QtWidgets.QLabel(data[0])
		saveFileLab.setStyleSheet("border: 2px solid rgb(200, 200, 200);border-radius: 5px;")
		acceptBtn = QtWidgets.QPushButton("Save")
		acceptBtn.setDisabled(False)

		dimWid = QtWidgets.QWidget()
		MapSizeX = QtWidgets.QSpinBox()
		MapSizeX.setRange(100, 10000)
		MapSizeX.setSingleStep(100)
		MapSizeX.setValue(self.parent.sceneOption["SM.mapX"])
		MapSizeY = QtWidgets.QSpinBox()
		MapSizeY.setRange(100, 10000)
		MapSizeY.setSingleStep(100)
		MapSizeY.setValue(self.parent.sceneOption["SM.mapY"])
		dimBox = QtWidgets.QHBoxLayout(parent)
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("Map dimensions"))
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("[W:"))
		dimBox.addWidget(MapSizeX)
		dimBox.addWidget(QtWidgets.QLabel(", H:"))
		dimBox.addWidget(MapSizeY)
		dimBox.addWidget(QtWidgets.QLabel("]"))
		dimWid.setLayout(dimBox)

		saveFileBtn.clicked.connect(updateSaveFile)
		acceptBtn.clicked.connect(lambda x: qDiag.accept())

		layout = QtWidgets.QGridLayout(parent)
		layout.addWidget(saveFileBtn, 0, 0, 1, 1)
		layout.addWidget(saveFileLab, 0, 1, 1, 2)
		layout.addWidget(dimWid, 1, 0, 1, 3)
		layout.addWidget(acceptBtn, 2, 0, 1, 3)
		qDiag.setLayout(layout)

		qDiag.setModal(True)
		qDiag.exec()

		saveFile = data
		if qDiag.result() == 0 or len(saveFile) != 2:
			# printWarning("Dialog cancelled.")
			return

		dim = [MapSizeY.value(), MapSizeX.value()]
		
		path = os.path.dirname(saveFile[0])
		basename, extension = os.path.splitext(os.path.basename(saveFile[0]))

		from ..generation.scanpath import toFixationMap
		from ..generation.saliency import saveImage

		self.fix_map = ((toFixationMap(self.fix_list[:, :2], dim) > .5) * 255).astype(np.uint8)

		saveImage(self.fix_map,	path+os.sep+basename, extension=extension[1:])

		self.MainWindow.setStatusBar("Output: {}{}".format(
			"…" if len(saveFile[0]) > 100 else "",
			saveFile[0][-100:]),
				perma=False)

	def saveScanpathImg (self, file_type="*", blend=False):
		# Save dialog window
		parent = self.MainWindow.container
		qDiag = QtWidgets.QDialog(parent)
		qDiag.setWindowTitle("Save {}scanpath image".format("blended " if blend else ""))

		check = {"savefile": False}
		global data; data = ["./{}scanpath.png".format("blended_" if blend else ""), ""]
		def updateSaveFile():
			global data
			data.clear()
			parent.spawnFileDialog(file_name="./{}scanpath.png".format("blended_" if blend else ""),
								   file_type=file_type, out=data)

			data = [data[0][0], data[0][1]]
			if len(data[0]) > 0:
				if data[0][-4:] not in [".png", ".jpg"]: data[0] += ".png"

				saveFileLab.setText(data[0] if len(data[0]) < 50 else "..."+data[0][-50:])
				check["savefile"] = True
			else:
				saveFileLab.setText("[no file selected]")
				check["savefile"] = False

			acceptBtn.setDisabled(not check["savefile"])

		def closeEvent(event): qDiag.reject(); qDiag.close()
		qDiag.closeEvent = closeEvent

		# Get save file location
		saveFileBtn = QtWidgets.QPushButton("File location")
		saveFileLab = QtWidgets.QLabel(data[0])
		saveFileLab.setStyleSheet("border: 2px solid rgb(200, 200, 200);border-radius: 5px;")
		acceptBtn = QtWidgets.QPushButton("Save")
		acceptBtn.setDisabled(False)

		dimWid = QtWidgets.QWidget()
		MapSizeX = QtWidgets.QSpinBox()
		MapSizeX.setRange(100, 10000)
		MapSizeX.setSingleStep(100)
		MapSizeX.setValue(self.parent.sceneOption["SM.mapX"])
		MapSizeY = QtWidgets.QSpinBox()
		MapSizeY.setRange(100, 10000)
		MapSizeY.setSingleStep(100)
		MapSizeY.setValue(self.parent.sceneOption["SM.mapY"])
		dimBox = QtWidgets.QHBoxLayout(parent)
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("Map dimensions"))
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("[W:"))
		dimBox.addWidget(MapSizeX)
		dimBox.addWidget(QtWidgets.QLabel(", H:"))
		dimBox.addWidget(MapSizeY)
		dimBox.addWidget(QtWidgets.QLabel("]"))
		dimWid.setLayout(dimBox)

		showTextVal = QtWidgets.QCheckBox("Write fixation indices?")
		showTextVal.setChecked(False)

		pointSizeVal = QtWidgets.QSpinBox()
		pointSizeVal.setRange(1, 500)
		pointSizeVal.setValue(5)
		pointSizeVal.setSuffix("px")

		global colour
		colour = QtGui.QColor(127, 200, 127)
		ptsColour = QtWidgets.QPushButton("Point colour")
		ptsColour.setPalette(QtGui.QPalette(colour))

		def colourCB(newcol):
			global colour
			colour = newcol
			ptsColour.setPalette(QtGui.QPalette(colour))

		ptsColour.clicked.connect(lambda x: self.parent.spawnColorDialog(colour=colour,
				callback=colourCB, modal=True,
				colOptions=QtWidgets.QColorDialog.DontUseNativeDialog))

		saveFileBtn.clicked.connect(updateSaveFile)
		acceptBtn.clicked.connect(lambda x: qDiag.accept())

		layout = QtWidgets.QGridLayout(parent)
		layout.addWidget(saveFileBtn, 0, 0, 1, 2)
		layout.addWidget(saveFileLab, 0, 2, 1, 2)
		layout.addWidget(dimWid, 1, 0, 1, 3)
		layout.addWidget(ptsColour, 2, 0, 1, 2)
		layout.addWidget(QtWidgets.QLabel("Point radius"), 2, 2, 1, 1, QtCore.Qt.AlignRight)
		layout.addWidget(pointSizeVal, 2, 3, 1, 1)
		layout.addWidget(showTextVal, 3, 0, 1, 2)
		layout.addWidget(acceptBtn, 4, 0, 1, 4)
		qDiag.setLayout(layout)

		qDiag.setModal(True)
		qDiag.exec()

		saveFile = data
		if qDiag.result() == 0 or len(saveFile) != 2:
			# printWarning("Dialog cancelled.")
			return

		dim = [MapSizeY.value(), MapSizeX.value()]
		
		path = os.path.dirname(saveFile[0])
		basename, extension = os.path.splitext(os.path.basename(saveFile[0]))

		from ..generation.scanpath import toImage

		if blend: blendDim = self.parent.Equirect.backgroundImage.shape

		toImage(self.fix_list[:, [0, 1]], dim, path+os.sep+basename,
			text=showTextVal.isChecked(),
			ptsSize=pointSizeVal.value(),
			extension=extension[1:],
			blend=self.parent.Equirect.backgroundImage.copy().reshape([blendDim[1], blendDim[0], blendDim[2]])[:, :, ::-1] if blend else None)

		self.MainWindow.setStatusBar("Output: {}{}".format(
			"…" if len(saveFile[0]) > 100 else "",
			saveFile[0][-100:]),
				perma=False)
