#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

try:
	from PyQt5 import QtGui
	from PyQt5 import QtWidgets
	from PyQt5 import QtCore
except:
	print("QT 5 wrapper for python not found")
	print("\trun `pip install pyqt5`")
	exit()

import numpy as np

from ..utils.misc import *
from .GazeData import GazeData
from .settings import sceneGLOptions

class WinContent(QtWidgets.QWidget):
	def __init__(self, parent):
		super(WinContent, self).__init__(parent)
		from .GLwidget import EquirectView

		self.parent = parent

		# Default settings
		self.sceneOption = sceneGLOptions()

		# Helper functions to process and transform data
		self.gazeData = GazeData(self)

		# SettingInterface elements
		self.SettingInterface = SettingInterface(self)
		self.SettingInterfaceBox = CollapsibleBox(None, title="Settings")
		self.SettingInterfaceBox.setContentLayout(self.SettingInterface.layout())

		# DisplayInterface elements
		self.DisplayInterface = DisplayInterface(self)
		self.DisplayInterfaceBox = CollapsibleBox(None, title="Display")
		self.DisplayInterfaceBox.setContentLayout(self.DisplayInterface.layout())

		# # SaveInterface elements
		self.SaveInterface = SaveInterface(self)
		self.SaveInterfaceBox = CollapsibleBox(None, title="Export")
		self.SaveInterfaceBox.setContentLayout(self.SaveInterface.layout())

		# OpenGL window
		self.Equirect = EquirectView(self, openGLVer=self.parent.args.opengl)

		mainLayout = QtWidgets.QGridLayout(self)
		mainLayout.addWidget(self.DisplayInterfaceBox, 0, 0, 1, 1, QtCore.Qt.AlignTop)
		mainLayout.addWidget(self.SaveInterfaceBox,    0, 1, 1, 1, QtCore.Qt.AlignTop)
		mainLayout.addWidget(self.SettingInterfaceBox, 0, 2, 1, 1, QtCore.Qt.AlignTop)
		mainLayout.addWidget(self.Equirect, 		   1, 0, 1, 3)

		mainLayout.setColumnStretch(0, 20)
		mainLayout.setColumnStretch(1, 25)
		mainLayout.setColumnStretch(2, 55)

		mainLayout.setRowStretch(0, 1)
		mainLayout.setRowStretch(1, 99)
		
		self.setLayout(mainLayout)

		self.setDispActions()
		self.setExportActions()
		self.setOptActions()
		self.doLast()
	
		if self.parent.args.show_settings:
			self.sceneOption.showDoc()
			exit()

	def applySettings(self, settings):
		# set GUI settings from dict (settings)
		options = self.sceneOption
		for key, val in settings.items():
			if key in options.getKeys():
				old = options[key]
				try:
					if key in ["GP.colour", "FP.colour"]:
						if val in QtGui.QColor.colorNames():
							import numpy as np
							options[key] = np.array(QtGui.QColor(val).getRgbF())
						elif type(val) == str and len(val.split(",")) == 4:
							options[key] = [float(el.strip()) for el in val.split(",")]
					else:
						# try to coerce passed value to be the same type as the preset
						tocast = type(old)
						if tocast == bool:
							options[key] == val == "True"
						else:
							options[key] = tocast(val)

					printNeutral("[scene options] new value for \"{}\" is {} (old: {})".format(
							key, options[key], old),
						verbose=2)
				except:
					printWarning("[scene options] Could not set value \"{}\" ({}) for option \"{}\" ({}).".format(val, type(val).__name__, key, type(old).__name__), verbose=0)
			else:
				printWarning("[scene options] \"{}\" is not a valid option key".format(key), verbose=0)

	def doLast(self):
		self.SettingInterface.currParseLab.setText("{}".format(self.sceneOption["GP.parse"]))

		# Set param passed by CLI in settings
		args = self.parent.args
		options = self.sceneOption

		# Parse fixation parser settings
		parsers = {k: v for k, v in
						{"IVT": args.IVT, "IHMM": args.IHMM, "ICT": args.ICT}.items() 
					if v is not None}
		if len(parsers) == 1:
			parserName, params = parsers.popitem()

			options["GP.parse"] = "I-" + parserName[1:]

			self.loadArgs(params, "GP.{}".format(parserName))

		# Parse filtering settings
		if args.filter is not None:
			options["GP.preproc.filter"] = args.filter
			if args.filter_opt is not None:

				self.loadArgs(args.filter_opt, "GP.preproc.filter.{}".format(args.filter))

		# Head trajectory
		options["GP.preproc.headwin"] = args.headwindow
		options["GP.preproc.resample"] = args.resample

		# Data tracking
		if args.tracking == "H":
			options["GP.tracking"] = 3 # Process head trajectory
		else:
			options["GP.tracking"] = list("LRB").index(args.eye)

	def loadArgs(self, args, basename):
		options = self.sceneOption

		for arg in args:
			cut = arg.split("=")

			if len(cut) != 2:
				printWarning("[scene options] \"{}\" is not a valid setting value. It must be \"name=value\"".format(cut), verbose=0)
				continue

			name, value = cut

			key = "{}.{}".format(basename, name)
			old = options[key]

			try:
				# try to coerce passed value to be the type as the preset
				options[key] = type(old)(value)

				printNeutral("[scene options] new value for \"{}\" is {} (old: {})".format(
						key, options[key], old),
					verbose=2)
			except:
				printWarning("[scene options] Could not set value \"{}\" ({}) for option \"{}\" ({}).".format(value, type(value).__name__, key, type(old).__name__), verbose=0)

	def setDispActions(self):
		gdata = self.gazeData
		disp = self.DisplayInterface

		disp.ResetBtn.clicked.connect(lambda x: gdata.reset())

	def setExportActions(self):
		gdata = self.gazeData
		save = self.SaveInterface

		# Salmap
		save.SMimg.clicked.connect(lambda x: gdata.saveSalmapImg("PNG image (*.png);; JPEG image (*.jpg)"))
		save.SMbimg.clicked.connect(lambda x: gdata.saveSalmapImg("PNG image (*.png);; JPEG image (*.jpg)",
			blend=True))
		save.SMbin.clicked.connect(lambda x: gdata.saveSalmapBin("Binary data (*.bin)"))
		save.SMcbin.clicked.connect(lambda x: gdata.saveSalmapBin("Compressed binary data (*.tar.gz)",
			compress=True))
		# Gaze data
		save.GDfl.clicked.connect(lambda x: gdata.saveGazeFixlist("Comma-separated values file (*.csv)"))
		save.GDfm.clicked.connect(lambda x: gdata.saveGazeFixmap("PNG image (*.png);; JPEG image (*.jpg)"))
		# Scanpath
		save.SPimg.clicked.connect(lambda x: gdata.saveScanpathImg("PNG image (*.png);; JPEG image (*.jpg)"))
		save.SPbimg.clicked.connect(lambda x: gdata.saveScanpathImg("PNG image (*.png);; JPEG image (*.jpg)",
			blend=True))

	def setOptActions(self):
		equi = self.Equirect
		gdata = self.gazeData
		options = self.sceneOption
		setting = self.SettingInterface

		# Binding between settings and openGL scene
		# 	Saliency
		global regenType; regenType = None
		def setRegenBtn(type_):
			global regenType
			setting.regenBtn.setDisabled(False)
			if regenType == "parser": return
			if regenType == "saliency" and type_ == "image": return
			regenType = type_

		#	Saliency
		options.setSetting("SM.Gauss", 2.,
			lambda: setRegenBtn("saliency"),
			setting.SMgaussVal.setValue,
			doc="Sigma of the gaussians that is drawn as saliency to model visual perception in the scene.")
		options.setSetting("SM.cm", "coolwarm",
			lambda: gdata.generate(saliency=False, rawmap=False, fixmap=False),
			setting.SMcolormapChoice.setCurrentText,
			doc="Colour map for the saliency image (among colour maps provided by matplotlib).")
		options.setSetting("SM.rev", False,
			lambda: gdata.generate(saliency=False, rawmap=False, fixmap=False),
			setting.SMreverseVal.setCheckState,
			doc="If true, will reverse the colour map.")
		options.setSetting("SM.fromfix", 1,
			lambda: setRegenBtn("saliency"),
			setting.SMdataChoice.setCurrentIndex,
			doc="If \"1\", saliency will be computed from fixations; if \"0\", from raw gaze data if available.")
		options.setSetting("SM.mapX", 2000,
			lambda: setRegenBtn("saliency"),
			setting.MapSizeX.setValue,
			doc="Saliency resolution (horizontal dimension).")
		options.setSetting("SM.mapY", 1000,
			lambda: setRegenBtn("saliency"),
			setting.MapSizeY.setValue,
			doc="Saliency resolution (vertical dimension).")
		# 	Viewport spans
		options.setSetting("VP.as-x", 90.,
			lambda: (equi.set_Viewport(), equi.update()),
			setting.VdegHorizVal.setValue,
			doc="Horizontal viewport span in degrees.")
		options.setSetting("VP.as-y", 90.,
			lambda: (equi.set_Viewport(), equi.update()),
			setting.VdegVertVal.setValue,
			doc="Vertical viewport span in degrees.")
		options.setSetting("VP.mult", 1.,
			lambda: (equi.set_Viewport(), equi.update()),
			setting.VpixSizeVal.setValue,
			doc="Multiplier for the size of the viewport displayed in the lower right part of the scene.")
		# 	Gaze points
		options.setSetting("GP.size", 5.,
			lambda: (equi.set_GazePointSize(), equi.update()),
			setting.GsizeVal.setValue,
			doc="Size of gaze points in pixels.")
		options.setSetting("GP.colour", np.array([110, 230, 180, 10])/255,
			lambda: (equi.set_GazePointColour(), equi.update()),
			lambda val: self.ColorDialogCallback(val, "GP.colour"),
			doc="Gaze points colour, should be a colour name, full list: `QtGui.QColor.colorNames()`.")
		options.setSetting("GP.type", "Auto",
			lambda: (equi.set_GazePointColour(), equi.update()),
			lambda val: [btn.setChecked(True) for btn in setting.Gcolchoice.buttons() if btn.text().lower() == val.lower()],
			doc="How to draw gaze points. Solid: use colour as is; Gradient: add a linear gradient to the colour; Auto: will use sequencing data to draw saccades and fixations in different colours.")
		options.setSetting("GP.tracking", 1,
			lambda: setRegenBtn("parser"),
			setting.GPtrackChoice.setCurrentIndex,
			doc="Type of data to use: \"HE\" for the combined gaze data (Head + Eye) or \"H\" for head only to output a head trajectory.")
		# 	Preproc
		options.setSetting("GP.preproc.headwin", 100,
			lambda: setRegenBtn("parser"),
			doc="Size of the window used when calculating head trajectory, head positions are extracted with a moving window of this size.")
		options.setSetting("GP.preproc.resample", 0,
			lambda: setRegenBtn("parser"),
			doc="Resampling rate in Hz, 0 means no resampling.")
		options.setSetting("GP.preproc.filter", "No filtering",
			lambda: setRegenBtn("parser"),
			doc="Which filter to use to smooth velocity data (\"gauss\" for Gaussian or \"savgol\" Savitzky-Golay). No filtering by default.")
		options.setSetting("GP.preproc.filter.gauss.sigma", 4,
			lambda: setRegenBtn("parser"),
			doc="Size of the Gaussian sigma to use to filter velocity data.")
		options.setSetting("GP.preproc.filter.savgol.win", 9,
			lambda: setRegenBtn("parser"),
			doc="With the Savitzky-Golay filter: size of the window.")
		options.setSetting("GP.preproc.filter.savgol.poly", 2,
			lambda: setRegenBtn("parser"),
			doc="With the Savitzky-Golay filter: polynomial order.")
		# 	Parser
		options.setSetting("GP.parse", "I-VT",
			lambda: setRegenBtn("parser"),
			self.SettingInterface.currParseLab.setText,
			doc="Fixation parsing algorithm to use. Choices are: \"I-VT\" (velocity), \"I-CT\" (clustering), \"I-HMM\" (Hidden Markov Model), \"I-DT\" (dispersion, not implemented).")
		options.setSetting("GP.IVT.threshold", 120,
			lambda: setRegenBtn("parser"),
			doc="Samples below this threshold are identified as parts of fixations.")
		options.setSetting("GP.IHMM.nStates", 2,
			lambda: setRegenBtn("parser"),
			doc="The number of hidden states in the HMM-based parsing algorithm.")
		options.setSetting("GP.ICT.eps", .005,
			lambda: setRegenBtn("parser"),
			doc="EPS parameter of DBSCAN.")
		options.setSetting("GP.ICT.minpts", 3,
			lambda: setRegenBtn("parser"),
			doc="Minimum points number parameter of DBSCAN.")
		# 	Fixation points
		options.setSetting("FP.size", 5.,
			lambda: (equi.set_GazePointSize(), equi.update()),
			setting.FsizeVal.setValue,
			doc="Size of fixation points in pixels.")
		options.setSetting("FP.colour", np.array([255, 127, 255, 255])/255,
			lambda: (equi.set_GazePointColour(), equi.update()),
			lambda val: self.ColorDialogCallback(val, "FP.colour"),
			doc="Fixation points colour, should be a colour name, full list: `QtGui.QColor.colorNames()`.")
		options.setSetting("FP.type", "Gradient",
			lambda: (equi.set_GazePointColour(), equi.update()),
			lambda val: [btn.setChecked(True) for btn in setting.Fcolchoice.buttons() if btn.text().lower() == val.lower()],
			doc="How to draw fixation points. Solid: use colour as is; Gradient: add a linear gradient to the colour.")
		# 	Toggle
		options.setSetting("Tog.viewp", 1,
			equi.update,
			setting.DvpVal.setChecked,
			doc="If True, the viewport view is rendered.")
		options.setSetting("Tog.sph", 1,
			equi.update,
			setting.DspVal.setChecked,
			doc="If True, the sphere projection is rendered.")
		options.setSetting("Tog.idle", 1,
			equi.update,
			setting.DidleVal.setChecked,
			doc="If True, the viewport animation is played after 2 seconds of inactivity.")
		options.setSetting("Tog.loop", 1,
			lambda: None,
			setting.DLoopVal.setChecked,
			doc="If True, the mouse position will loop horizontally when this window has focus.")

		# Setting change effects
		#	Saliency
		setting.SMgaussVal.valueChanged.connect(options.settings["SM.Gauss"])
		setting.SMcolormapChoice.currentTextChanged.connect(options.settings["SM.cm"])
		setting.SMreverseVal.stateChanged.connect(options.settings["SM.rev"])
		setting.SMdataChoice.currentIndexChanged.connect(options.settings["SM.fromfix"])
		setting.MapSizeX.valueChanged.connect(options.settings["SM.mapX"])
		setting.MapSizeY.valueChanged.connect(options.settings["SM.mapY"])
		#	viewport
		setting.VdegHorizVal.valueChanged.connect(options.settings["VP.as-x"])
		setting.VdegVertVal.valueChanged.connect(options.settings["VP.as-y"])
		setting.VpixSizeVal.valueChanged.connect(options.settings["VP.mult"])
		# 	Gaze points
		setting.GsizeVal.valueChanged.connect(options.settings["GP.size"])
		setting.Gcolour.clicked.connect(lambda x: self.spawnColorDialog("GP.colour", modal=True))
		setting.Gcolchoice.buttonClicked.connect(options.settings["GP.type"])
		setting.GPtrackChoice.currentIndexChanged.connect(options.settings["GP.tracking"])
		# 	Preproc
		#		Set in modal window
		# 	Parser
		#		Set in modal window
		# 	Fixation points
		setting.FsizeVal.valueChanged.connect(options.settings["FP.size"])
		setting.Fcolour.clicked.connect(lambda x: self.spawnColorDialog("FP.colour", modal=True))
		setting.Fcolchoice.buttonClicked.connect(options.settings["FP.type"])
		#	Toggle
		setting.DvpVal.stateChanged.connect(options.settings["Tog.viewp"])
		setting.DspVal.stateChanged.connect(options.settings["Tog.sph"])
		setting.DidleVal.stateChanged.connect(options.settings["Tog.idle"])
		setting.DLoopVal.stateChanged.connect(options.settings["Tog.loop"])

		#	Regenerate everything
		def regen():
			global regenType
			if regenType != "images":
				gdata.generate(parse=True)
			else:
				gdata.generate(saliency=False, rawmap=False, fixmap=False)
			regenType = None
			setting.regenBtn.setDisabled(True)

		setting.regenBtn.clicked.connect(regen)

		#	Write settings to file
		def saveSettings():
			data = []
			self.spawnFileDialog(file_name="settings.set", file_type="*.set", out=data)
			data = data[0][0]

			if data != "":
				if options.saveToFile(data):
					self.parent.setStatusBar("Settings saved to {}".format(data))

		setting.saveSettBtn.clicked.connect(saveSettings)

	def spawnFileDialog(self, file_name=None, title=None, file_type="*", out=None):
		fileDiag = QtWidgets.QFileDialog.getSaveFileName(
			self.parent,
			title,
			file_name,
			filter=file_type)

		if out is not None:
			out.append(fileDiag)

	def spawnColorDialog(self,
		target=None, colour=None,
		callback=None, modal=False,
		colOptions=QtWidgets.QColorDialog.DontUseNativeDialog | QtWidgets.QColorDialog.NoButtons | QtWidgets.QColorDialog.ShowAlphaChannel):
		options = self.sceneOption
		setting = self.SettingInterface

		colorDiag = QtWidgets.QColorDialog(self.parent)
		colorDiag.setOptions(colOptions)

		if target is not None:
			colorDiag.setCurrentColor(QtGui.QColor.fromRgbF(*options[target]))
			colorDiag.currentColorChanged.connect(lambda x: self.ColorDialogCallback(x, target))
		else:
			if colour is not None:
				colorDiag.setCurrentColor(colour)
			if callback is not None and callable(callback):
				colorDiag.currentColorChanged.connect(lambda x: callback(x))

		colorDiag.setModal(modal)
		colorDiag.show()

		return colorDiag

	def ColorDialogCallback(self, color, target):

		options = self.sceneOption
		setting = self.SettingInterface

		if type(color) == str:
			if color in QtGui.QColor.colorNames():
				color = QtGui.QColor(color)
			else:
				printWarning("Cannot interpret \"{}\" as a color name.".format(color))
				return
		elif type(color) in [np.ndarray, tuple, list]:
			color = QtGui.QColor.fromRgbF(*color)

		if type(color) != QtGui.QColor or not color.isValid():
			return
		
		options.changeSetting(target, np.array(color.getRgbF()))

		buttonLab = setting.Gcolour if target[0] == "G" else setting.Fcolour
		buttonLab.setText(color.name())
		buttonLab.setPalette(QtGui.QPalette(QtGui.QColor(color)))

class DisplayInterface(QtWidgets.QWidget):
	def __init__(self, parent):
		super(DisplayInterface, self).__init__(parent)
		self.parent = parent

		self.BGLab = QtWidgets.QLabel("Background")
		self.BGImg = ImageButton(self)
		self.GPLab = QtWidgets.QLabel("Raw gaze")
		self.GPImg = ImageButton(self)
		self.SPLab = QtWidgets.QLabel("Fixation map")
		self.SPImg = ImageButton(self)
		self.SMLab = QtWidgets.QLabel("Saliency map")
		self.SMImg = ImageButton(self)

		self.BGLab.setAlignment(QtCore.Qt.AlignHCenter)
		self.GPLab.setAlignment(QtCore.Qt.AlignHCenter)
		self.SPLab.setAlignment(QtCore.Qt.AlignHCenter)
		self.SMLab.setAlignment(QtCore.Qt.AlignHCenter)

		self.BGLab.setStyleSheet("font-weight: bold;")
		self.GPLab.setStyleSheet("font-weight: bold;")
		self.SPLab.setStyleSheet("font-weight: bold;")
		self.SMLab.setStyleSheet("font-weight: bold;")

		self.ResetBtn = QtWidgets.QPushButton(" Reset")
		self.ResetBtn.setIcon(QtGui.QIcon.fromTheme("document-revert"))
		self.ResetBtn.setStyleSheet("text-align:center;")

		mainLayout = QtWidgets.QGridLayout(self)
		mainLayout.addWidget(self.BGLab, 0, 0, 1, 1)
		mainLayout.addWidget(self.BGImg, 1, 0, 1, 1)

		mainLayout.addWidget(self.SMLab, 2, 0, 1, 1)
		mainLayout.addWidget(self.SMImg, 3, 0, 1, 1)

		mainLayout.addWidget(self.GPLab, 0, 1, 1, 1)
		mainLayout.addWidget(self.GPImg, 1, 1, 1, 1)

		mainLayout.addWidget(self.SPLab, 2, 1, 1, 1)
		mainLayout.addWidget(self.SPImg, 3, 1, 1, 1)

		mainLayout.addWidget(self.ResetBtn, 4, 0, 1, 2)

		# mainLayout.setAlignment(QtCore.Qt.AlignHCenter)
		self.setLayout(mainLayout)
		mainLayout.setContentsMargins(0, 0, 0, 0)

class SaveInterface(QtWidgets.QWidget):
	def __init__(self, parent):
		super(SaveInterface, self).__init__(parent)
		self.parent = parent

		# Salmap
		self.SMimg = QtWidgets.QPushButton(" Image")
		self.SMbimg = QtWidgets.QPushButton(" Blended image")
		self.SMbin = QtWidgets.QPushButton(" Binary")
		self.SMcbin = QtWidgets.QPushButton(" Compressed binary")
		# Gaze data
		# self.GDrg = QtWidgets.QPushButton(" Raw gaze")
		self.GDfl = QtWidgets.QPushButton(" Fixation list")
		self.GDfm = QtWidgets.QPushButton(" Fixation map")
		# Scanpath
		self.SPimg = QtWidgets.QPushButton(" Image")
		self.SPbimg = QtWidgets.QPushButton(" Blended image")

		icon = QtGui.QIcon.fromTheme("document-save")

		self.salmapGBox = QtWidgets.QGroupBox("Saliency map")
		boxLayout = QtWidgets.QGridLayout(self)
		boxLayout.addWidget(self.SMimg,  0, 0, 1, 1)
		boxLayout.addWidget(self.SMbimg, 1, 0, 1, 1)
		boxLayout.addWidget(self.SMbin,  2, 0, 1, 1)
		boxLayout.addWidget(self.SMcbin, 3, 0, 1, 1)
		self.salmapGBox.setLayout(boxLayout)

		self.gazedGBox = QtWidgets.QGroupBox("Gaze data")
		boxLayout = QtWidgets.QGridLayout(self)
		# boxLayout.addWidget(self.GDrg, 0, 0, 1, 1)
		boxLayout.addWidget(self.GDfl, 0, 0, 1, 1)
		boxLayout.addWidget(self.GDfm, 1, 0, 1, 1)
		self.gazedGBox.setLayout(boxLayout)

		self.scanpathGBox = QtWidgets.QGroupBox("Scanpath")
		boxLayout = QtWidgets.QGridLayout(self)
		boxLayout.addWidget(self.SPimg, 0, 0, 1, 1)
		boxLayout.addWidget(self.SPbimg, 1, 0, 1, 1)
		self.scanpathGBox.setLayout(boxLayout)


		mainLayout = QtWidgets.QGridLayout(self)
		mainLayout.addWidget(self.salmapGBox,   0, 0, 2, 1)
		mainLayout.addWidget(self.gazedGBox,    0, 1, 1, 1)
		mainLayout.addWidget(self.scanpathGBox, 1, 1, 2, 1)

		self.salmapGBox.setAlignment(QtCore.Qt.AlignRight)
		self.gazedGBox.setAlignment(QtCore.Qt.AlignRight)
		self.scanpathGBox.setAlignment(QtCore.Qt.AlignRight)

		self.salmapGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.gazedGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.scanpathGBox.setStyleSheet("QGroupBox{font-weight: bold;}")

		# mainLayout.setAlignment(QtCore.Qt.AlignHCenter)
		self.setLayout(mainLayout)
		mainLayout.setContentsMargins(0, 0, 0, 0)

		self.SetSaveButtonsDisabled(True)

	def SetSaveButtonsDisabled(self, status=False):
		[el.setDisabled(status) for el in [
			self.SMimg, self.SMbimg, self.SMbin, self.SMcbin,
			self.GDfl, self.GDfm,
			self.SPimg, self.SPbimg]
			]

class SettingInterface(QtWidgets.QWidget):
	def __init__(self, parent):
		super(SettingInterface, self).__init__(parent)
		self.parent = parent

		self.salmapGBox = QtWidgets.QGroupBox("Saliency")
		self.SMgaussVal = QtWidgets.QDoubleSpinBox()
		self.SMgaussVal.setRange(.5, 50.)
		self.SMgaussVal.setSingleStep(.1)
		self.SMgaussVal.setValue(2.)
		self.SMgaussVal.setSuffix("°")

		self.SMcolormapChoice = QtWidgets.QComboBox(self)
		self.SMdataChoice = QtWidgets.QComboBox(self)
		self.SMreverseVal = QtWidgets.QCheckBox("Reverse")

		from matplotlib.pyplot import colormaps, get_cmap
		imap = 0
		for cmap in colormaps():
			if cmap[-2:] == "_r": continue

			# Set item name
			self.SMcolormapChoice.addItem(cmap)
			# Create and set item icon
			cmap = get_cmap(cmap)
			icon = QtGui.QIcon(
					QtGui.QPixmap(
						QtGui.QImage(
							   np.repeat(cmap(np.linspace(0, 1, 128))[None, :, :] * 255, 64, axis=0).astype(np.uint8),
							   128,
							   64,
							   QtGui.QImage.Format_RGBA8888
							   )
						)
					)
			self.SMcolormapChoice.setItemIcon(imap, icon)
			imap +=1
		# self.SMcolormapChoice.setCurrentIndex(45)

		self.SMdataChoice.addItem("Raw gaze samples")
		self.SMdataChoice.addItem("Fixation samples")
		self.SMdataChoice.setCurrentIndex(1)

		dimWid = QtWidgets.QWidget()
		self.MapSizeX = QtWidgets.QSpinBox()
		self.MapSizeX.setRange(100, 10000)
		self.MapSizeX.setSingleStep(100)
		self.MapSizeX.setValue(2000)
		self.MapSizeX.setSuffix("px")
		self.MapSizeY = QtWidgets.QSpinBox()
		self.MapSizeY.setRange(100, 10000)
		self.MapSizeY.setSingleStep(100)
		self.MapSizeY.setValue(1000)
		self.MapSizeY.setSuffix("px")
		dimBox = QtWidgets.QHBoxLayout(self)
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("Map dimensions"))
		dimBox.addStretch()
		dimBox.addWidget(QtWidgets.QLabel("[W:"))
		dimBox.addWidget(self.MapSizeX)
		dimBox.addWidget(QtWidgets.QLabel(", H:"))
		dimBox.addWidget(self.MapSizeY)
		dimBox.addWidget(QtWidgets.QLabel("]"))
		dimWid.setLayout(dimBox)

		boxLayout = QtWidgets.QGridLayout(self)
		boxLayout.addWidget(QtWidgets.QLabel("Gaussian Sigma"), 0, 0, 1, 2, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.SMgaussVal, 0, 2, 1, 1)
		boxLayout.addWidget(QtWidgets.QLabel("Data"), 1, 0, 1, 2, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.SMdataChoice, 1, 2, 1, 1)
		boxLayout.addWidget(dimWid, 2, 0, 1, 3)
		boxLayout.addWidget(QtWidgets.QLabel("Color map"), 3, 0, 1, 1, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.SMcolormapChoice, 3, 1, 1, 1)
		boxLayout.addWidget(self.SMreverseVal, 3, 2, 1, 1)
		self.salmapGBox.setLayout(boxLayout)
		boxLayout.setContentsMargins(0, 0, 0, 0)

		self.gazeptsGBox = QtWidgets.QGroupBox("Gaze points")
		self.GsizeVal = QtWidgets.QSpinBox()
		self.GsizeVal.setRange(1, 200)
		self.GsizeVal.setSingleStep(1)
		self.GsizeVal.setValue(5)

		self.Gcolour = QtWidgets.QPushButton("Colour")
		colour = QtGui.QColor(110, 230, 180, 10)
		self.Gcolour.setPalette(QtGui.QPalette(colour))
		self.Gcolour.setText(colour.name())
		self.Gcolchoice = QtWidgets.QButtonGroup()
		GcolGradVal = QtWidgets.QRadioButton("Gradient")
		GcolSolVal  = QtWidgets.QRadioButton("Solid")
		GcolAutoVaL = QtWidgets.QRadioButton("Label S/F")
		[self.Gcolchoice.addButton(el) for el in [GcolGradVal, GcolSolVal, GcolAutoVaL]]
		GcolAutoVaL.setChecked(True)

		self.GshowPreprocBtn = QtWidgets.QPushButton("Preprocessing")

		self.GshowParseBtn = QtWidgets.QPushButton("Gaze parsing")
		self.currParseLab = QtWidgets.QLabel("Nothing")

		def showPreprocOpt():
			qDiag = QtWidgets.QDialog(parent)
			diagLayout = QtWidgets.QGridLayout()

			qDiag.setWindowTitle("Preprocessing settings")
			qDiag.setLayout(diagLayout)
			
			# Head trajectory
			headBox = QtWidgets.QGroupBox("Head trajectory")
			boxLayout = QtWidgets.QGridLayout()

			Head_winsize = QtWidgets.QDoubleSpinBox()
			Head_winsize.setRange(1, 500)
			Head_winsize.setSingleStep(.1)
			Head_winsize.setValue(parent.sceneOption["GP.preproc.headwin"])
			Head_winsize.setSuffix(" msec")
			Head_winsize.valueChanged.connect(parent.sceneOption.settings["GP.preproc.headwin"])

			boxLayout.addWidget(QtWidgets.QLabel("Temporal window size"), 1,0,1,1)
			boxLayout.addWidget(Head_winsize, 1,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("This is used when processing \"head trajectories\". A head trajectory\nposition is calculated with a moving and non-overlapping temporal window."), 2,0,1,2)
			headBox.setLayout(boxLayout)
			
			# Resampling by interpolation
			samplBox = QtWidgets.QGroupBox("Resampling")
			boxLayout = QtWidgets.QGridLayout()

			Sampl_rate = QtWidgets.QSpinBox()
			Sampl_rate.setRange(0, 2000)
			Sampl_rate.setSingleStep(.1)
			Sampl_rate.setValue(parent.sceneOption["GP.preproc.resample"])
			Sampl_rate.setSuffix(" Hz")
			Sampl_rate.valueChanged.connect(parent.sceneOption.settings["GP.preproc.resample"])

			boxLayout.addWidget(QtWidgets.QLabel("Resampling rate"), 1,0,1,1)
			boxLayout.addWidget(Sampl_rate, 1,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("A value of 0 means no resampling"), 2,0,1,2)
			samplBox.setLayout(boxLayout)
			
			# Filtering
			filterBox = QtWidgets.QGroupBox("Smoothing (filtering) of velocity signal")
			boxLayout = QtWidgets.QGridLayout()

			gaussInfoLab = QtWidgets.QLabel("(<a href=\"https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html\">more information<a/>)")
			gaussInfoLab.setOpenExternalLinks(True)

			SavGolInfoLab = QtWidgets.QLabel("(<a href=\"https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html\">more information<a/>)")
			SavGolInfoLab.setOpenExternalLinks(True)

			Gauss_win = QtWidgets.QDoubleSpinBox()
			Gauss_win.setRange(.01, 500)
			Gauss_win.setSingleStep(.1)
			Gauss_win.setValue(parent.sceneOption["GP.preproc.filter.gauss.sigma"])
			Gauss_win.setSuffix(" samples")
			Gauss_win.valueChanged.connect(parent.sceneOption.settings["GP.preproc.filter.gauss.sigma"])

			SavGol_win = QtWidgets.QSpinBox()
			SavGol_win.setRange(1, 500)
			SavGol_win.setSingleStep(1)
			SavGol_win.setValue(parent.sceneOption["GP.preproc.filter.savgol.win"])
			SavGol_win.setSuffix(" samples")
			SavGol_win.valueChanged.connect(parent.sceneOption.settings["GP.preproc.filter.savgol.win"])

			SavGol_poly = QtWidgets.QSpinBox()
			SavGol_poly.setRange(1, 500)
			SavGol_poly.setSingleStep(1)
			SavGol_poly.setValue(parent.sceneOption["GP.preproc.filter.savgol.poly"])
			SavGol_poly.valueChanged.connect(parent.sceneOption.settings["GP.preproc.filter.savgol.poly"])

			filterChoiceGp = QtWidgets.QButtonGroup()
			fitlerNoneVal = QtWidgets.QRadioButton("No filtering")
			fitlerGaussVal = QtWidgets.QRadioButton("Gaussian filter")
			fitlerSavGolVal = QtWidgets.QRadioButton("Savitzky-Golay filter")
			filterBtn = [fitlerNoneVal, fitlerGaussVal, fitlerSavGolVal]
			filterVal = [el.text() for el in filterBtn]
			[filterChoiceGp.addButton(el) for el in filterBtn]
			filterChoiceGp.buttonClicked.connect(parent.sceneOption.settings["GP.preproc.filter"])

			filterSelected = ["n", "g", "s"].index(parent.sceneOption["GP.preproc.filter"][0].lower())
			if filterSelected == -1:
				filterSelected = 0
			filterBtn[filterSelected].setChecked(True)

			closeBtn = QtWidgets.QPushButton("Close")
			closeBtn.clicked.connect(lambda: qDiag.close())

			boxLayout.addWidget(fitlerNoneVal, 0,0,1,2)
			boxLayout.addWidget(QHLine(), 1,0,1,2)
			boxLayout.addWidget(fitlerGaussVal, 2,0,1,1)
			boxLayout.addWidget(gaussInfoLab, 2,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("Standard deviation"), 3,0,1,1)
			boxLayout.addWidget(Gauss_win, 3,1,1,1)
			boxLayout.addWidget(QHLine(), 4,0,1,3)
			boxLayout.addWidget(fitlerSavGolVal, 5,0,1,1)
			boxLayout.addWidget(SavGolInfoLab, 5,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("Window length"), 6,0,1,1)
			boxLayout.addWidget(SavGol_win, 6,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("Polynomial order"), 7,0,1,1)
			boxLayout.addWidget(SavGol_poly, 7,1,1,1)
			boxLayout.addWidget(QtWidgets.QLabel("Polynomial order must be less than the window's length"), 8,0,1,2)
			filterBox.setLayout(boxLayout)

			diagLayout.addWidget(samplBox,  0,0,1,1)
			diagLayout.addWidget(filterBox, 1,0,1,1)
			diagLayout.addWidget(headBox, 	2,0,1,1)
			diagLayout.addWidget(closeBtn, 	3,0,1,2, QtCore.Qt.AlignRight)

			headBox.setStyleSheet("QGroupBox{font-weight: bold;}")
			samplBox.setStyleSheet("QGroupBox{font-weight: bold;}")
			filterBox.setStyleSheet("QGroupBox{font-weight: bold;}")

			qDiag.setModal(True)
			qDiag.exec()

		def showParseOpt():
			qDiag = QtWidgets.QDialog(parent)
			qDiag.setWindowTitle("Parser settings")

			diagLayout = QtWidgets.QGridLayout()
			qDiag.setLayout(diagLayout)

			winLabel = QtWidgets.QLabel("<u>Currently using</u>: ... algorithm.")
			winInfoLabel = QtWidgets.QLabel("These algorithms are used to identify fixations and saccades<br />in raw eye tracking data (see <a href=\"https://dl.acm.org/doi/abs/10.1145/355017.355028\">Salvucci et Goldberg, 2000</a>).")
			winInfoLabel.setOpenExternalLinks(True)
			tabWidget = QtWidgets.QTabWidget()

			closeBtn = QtWidgets.QPushButton("Close")
			closeBtn.clicked.connect(lambda: qDiag.close())

			diagLayout.addWidget(winLabel, 0,0,1,1)
			diagLayout.addWidget(winInfoLabel, 1,0,1,1)
			diagLayout.addWidget(tabWidget, 2,0,1,1)
			diagLayout.addWidget(closeBtn, 3,0,1,1, QtCore.Qt.AlignRight)

			# 			VT     HMM   CT    DT
			available = [True, True, True, False]

			def updateTabLabels(idx):
				# Fallback: IVT
				if not available[idx]: idx = 0
				for itab in range(tabWidget.count()):
					label = tabWidget.tabText(itab).split(" ")[0]
					if itab == idx:
						label += " (selected)"
					tabWidget.setTabText(itab, label)

				labelSelected = tabWidget.tabText(idx).split(" ")[0]
				winLabel.setText("<u>Currently using</u>: {} algorithm.".format(labelSelected))

				parent.sceneOption.changeSetting("GP.parse", labelSelected)
				parent.sceneOption.settings["GP.parse"].updateUIFunc(labelSelected)

			def getTabIdxFromLabel(text):
				for itab in range(tabWidget.count()):
					label = tabWidget.tabText(itab).split(" ")[0]
					if label == text:
						return itab
				return -1

			# I-VT
			tabIVT = QtWidgets.QWidget(qDiag)

			info = QtWidgets.QLabel("<b>Velocity-threshold identification</b>")
			info.setToolTip("<u>Velocity</u>-based detection algorithm.\n<b>Note: short computation times.</b>")

			IVT_thresh = QtWidgets.QDoubleSpinBox()
			IVT_thresh.setRange(1, 500)
			IVT_thresh.setSingleStep(.1)
			IVT_thresh.setValue(parent.sceneOption["GP.IVT.threshold"])
			IVT_thresh.setSuffix(" deg/msec")
			IVT_thresh.valueChanged.connect(parent.sceneOption.settings["GP.IVT.threshold"])

			layoutT = QtWidgets.QGridLayout()
			layoutT.addWidget(info, 0,0,1,2)
			layoutT.addWidget(QtWidgets.QLabel("Velocity threshold"), 1,0,1,1)
			layoutT.addWidget(IVT_thresh, 1,1,1,1)
			layoutT.addWidget(QtWidgets.QLabel("Velocity samples below this threshold are labelled as part of fixations,\nabove as part of an eye movements (e.g., saccade, blink, smooth pursuit)."), 2,0,1,2)
			tabIVT.setLayout(layoutT)
			tabWidget.addTab(tabIVT, "I-VT")

			# I-HMM
			tabIHMM = QtWidgets.QWidget(qDiag)
			layoutT = QtWidgets.QGridLayout()

			try:
				import pomegranate
				info = QtWidgets.QLabel("<b>HMM identification</b>.<br />A Hidden Markov Model with Gaussian emissions is trained on the gaze velocity signal.")
				info.setToolTip("<u>HMM</u>-based detection algorithm.\n<b>Note: medium computation times.</b>")
				IHMM_nstate = QtWidgets.QSpinBox()
				IHMM_nstate.setRange(2, 10)
				IHMM_nstate.setValue(parent.sceneOption["GP.IHMM.nStates"])
				IHMM_nstate.valueChanged.connect(parent.sceneOption.settings["GP.IHMM.nStates"])

				layoutT.addWidget(QtWidgets.QLabel("Number of hidden states"), 1,0,1,1)
				layoutT.addWidget(IHMM_nstate, 1,1,1,1)
				layoutT.addWidget(QtWidgets.QLabel("Whatever the chosen number of hidden states (min: 2), fixations are decided\nas belonging to the hidden state showing the lowest mean velocity.\n\nMore than two hidden states can serve to identify more precisely eye movements,\nfor instance as saccades (moderate to high velocity) and blinks (very high velocity),\n in order to improve fixation identification."), 2,0,1,2)

			except:
				info = QtWidgets.QLabel("<u>HMM</u>-based detection algorithm <b>requires module \"pomegranate\"</b>.<br />Fallback: I-VT.")
				available[1] = False

			layoutT.addWidget(info, 0,0,1,2)
			tabIHMM.setLayout(layoutT)
			tabWidget.addTab(tabIHMM, "I-HMM")

			# I-CT
			tabICT = QtWidgets.QWidget(qDiag)
			layoutT = QtWidgets.QGridLayout()

			try:
				import sklearn
				info = QtWidgets.QLabel("<b>Cluster-based identification</b>")
				info.setToolTip("<u>Cluster</u>-based detection algorithm.\n<b>Note: long computation times.</b>")

				ICT_eps = QtWidgets.QDoubleSpinBox()
				ICT_eps.setRange(0, 10000)
				ICT_eps.setSingleStep(.0001)
				ICT_eps.setDecimals(4) 
				ICT_eps.setValue(parent.sceneOption["GP.ICT.eps"])
				ICT_eps.valueChanged.connect(parent.sceneOption.settings["GP.ICT.eps"])

				ICT_minpts = QtWidgets.QSpinBox()
				ICT_minpts.setRange(1, 500)
				ICT_minpts.setValue(parent.sceneOption["GP.ICT.minpts"])
				ICT_minpts.valueChanged.connect(parent.sceneOption.settings["GP.ICT.minpts"])

				ICT_refLink = QtWidgets.QLabel("To learn about this algorithm and its parameters visit its Sciki-Learn <a href=\"https://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html\">documentation page</a>.")
				ICT_refLink.setOpenExternalLinks(True)

				layoutT.addWidget(QtWidgets.QLabel("eps"), 1,0,1,1)
				layoutT.addWidget(ICT_eps, 1,1,1,1)
				layoutT.addWidget(QtWidgets.QLabel("min_samples"), 2,0,1,1)
				layoutT.addWidget(ICT_minpts, 2,1,1,1)
				layoutT.addWidget(ICT_refLink, 3,0,1,2)
			except:
				info = QtWidgets.QLabel("<u>Cluster</u>-based detection algorithm <b>requires module \"sklearn\"</b>.<br />Fallback: I-VT.")
				available[2] = False

			layoutT.addWidget(info, 0,0,1,2)
			tabICT.setLayout(layoutT)
			tabWidget.addTab(tabICT, "I-CT")

			# I-DT - not implemented
			tabIDT = QtWidgets.QWidget(qDiag)

			info = QtWidgets.QLabel("<b>Dispersion-based identification</b>")

			layoutT = QtWidgets.QGridLayout()
			layoutT.addWidget(info)
			layoutT.addWidget(QtWidgets.QLabel("<b>Not yet implemented</b>.<br />Fallback: I-VT."))
			tabIDT.setLayout(layoutT)
			tabWidget.addTab(tabIDT, "I-DT")

			tabWidget.currentChanged.connect(updateTabLabels)
			iSelectedParser = getTabIdxFromLabel(parent.sceneOption["GP.parse"])
			tabWidget.setCurrentIndex(iSelectedParser)
			updateTabLabels(iSelectedParser)

			qDiag.setModal(True)
			qDiag.exec()

		self.GshowPreprocBtn.clicked.connect(showPreprocOpt)
		self.GshowParseBtn.clicked.connect(showParseOpt)

		self.GPtrackChoice = QtWidgets.QComboBox(self)
		self.GPtrackChoice.addItem("Left eye")
		self.GPtrackChoice.addItem("Right eye")
		self.GPtrackChoice.addItem("Binocular")
		self.GPtrackChoice.addItem("Head trajectory")
		self.GPtrackChoice.setCurrentIndex(1)

		boxLayout = QtWidgets.QGridLayout(self)
		boxLayout.addWidget(QtWidgets.QLabel("Size"), 0, 0, 1, 1, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.GsizeVal, 0, 1, 1, 1)
		boxLayout.addWidget(self.Gcolour, 0, 2, 1, 1)
		boxLayout.addWidget(GcolGradVal, 1, 0, 1, 1)
		boxLayout.addWidget(GcolSolVal, 1, 1, 1, 1)
		boxLayout.addWidget(GcolAutoVaL, 1, 2, 1, 1)
		boxLayout.addWidget(self.GshowPreprocBtn, 2, 0, 1, 1)
		boxLayout.addWidget(self.GPtrackChoice, 2, 1, 1, 1)
		boxLayout.addWidget(self.GshowParseBtn, 3, 0, 1, 1)
		boxLayout.addWidget(self.currParseLab, 3,1 , 1, 1)
		self.gazeptsGBox.setLayout(boxLayout)
		boxLayout.setContentsMargins(0, 0, 0, 0)

		self.fixptsGBox = QtWidgets.QGroupBox("Fixation points")
		self.FsizeVal = QtWidgets.QSpinBox()
		self.FsizeVal.setRange(1, 200)
		self.FsizeVal.setSingleStep(1)
		self.FsizeVal.setValue(5)

		self.Fcolour = QtWidgets.QPushButton("Colour")
		colour = QtGui.QColor(255, 127, 255, 255)
		self.Fcolour.setPalette(QtGui.QPalette(colour))
		self.Fcolour.setText(colour.name())
		self.Fcolchoice = QtWidgets.QButtonGroup()
		FcolGradVal = QtWidgets.QRadioButton("Gradient")
		FcolSolVal  = QtWidgets.QRadioButton("Solid")
		[self.Fcolchoice.addButton(el) for el in [FcolGradVal, FcolSolVal]]
		FcolGradVal.setChecked(True)

		boxLayout = QtWidgets.QGridLayout(self)
		boxLayout.addWidget(QtWidgets.QLabel("Size"), 0, 0, 1, 1, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.FsizeVal, 0, 1, 1, 1)
		boxLayout.addWidget(self.Fcolour, 0, 2, 1, 1)
		boxLayout.addWidget(FcolGradVal, 1, 1, 1, 1)
		boxLayout.addWidget(FcolSolVal, 1, 2, 1, 1)
		self.fixptsGBox.setLayout(boxLayout)
		boxLayout.setContentsMargins(0, 0, 0, 0)

		self.viewportGBox = QtWidgets.QGroupBox("Viewport")
		self.VdegHorizVal = QtWidgets.QDoubleSpinBox()
		self.VdegHorizVal.setSuffix("°")
		self.VdegVertVal = QtWidgets.QDoubleSpinBox()
		self.VdegVertVal.setSuffix("°")
		self.VpixSizeVal = QtWidgets.QDoubleSpinBox()

		self.VdegHorizVal.setRange(10, 360); self.VdegVertVal.setRange(10, 180)
		self.VpixSizeVal.setRange(1., 10.); self.VpixSizeVal.setValue(1.); self.VpixSizeVal.setSingleStep(0.25)

		self.VdegHorizVal.setValue(90); self.VdegVertVal.setValue(90)

		boxLayout = QtWidgets.QGridLayout(self)
		self.viewportGBox.setLayout(boxLayout)
		boxLayout.addWidget(QtWidgets.QLabel("Angle Horiz."), 0, 0, 1, 1, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.VdegHorizVal, 0, 1, 1, 1)
		boxLayout.addWidget(QtWidgets.QLabel("Vert."), 0, 2, 1, 1, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.VdegVertVal, 0, 3, 1, 1)
		boxLayout.addWidget(QtWidgets.QLabel("Size multiplicator"), 1, 0, 1, 3, QtCore.Qt.AlignRight)
		boxLayout.addWidget(self.VpixSizeVal, 1, 3, 1, 1)
		boxLayout.setContentsMargins(0, 0, 0, 0)

		self.displayGBox = QtWidgets.QGroupBox("Toggle")
		self.DspVal = QtWidgets.QCheckBox("Sphere")
		self.DvpVal = QtWidgets.QCheckBox("Viewport")
		self.DidleVal = QtWidgets.QCheckBox("Idle anim.")
		self.DLoopVal = QtWidgets.QCheckBox("Mouse looping")
		self.DspVal.setChecked(True)
		self.DvpVal.setChecked(True)
		self.DidleVal.setChecked(True)
		self.DLoopVal.setChecked(True)

		boxLayout = QtWidgets.QHBoxLayout(self)
		self.displayGBox.setLayout(boxLayout)
		boxLayout.addWidget(self.DspVal, QtCore.Qt.AlignLeft)
		boxLayout.addWidget(self.DvpVal, QtCore.Qt.AlignLeft)
		boxLayout.addWidget(self.DidleVal, QtCore.Qt.AlignLeft)
		boxLayout.addWidget(self.DLoopVal, QtCore.Qt.AlignLeft)
		boxLayout.setContentsMargins(0, 0, 0, 0)

		self.salmapGBox.setAlignment(QtCore.Qt.AlignRight)
		self.fixptsGBox.setAlignment(QtCore.Qt.AlignRight)
		self.gazeptsGBox.setAlignment(QtCore.Qt.AlignRight)
		self.viewportGBox.setAlignment(QtCore.Qt.AlignRight)
		self.displayGBox.setAlignment(QtCore.Qt.AlignRight)

		self.salmapGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.fixptsGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.gazeptsGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.viewportGBox.setStyleSheet("QGroupBox{font-weight: bold;}")
		self.displayGBox.setStyleSheet("QGroupBox{font-weight: bold;}")

		self.regenBtn = QtWidgets.QPushButton(" Update data")
		self.regenBtn.setIcon(QtGui.QIcon.fromTheme("list-add"))
		self.regenBtn.setDisabled(True)

		self.saveSettBtn = QtWidgets.QPushButton(" Save settings")
		self.saveSettBtn.setIcon(QtGui.QIcon.fromTheme("document-save-as"))

		mainLayout = QtWidgets.QGridLayout(self)
		mainLayout.addWidget(self.salmapGBox,   0, 0, 1, 2)
		mainLayout.addWidget(self.gazeptsGBox,  0, 2, 1, 2)
		mainLayout.addWidget(self.viewportGBox, 1, 0, 1, 2)
		mainLayout.addWidget(self.fixptsGBox,   1, 2, 1, 2)
		mainLayout.addWidget(self.regenBtn, 	2, 0, 1, 1, QtCore.Qt.AlignBottom)
		mainLayout.addWidget(self.displayGBox,  2, 1, 1, 2)
		mainLayout.addWidget(self.saveSettBtn,  2, 3, 1, 1, QtCore.Qt.AlignBottom)
		self.setLayout(mainLayout)

		mainLayout.setColumnStretch(0, 50)
		mainLayout.setColumnStretch(1, 50)

		# mainLayout.setContentsMargins(0, 0, 0, 0)

class ImageButton(QtWidgets.QLabel):
	def __init__(self, parent, QPixmap=None, size=QtCore.QSize(100, 50)):
		super(ImageButton, self).__init__(parent)
		self.parent = parent

		self.isSelected = False

		self.callback = lambda bool_: None

		if QPixmap is None:
			QPixmap = QtGui.QPixmap(100, 50)
			QPixmap.fill(QtGui.QColor(0, 0, 0))
		else:
			self.isSelected = True

		self.setPixmap(QPixmap)
		self.setFixedSize(size)
		self.setScaledContents(True)
		self.toggleSelected(True)

	def mousePressEvent(self, event):
		self.setStyleSheet("border: 5px solid rgb(111, 30, 50);")

	def mouseReleaseEvent(self, event):
		self.toggleSelected()

	def setPixmap(self, QPixmap):
		super(ImageButton, self).setPixmap(QPixmap)

	def toggleSelected(self, bool_= None):
		self.isSelected = not self.isSelected if bool_ is None else bool_

		if self.isSelected:
			self.setStyleSheet("border: 2px solid rgb(60, 210, 40);")
		else:
			self.setStyleSheet("border: 2px solid  rgb(210, 30, 20);")

		self.callback(self.isSelected)

class CollapsibleBox(QtWidgets.QWidget):
	"""
	Author: eyllanesc
	Source: stackoverflow.com/questions/52615115/how-to-create-collapsible-box-in-pyqt/52617714#52617714
	"""
	def __init__(self, parent, title=""):
		super(CollapsibleBox, self).__init__(parent)

		self.parent = parent
		self.title = title

		self.toggle_button = QtWidgets.QToolButton(text=self.title, checkable=True, checked=False)
		self.toggle_button.setStyleSheet("font-weight: bold;")
		self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.toggle_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
		self.toggle_button.pressed.connect(self.on_pressed)

		self.toggle_animation = QtCore.QParallelAnimationGroup(self)

		self.content_area = QtWidgets.QScrollArea(maximumHeight=0, minimumHeight=0)
		self.content_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

		lay = QtWidgets.QVBoxLayout(self)
		lay.setSpacing(0)
		lay.setContentsMargins(0, 0, 0, 0)
		lay.addWidget(self.toggle_button)
		lay.addWidget(self.content_area)

		self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
		self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
		self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self.content_area, b"maximumHeight"))

	@QtCore.pyqtSlot()
	def on_pressed(self):
		checked = self.toggle_button.isChecked()
		self.toggle_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if not checked else QtCore.Qt.ArrowType.RightArrow)
		self.toggle_animation.setDirection(QtCore.QAbstractAnimation.Forward if not checked else QtCore.QAbstractAnimation.Backward)
		self.toggle_animation.start()

	def setContentLayout(self, layout):
		self.content_area.setLayout(layout)
		collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
		content_height = layout.sizeHint().height()

		for i in range(self.toggle_animation.animationCount()):
			animation = self.toggle_animation.animationAt(i)
			animation.setDuration(100)
			animation.setStartValue(collapsed_height)
			animation.setEndValue(collapsed_height + content_height)

		content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
		content_animation.setDuration(100)
		content_animation.setStartValue(0)
		content_animation.setEndValue(content_height)

class QHLine(QtWidgets.QFrame):
	# https://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt#answer-41068447
	def __init__(self):
		super(QHLine, self).__init__()
		self.setFrameShape(QtWidgets.QFrame.HLine)
		self.setFrameShadow(QtWidgets.QFrame.Sunken)
