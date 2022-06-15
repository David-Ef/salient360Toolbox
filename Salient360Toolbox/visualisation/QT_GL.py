#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019-2020
# Lab: IPI, LS2N, Nantes, France; Scene Grammar Lab, Frankfurt aM, Germany
# Comment: GUI to display, generate and save gaze data
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

try:
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GL.shaders import *
except:
	print("OpenGL wrapper for python not found")
	print("\trun `pip install PyOpenGL`")
	exit()

try:
	from PyQt5 import QtGui
	from PyQt5 import QtWidgets
	from PyQt5 import QtCore
except:
	print("QT 5 wrapper for python not found")
	print("\trun `pip install pyqt5`")
	exit()

import sys

from ..utils.misc import *
from . import GUIwidget

class MainWindow(QtWidgets.QMainWindow):
	instance = None

	def __init__(self, args):
		super(MainWindow, self).__init__()
		# General CLI args
		self.args = args
		# Menu
		self.statusBar = self.statusBar()
		self.lastPermaStatusWidget = None
		self.setStatusBar("Developed by Erwan David at (IPI, LS2N lab; Nantes, France) and (Scene Grammar Lab; Frankfurt aM, Germany)", perma=True)
		# Window content
		self.container = GUIwidget.WinContent(self)

		self.setCentralWidget(self.container)
		self.setMinimumSize(1200, 600)
		
		self.setGeometry(1920*2 + self.width()/2, 0 + self.height()/2,
						 self.width(), self.height())

		# Shortcut
		#	Save screenshot
		shortcut_save = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+s"), self)
		shortcut_save.activated.connect(self.container.Equirect.save_frame)
		#	Exit
		shortcut_save = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+w"), self)
		shortcut_save.activated.connect(QtWidgets.QApplication.quit)
		#	Toggle image display
		shortcut_save = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+r"), self)
		shortcut_save.activated.connect(lambda: None)
		#	Toggle gaze data display
		shortcut_save = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+t"), self)
		shortcut_save.activated.connect(lambda: None)

		MainWindow.instance = self

	def setStatusBar(self, text, perma=True):
		self.statusBar.clearMessage()
		printNeutral(text, "\n")
		if perma:
			if self.lastPermaStatusWidget is not None:
				self.statusBar.removeWidget(self.lastPermaStatusWidget)
			self.lastPermaStatusWidget = QtWidgets.QLabel(text)
			self.statusBar.addPermanentWidget(self.lastPermaStatusWidget)
		else:
			self.statusBar.showMessage(text)

	def resizeEvent(self, event):
		self.resize(self.width(), self.width()//2+80)
		# self.update()

def startApplication(paths=None, args=None, settings=None, display=None):
	app = QtWidgets.QApplication(['Salient360! Toolbox - Visualize and Generate 360° gaze data'])

	window = MainWindow(args)

	window.show()
	QtCore.QCoreApplication.processEvents()

	if args.load_settings is not None:
		# Load settings from file by overwriting CLI params
		if window.container.sceneOption.loadFromFile(args.load_settings, settings):
			window.setStatusBar("Loaded settings from {}".format(args.load_settings))

	if settings is not None:
		window.container.applySettings(settings)

	if paths is not None:
		for path in paths:
			window.container.Equirect.addData(path, concatenate=True)

	if type(display) == dict:
		window.container.DisplayInterface.BGImg.toggleSelected(display["bg"])
		window.container.DisplayInterface.SMImg.toggleSelected(display["sm"])
		window.container.DisplayInterface.SPImg.toggleSelected(display["sp"])
		window.container.DisplayInterface.GPImg.toggleSelected(display["gp"])

	sys.exit(app.exec_())

if __name__ == '__main__':
	startApplication()
