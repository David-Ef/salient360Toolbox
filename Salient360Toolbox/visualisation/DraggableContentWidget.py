#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019
# Lab: IPI, LS2N, Nantes, France
# Comment: Modified from a pyqt example
# Source: https://github.com/pyqt/examples/blob/master/draganddrop/fridgemagnets/fridgemagnets.py
# ---------------------------------

#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

from PyQt5.QtCore import (QByteArray, QDataStream, QIODevice, QMimeData,
		QPoint, QRect, QRectF, Qt, pyqtSignal)
from PyQt5.QtGui import (QDrag, QFont, QFontMetrics, QImage, QPainter,
		QPalette, QPixmap, qRgba, QIcon, QColor)
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QGridLayout, QHBoxLayout, QGroupBox

import os, sys
loc = os.path.dirname(__file__)
loc = loc if len(loc) > 0 else "."

# Toolbox
from ..utils.divergingColorMaps import getColormapByName
colormaps = getColormapByName("nipy_spectral", levels=29) # Rainbow colormap

class DragLabel(QLabel):

	def __init__(self, true_idx, text, tooltip, parent):
		super(DragLabel, self).__init__(parent)

		self.text = text
		self.tooltip = tooltip
		self.true_idx = true_idx
		self.idx = None

		self.BgColour = QColor(*colormaps[self.true_idx].tolist(), 255)
		self.FontColour = (self.BgColour.redF()   * 0.299 +
						   self.BgColour.greenF() * 0.587 +
						   self.BgColour.blueF()  * 0.114) # Get luminance
		self.FontColour = Qt.white if self.FontColour < .4\
						  else Qt.black # Black or white font colour according to luminance

		self.setIndex(true_idx)
		self.setToolTip(self.tooltip)
		self.setAttribute(Qt.WA_DeleteOnClose, True)

	def setIndex(self, idx):
		self.idx = idx
		label = "[{}] {}".format(self.idx, self.text)

		self.position = self.pos()

		metric = QFontMetrics(self.font())
		size = metric.size(Qt.TextSingleLine, label)

		image = QImage(size.width() + 12, size.height() + 12,
				QImage.Format_ARGB32_Premultiplied)
		image.fill(qRgba(0, 0, 0, 0))

		font = QFont()
		font.setStyleStrategy(QFont.ForceOutline)

		painter = QPainter()
		painter.begin(image)
		painter.setRenderHint(QPainter.Antialiasing)
		painter.setBrush(self.BgColour)
		painter.drawRoundedRect(
				QRectF(0.5, 0.5, image.width()-1, image.height()-1),
				25, 25, Qt.RelativeSize)

		painter.setFont(font)
		painter.setPen(self.FontColour)
		painter.drawText(QRect(QPoint(6, 6), size), Qt.AlignCenter, label)
		painter.end()

		self.setPixmap(QPixmap.fromImage(image))

	def mousePressEvent(self, event):
		if event.button() == Qt.RightButton:
			if self.parent().name == "IN":
				self.setParent(self.parent().parent().ContentOut)
			else:
				self.setParent(self.parent().parent().ContentIn)
			self.move(1e10, 1e10)

			self.parent().parent().ContentIn.refreshChildrenOrder()
			self.parent().parent().ContentOut.refreshChildrenOrder()

			self.show()
			return

		itemData = QByteArray()
		dataStream = QDataStream(itemData, QIODevice.WriteOnly)
		dataStream << QByteArray(self.text.encode())\
				   << QByteArray(self.tooltip.encode())\
				   << QByteArray(self.true_idx.to_bytes(1, byteorder='big'))\
				   << QPoint(event.pos() - self.rect().topLeft())

		mimeData = QMimeData()
		mimeData.setData('application/x-dragGazeFeature', itemData)
		mimeData.setText(self.text)

		drag = QDrag(self)
		drag.setMimeData(mimeData)
		drag.setHotSpot(event.pos() - self.rect().topLeft())
		drag.setPixmap(self.pixmap())

		self.hide()

		# Dropped over same container: delete
		if drag.exec_(Qt.MoveAction | Qt.CopyAction, Qt.CopyAction) == Qt.MoveAction:
			self.close()
		else:
			# Dropped over other container: delete
			if self.parent().parent().ContentIn.underMouse() or\
			   self.parent().parent().ContentOut.underMouse():
				self.close()
			# Dropped over something else: move back to original position
			else:
				self.move(self.position)
				self.show()

	def closeEvent(self, event):
		super(DragLabel, self).closeEvent(event)

		exparent = self.parent()
		self.setParent(None)

		exparent.parent().ContentIn.refreshChildrenOrder()
		exparent.parent().ContentOut.refreshChildrenOrder()

class DragWidget(QWidget):
	def __init__(self, feature_list, name, parent):
		super(DragWidget, self).__init__(parent)

		self.name = name

		x = 5; xlim = 600
		y = 5
		nlines = 1

		for data in feature_list:
			wordLabel = DragLabel(*data, self)
			wordLabel.move(x, y)
			wordLabel.show()

			x += wordLabel.width() + 2
			if x >= xlim:
				x = 5
				y += wordLabel.height() + 2
				nlines += 1

				wordLabel.move(x, y)
				x += wordLabel.width() + 2

		newPalette = self.palette()
		newPalette.setColor(QPalette.Window, Qt.white)
		self.setPalette(newPalette)

		self.setAcceptDrops(True)
		self.setMinimumSize(max(100, xlim), max(100, y+30))

	def parent(self):
		return super(DragWidget, self).parent().parent()

	def dragEnterEvent(self, event):
		if event.mimeData().hasFormat('application/x-dragGazeFeature'):
			if event.source() in self.children():
				event.setDropAction(Qt.MoveAction)
				event.accept()
			else:
				event.acceptProposedAction()
		else:
			event.ignore()

	dragMoveEvent = dragEnterEvent

	def dropEvent(self, event):
		if event.mimeData().hasFormat('application/x-dragGazeFeature'):
			mime = event.mimeData()
			itemData = mime.data('application/x-dragGazeFeature')
			dataStream = QDataStream(itemData, QIODevice.ReadOnly)

			text = QByteArray()
			tooltip = QByteArray()
			trueidx = QByteArray()
			offset = QPoint()
			dataStream >> text\
					   >> tooltip\
					   >> trueidx\
					   >> offset
			
			text = str(text, encoding='latin1')
			tooltip = str(tooltip, encoding='latin1')
			trueidx = int.from_bytes(trueidx, byteorder='big')

			newLabel = DragLabel(trueidx, text, tooltip, self)

			height = newLabel.height()
			pos = event.pos() - offset

			newLabel.move(max(pos.x(), 0), max(pos.y()//newLabel.height() * height, 0) + 5)
			newLabel.show()

			if event.source() in self.children():
				event.setDropAction(Qt.MoveAction)
				event.accept()
			else:
				event.acceptProposedAction()
		else:
			event.ignore()

	def getFeatureOrder(self):
		import numpy as np

		pos = np.zeros([len(self.children())])
		label = []

		for iel, el in enumerate(self.children()):
			pos[iel] = el.pos().y()/el.height() * self.width() +  el.pos().x()
			label.append(el)

		order = np.argsort(pos)

		return order, label

	def refreshChildrenOrder(self):
		order, label = self.getFeatureOrder()

		x = 5; xlim = self.width()-10
		y = 5
		nlines = 1
		for ii, i in enumerate(order):
			# ii: order in parent
			# i: order in space
			label[int(i)].move(x, y)

			x += label[int(i)].width() + 2
			if x >= xlim:
				x = 5
				y += label[int(i)].height() + 2
				nlines += 1

				label[int(i)].move(x, y)
				x += label[int(i)].width() + 2
			label[int(i)].setIndex(ii)

		if self.name == "OUT":
			self.parent().emit_changed()

class DragFeatureWidget(QWidget):
	changed = pyqtSignal(list)

	def __init__(self, parent=None):
		super(DragFeatureWidget, self).__init__(parent)
		self.parent = parent

		# Double click to modify feature name
		# Add short howto at the top of the window

		from ..generation.scanpath import scanpath_header, scanpath_info
		scanpath_header = scanpath_header.split(",")

		headerIn = [[i, scanpath_header[i], scanpath_info[i].split(": ")[1]] for i in range(len(scanpath_header))]
		# Default: Idx[9], Timestamp[12], Gaze Long[0], Gaze Lat[1]
		headerOut = [headerIn.pop(i) for i in [9, 11, 0, 0]]

		BoxIn = QGroupBox("Unused:", self)
		BoxOut = QGroupBox("To export:", self)

		BoxIn.setStyleSheet("QGroupBox{font-weight: bold;}")
		BoxOut.setStyleSheet("QGroupBox{font-weight: bold;}")

		self.ContentIn = DragWidget(headerIn, "IN", parent=self)
		self.ContentOut = DragWidget(headerOut, "OUT", parent=self)

		boxLayout = QHBoxLayout(self)
		BoxIn.setLayout(boxLayout)
		boxLayout.addWidget(self.ContentIn)
		BoxIn.setLayout(boxLayout)

		boxLayout = QHBoxLayout(self)
		BoxOut.setLayout(boxLayout)
		boxLayout.addWidget(self.ContentOut)
		BoxIn.setLayout(boxLayout)

		MoveBottomBtn = QPushButton(parent=self)
		MoveBottomBtn.setIcon(QIcon.fromTheme("go-down"))
		MoveBottomBtn.setToolTip("Move all features to the bottom container.")
		MoveTopBtn = QPushButton(parent=self)
		MoveTopBtn.setIcon(QIcon.fromTheme("go-up"))
		MoveTopBtn.setToolTip("Move all features to the top container.")
		ResetBtn = QPushButton(parent=self)
		ResetBtn.setIcon(QIcon.fromTheme("document-revert"))
		ResetBtn.setToolTip("Move features to their default containers.")

		MoveBottomBtn.clicked.connect(lambda x: self.moveEverything(top=False))
		MoveTopBtn.clicked.connect(lambda x: self.moveEverything(top=True))
		ResetBtn.clicked.connect(lambda x: self.resetFeatureLocation())

		mainLayout = QGridLayout(self)

		mainLayout.addWidget(BoxIn, 0, 0, 1, 3)
		mainLayout.addWidget(BoxOut, 1, 0, 1, 3)

		mainLayout.addWidget(MoveBottomBtn, 2, 0, 1, 1)
		mainLayout.addWidget(MoveTopBtn, 2, 1, 1, 1)
		mainLayout.addWidget(ResetBtn, 2, 2, 1, 1)

		mainLayout.setRowStretch(0, 49)
		mainLayout.setRowStretch(1, 49)
		mainLayout.setRowStretch(2, 2)

		self.ContentOut.resize(self.ContentIn.width(), self.ContentIn.height())

		self.setMinimumSize(500, self.ContentIn.height() * 2 + 20)
		# self.adjustSize()

	def show(self):
		super(DragFeatureWidget, self).show()

		self.ContentIn.refreshChildrenOrder()
		self.ContentOut.refreshChildrenOrder()

	def moveEverything(self, top=False):
		children = self.ContentOut.children() if top else self.ContentIn.children()
		target = self.ContentIn if top else self.ContentOut

		for child in children:
			child.setParent(target)
			child.move(child.pos() + QPoint(0, 1e10))
			child.show()

		self.ContentIn.refreshChildrenOrder()
		self.ContentOut.refreshChildrenOrder()

	def resetFeatureLocation(self):
		from ..generation.scanpath import scanpath_header
		scanpath_header = scanpath_header.split(",")

		OUT = [9,12, 0,1]
		outNames = [scanpath_header[o] for o in OUT]
		inNames = [scanpath_header[i] for i in range(29) if i not in OUT]

		self.moveEverything()

		for child in [*self.ContentOut.children(), *self.ContentIn.children()]:
			idx = None
			if child.text in outNames:
				idx = outNames.index(child.text)
				child.setParent(self.ContentOut)
			else:
				idx = inNames.index(child.text)
				child.setParent(self.ContentIn)
			child.move(0, idx)
			child.show()

		self.ContentIn.refreshChildrenOrder()
		self.ContentOut.refreshChildrenOrder()
		
		self.changed.emit(self.getOutFeaturesSorted())

	def getOutFeaturesSorted(self):
		order, labels = self.ContentOut.getFeatureOrder()
		out = list(range(len(labels)))

		for ii, i in enumerate(order):
			out[ii] = [labels[i].true_idx, labels[i].text]

		return out

	def resizeEvent(self, event):
		self.ContentIn.refreshChildrenOrder()
		self.ContentOut.refreshChildrenOrder()

	def emit_changed(self):
		self.changed.emit(self.getOutFeaturesSorted())
