#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2020
# Lab: SGL, Goethe University, Frankfurt
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

class errorCheck(QtWidgets.QDialog):
	def __init__(self, title="Modal title", message="Empty message", parent=None):
		super(errorCheck, self).__init__(parent)
		
		self.setWindowTitle(title)
			
		# Show warning message
		messageLab = QtWidgets.QLabel(message)
		# Warning presence
		self.donotshow = QtWidgets.QCheckBox("Do not show this message in the future?")
		# Close modal window
		acceptBtn = QtWidgets.QPushButton("Close")
		acceptBtn.clicked.connect(lambda x: self.accept())

		layout = QtWidgets.QGridLayout(self)
		layout.addWidget(messageLab, 0, 0, 1, 4)
		layout.addWidget(self.donotshow, 1, 0, 1, 4)
		layout.addWidget(acceptBtn, 2, 3, 1, 1)
		self.setLayout(layout)
		
		self.setModal(True)

	def close(self):
		super(showMessage, self).close()

	def getResults(self):
		return self.result(), self.donotshow.isChecked()

class warningCheck(QtWidgets.QDialog):
	def __init__(self, title="Modal title", message="Empty message", parent=None):
		super(warningCheck, self).__init__(parent)
		
		self.setWindowTitle(title)
			
		# Show warning message
		messageLab = QtWidgets.QLabel(message)
		# Warning presence
		self.donotshow = QtWidgets.QCheckBox("Automatically apply this decision in the future as well?")
		# Close modal window
		rejectBtn = QtWidgets.QPushButton("No")
		rejectBtn.clicked.connect(lambda x: self.reject())
		acceptBtn = QtWidgets.QPushButton("yes")
		acceptBtn.clicked.connect(lambda x: self.accept())

		layout = QtWidgets.QGridLayout(self)
		layout.addWidget(messageLab, 0, 0, 1, 4)
		layout.addWidget(self.donotshow, 1, 0, 1, 4)
		layout.addWidget(rejectBtn, 2, 2, 1, 1)
		layout.addWidget(acceptBtn, 2, 3, 1, 1)
		self.setLayout(layout)
		
		self.setModal(True)

	def close(self):
		super(showMessage, self).close()

	def getResults(self):
		return self.result(), self.donotshow.isChecked()

class showMessage(QtWidgets.QDialog):
	def __init__(self, title="Modal title", message="Empty message", parent=None):
		super(showMessage, self).__init__(parent)
		
		self.setWindowTitle(title)

		messageLab = QtWidgets.QLabel(message)
		acceptBtn = QtWidgets.QPushButton("Save")

		layout = QtWidgets.QGridLayout(self)
		layout.addWidget(messageLab, 1, 0, 1, 3)
		layout.addWidget(acceptBtn, 5, 2, 1, 1)
		self.setLayout(layout)

		acceptBtn.clicked.connect(lambda x: self.accept())
		
		self.setModal(True)

		# from ModalWindows import showMessage
		# showMessage("Title", "Messsage", parent).exec()

	def close(self):
		super(showMessage, self).close()

class progressBar(QtWidgets.QProgressDialog):
	def __init__(self, title, cancelTxt, parent=None):
		super(progressBar, self).__init__(title, cancelTxt, 0, 100, parent)
		
		setWaitCursor(True)

		self.setAutoClose(True)
		self.setModal(True)

	def setValue(self, val, text=None):
		super(progressBar, self).setValue(val * 100)

		if text is not None:
			self.setLabelText(text)

		QtWidgets.QApplication.processEvents()

		if self.wasCanceled():
			self.close()
			return False
		return True

	def close(self):
		super(progressBar, self).close()
		setWaitCursor(False)

def setWaitCursor(status):
	if status:
		QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
	else:
		QtWidgets.QApplication.restoreOverrideCursor()
