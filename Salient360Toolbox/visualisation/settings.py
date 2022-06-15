#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

import os

class Setting():
	def __init__(self, name, value, action, updateUIFunc, doc):
		self.name = name
		self.value = value
		self.action = action
		self.updateUIFunc = updateUIFunc
		self.doc = doc

	def __call__(self, value, *args):
		"""Update option value
		Mostly called when updating with UI
		Can trigger an effect (self.action)
		"""
		if str(value.__class__.__bases__[0].__name__) == "QAbstractButton": value = value.text()
		self.value = value

		if self.action is not None:
			self.action(*args)

	def hasDoc(self):
		return isinstance(self.doc, str) and len(self.doc) > 0

class sceneGLOptions(object):
	def __init__(self):
		self.settings = {}

	def setSetting(self, name, value, action, updateUIFunc=lambda x: None, doc=""):
		self.settings[name] = Setting(name, value, action, updateUIFunc, doc)

		try:
			self.settings[name].updateUIFunc(self.settings[name].value)
		except Exception as e:
			pass
			# print("Could not set default value of option {}".format(self.settings[name].name))
			# print(e)

	def changeSetting(self, name, value):
		"""Update setting value without UI callback
		"""
		self.settings[name](value)

	def getKeys(self):
		"""Return all setting names
		"""
		return self.settings.keys()

	def showDoc(self):
		# Toolbox
		from ..utils.misc import printNeutral, printWarning
		# max_len = max([len(el[0]) for el in self.sceneOption])
		for name, setting in self:
			doc = setting.doc if setting.hasDoc() else "No documentation."
			value = setting.value

			printWarning("{}".format(doc),
				header="{}".format(name),
				# tab=float(max_len-len(name)),
				bold=False,
				verbose=-1)
			printNeutral("{}Current value: {}".format(" "* (len(name)), value),
				verbose=-1)

	def saveToFile(self, path, force=False):
		import numpy as np

		if os.path.isfile(path) and not force:
			return False
		else:
			with open(path, "w") as f:
				for name, setting in self:
					value = setting.value
					if type(setting.value) in [list, tuple, np.ndarray]:
						value = (",".join(["{}"]*len(value))).format(*value)
					
					f.write("{}:{}\n".format(name, value))

		return True

	def loadFromFile(self, path, args):
		from ..utils.misc import printWarning
		
		if os.path.isfile(path):
			with open(path, "r") as f:
				for line in f.readlines():
					line = line.split(":")
					if len(line) == 2:
						key, val = line
						args[key] = val.strip()
			return True
		else:
			printWarning("Could not load settings from file \"{}\"".format(os.path.realpath(path)))
			return False

	def __getitem__(self, key):
		"""Return a setting's current value
		"""
		if key in self.getKeys():
			return self.settings[key].value
		else:
			return None

	def __setitem__(self, key, value):
		"""Update setting value
		Mostly called when updating settings without the UI
		Will trigger a callback that will update the UI component's value
		"""
		if key in self.getKeys():
			self.settings[key].value = value
			self.settings[key].updateUIFunc(value)

	def __str__(self):
		"""Return all settings data as a string. Their name, value and type
		"""
		keys = list(self.getKeys())
		keys.sort()

		for key in keys:
			print("{} = {} ({})".format(key, self[key], type(self[key]).__name__))

	def __iter__(self):
		return iter(self.settings.items())
