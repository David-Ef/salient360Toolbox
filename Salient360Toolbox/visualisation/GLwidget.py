#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2019-2020
# Lab: IPI, LS2N, Nantes, France
# Comment: 
# ---------------------------------

try:
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GL.shaders import *
except:
	print("OpenGL wrapper for python not found")
	print("\trun `pip install PyOpenGL glu`")
	exit()

try:
	from PyQt5 import QtGui
	from PyQt5 import QtWidgets
	from PyQt5 import QtCore
except:
	print("QT 5 wrapper for python not found")
	print("\trun `pip install pyqt5`")
	exit()

import time
import numpy as np
import cv2

def imread(path):
	return cv2.imread(path)[:, :, [2, 1, 0]]
def imsave(path, mat):
	return cv2.imwrite(path, mat[:, :, [2, 1, 0]])

from ..utils.misc import *
from .GazeData import GazeData

class EquirectView(QtWidgets.QOpenGLWidget):
	"""
	Original implementation based on:
		Author: rdmilligan
		Source: rdmilligan.wordpress.com/2016/08/27/opengl-shaders-using-python -- archive.fo/h0EVf
	"""

	vertexShader = "./shaders/quadProj.vs"
	"""
	vertex shader (simple quad mapping)
	"""

	fragmentShaderEqui = "./shaders/EquiRec.fs"
	"""
	fragment shader showing equirectangular content and viewport limit (red border)
	"""

	fragmentShaderVP = "./shaders/Viewport.fs"
	"""
	fragment shader showing viewport content. Lower right corner with red border
	"""

	vertexShaderGP = "./shaders/gazePoints.vs"
	"""
	vertex shader showing gaze points. Varies position and size
	"""

	fragmentShaderGP = "./shaders/gazePoints.fs"
	"""
	fragment shader showing gaze points. Varies shape and colour
	"""

	__doc__ = """How to:
  `Ctrl+w`: quit

  Drag-and-drop raw gaze file(s) (CSV format) to process and be displayed
  Drag-and-drop fixation list file(s) (CSV format) to load and be displayed
  Drag-and-drop a picture to replace the background image. If you drop a video file a frame will be extracted
	"""
	"""
	Help: usage
	"""

	def __init__(self, parent,
			openGLVer=130):
		super(EquirectView, self).__init__(parent)
		self.parent = parent
		self.mainWin = parent.parent

		self.openGLVer = openGLVer
		
		# Default background image
		self.BACKGROUND_IMAGE = None
		self.backgroundImage = None

		self.setMinimumSize(640, 320)

		self.camAngle = [np.pi, 0, np.pi]
		"""camera angle set by mouse position controlling viewport location. Set to middle of equirectangular by default."""

		self.angleSpan = lambda: np.array([self.parent.sceneOption["VP.as-x"],
										   self.parent.sceneOption["VP.as-y"]])
		""" Camera angle span (as set by the eytracker)"""

		self.VPpos = np.zeros(2)
		"""Current viewport position in window"""

		self.VPbackgroundVertices = None
		"""Default viewport position (bottom-right). Can be translated by holding mouse btn 1 and moving."""

		self.sphereRotTimer = time.time()-2
		"""Time since last mouse interaction. We wait 2sec before triggering the sphere idle animation"""

		# Gaze data
		GazeData.iconCallback = self.setIntPicture

		self.frame_idx = 0
		self.record = False
		self.path = "./frame_out/"

		# Interactivity
		self._timer = QtCore.QBasicTimer()
		self._timer.start(1000 / 60, self)
		self.drag_in = False

		self.TimeSinceLastUpdate = QtCore.QElapsedTimer()
		self.TimeSinceLastUpdate.start()

		# Max number of points to display (gaze or fixation points)
		self.max_fixpts = 1e6

		self.backgroundImageData = None
		"""Equirectangular picture."""

		print(EquirectView.__doc__)

		self.showSM = True
		self.showBG = True
		self.showGP = True
		self.showSP = True

		self.init_complete = False

		self.parent.DisplayInterface.BGImg.callback = lambda b: self.toggleDisplay("bg", b) # background
		self.parent.DisplayInterface.SMImg.callback = lambda b: self.toggleDisplay("sm", b) # saliency map
		self.parent.DisplayInterface.GPImg.callback = lambda b: self.toggleDisplay("gp", b) # raw gaze points
		self.parent.DisplayInterface.SPImg.callback = lambda b: self.toggleDisplay("sp", b) # scanpath

		# self.setBackgroundTexture(self.BACKGROUND_IMAGE)

	def openShaderScipt(self, path):
		loc = os.path.dirname(__file__)
		loc = loc if len(loc) > 0 else "."

		path = loc+"/"+path
		with open(path, "r") as file:
			content = file.read()

		# Change openGL version, in case 130 is too old (as seen with some Macs, 400 worked)
		content = content.replace("version 130", "version {:03}".format(self.openGLVer))
		return content

	def toggleDisplay(self, type_, value):
		if type_ == "bg":
			self.showBG = value
			self.update_texture()

		elif type_ == "gp":
			if value and self.showSP:
				self.parent.DisplayInterface.SPImg.toggleSelected(False)
			self.showGP = value
			self.set_GazePointData()

		elif type_ == "sp":
			if value and self.showGP:
				self.parent.DisplayInterface.GPImg.toggleSelected(False)
			self.showSP = value
			self.set_GazePointData()

		elif type_ == "sm":
			self.showSM = value
			self.update_texture()

		self.update()

	def setIntPicture(self, name, data, dim):
		if name == "bg":
			obj = self.parent.DisplayInterface.BGImg
		elif name == "gp":
			obj = self.parent.DisplayInterface.GPImg
		elif name == "sp":
			obj = self.parent.DisplayInterface.SPImg
		elif name == "sm":
			obj = self.parent.DisplayInterface.SMImg
		else:
			return

		obj.setPixmap(
			QtGui.QPixmap(
				QtGui.QImage(
					   data.copy(),
					   dim[1],
					   dim[0],
					   QtGui.QImage.Format_RGB888
					   )
				)
			)

	def setBackgroundTexture(self, data):
		# Data leak
		if self.backgroundImage is not None:
			self.backgroundImage = None

		if type(data) is str:
			if not os.path.exists(data):
				self.mainWin.setStatusBar("Could load image at "+data, perma=False)
				return
			img = imread(data)#, mode="RGB")
			img = cv2.resize(img, (1000,2000))
			self.backgroundImage = img
		else:
			self.backgroundImage = data

		dim = self.backgroundImage.shape
		# Reorder H,W,C
		self.backgroundImage = self.backgroundImage.reshape([dim[1], dim[0], dim[2]])
		# Reshape W*H, C
		self.backgroundImageData = self.backgroundImage.reshape([np.prod(dim[:2]), dim[-1]])
		# # Remove alpha
		# self.backgroundImageData = self.backgroundImageData[:, :3]

		self.setIntPicture("bg", self.backgroundImage, dim)

		if self.parent.gazeData.fix_list is not None:
			# Generate blended saliency map
			self.parent.gazeData.generate(saliency=False, fixmap=False, rawmap=False, salmap=False)
		else:
			self.update_texture()
			self.update()

	def getViewportVertexPos(self, mx, my):
		newVPbackgroundVertices = np.zeros([6, 3], dtype=np.float32)

		self.VPpos[:] = [mx, my]

		mx = mx * 2  -1 # [0, 1] to [1, -1]
		my = (2- my * 2) -1

		rectSize = (self.angleSpan() * self.parent.sceneOption["VP.mult"]) / [self.width(), self.height()]

		# Triangle 1
		newVPbackgroundVertices[0] = [mx - rectSize[0], my + rectSize[1], 0]
		newVPbackgroundVertices[1] = [mx - rectSize[0], my - rectSize[1], 0]
		newVPbackgroundVertices[2] = [mx + rectSize[0], my + rectSize[1], 0]
		# Triangle 2
		newVPbackgroundVertices[3] = [mx + rectSize[0], my + rectSize[1], 0]
		newVPbackgroundVertices[4] = [mx - rectSize[0], my - rectSize[1], 0]
		newVPbackgroundVertices[5] = [mx + rectSize[0], my - rectSize[1], 0]

		return newVPbackgroundVertices.flatten()

	def init_Equirect(self):
		"""
		Initialize openGL elements necessary to display the equirectangular view.
		"""
		# create shader program
		vs = compileShader(self.vertexShader, GL_VERTEX_SHADER)
		fs = compileShader(self.fragmentShaderEqui, GL_FRAGMENT_SHADER)
		self.evProgram = compileProgram(vs, fs)

		# obtain uniforms and attributes
		self.eVert = glGetAttribLocation(self.evProgram, "vert")
		self.eUV = glGetAttribLocation(self.evProgram, "uV")

		self.eBright = glGetUniformLocation(self.evProgram, "brightness")
		self.eMVMatrix = glGetUniformLocation(self.evProgram, "mvMatrix")
		self.eangleSpan = glGetUniformLocation(self.evProgram, "angleSpan")
		self.ecamAngle = glGetUniformLocation(self.evProgram, "camAngle")
		self.eshowBorder = glGetUniformLocation(self.evProgram, "showBorder")
		self.eBackgroundTexture = glGetUniformLocation(self.evProgram, "backgroundTexture")

		backgroundVertices = [
			-1.0,  1, 0.0, 
			-1.0, -1, 0.0,
			 1.0,  1, 0.0, 
			 1.0,  1, 0.0, 
			-1.0, -1, 0.0, 
			 1.0, -1, 0.0]

		self.evertexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.evertexBuffer)
		vertexData = np.array(backgroundVertices, np.float32)
		glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

		# set background UV
		backgroundUV = [
			0.0, 0.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			1.0, 1.0]

		self.uvBufferE = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferE)
		uvData = np.array(backgroundUV, np.float32)
		glBufferData(GL_ARRAY_BUFFER, 4 * len(uvData), uvData, GL_STATIC_DRAW)

	def init_texture(self):
		if not self.init_complete: return

		self.backgroundTextureRef = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

		self.update_texture()

	def update_texture(self):
		if self.showBG and self.showSM and self.parent.gazeData.blend_sal_map is not None:
			# blend
			dim = self.parent.gazeData.blend_sal_map.shape[:2]
			data = self.parent.gazeData.blend_sal_map.flatten()
		elif self.showBG and self.backgroundImageData is not None:
			# BG
			dim = self.backgroundImage.shape[:2]
			data = self.backgroundImageData
		elif self.showSM and self.parent.gazeData.sal_map is not None:
			# salmap
			dim = self.parent.gazeData.sal_image.shape[:2]
			data = self.parent.gazeData.sal_image.flatten()
		else:
			# empty
			dim = [1, 1]
			data = np.array([127, 127, 127])

		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim[0], dim[1], 0, GL_RGB, GL_UNSIGNED_BYTE, data)

	def draw_Equirect(self):
		"""
		Draw equirectangular view.
		"""

		# use shader evProgram
		glUseProgram(self.evProgram)

		# set uniforms
		glUniform1f(self.eBright, .5 if self.drag_in else 1)

		glUniformMatrix4fv(self.eMVMatrix, 1, GL_FALSE, np.eye(4))

		glUniform3f(self.ecamAngle, *self.camAngle)
		glUniform2f(self.eangleSpan, *np.deg2rad(self.angleSpan()))
		glUniform1i(self.eshowBorder, int(self.parent.sceneOption["Tog.viewp"]))

		glUniform1i(self.eBackgroundTexture, 0)

		# enable attribute arrays
		glEnableVertexAttribArray(self.eVert)
		glEnableVertexAttribArray(self.eUV)

		# set vertex and UV buffers
		glBindBuffer(GL_ARRAY_BUFFER, self.evertexBuffer)
		glVertexAttribPointer(self.eVert, 3, GL_FLOAT, GL_FALSE, 0, None)
		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferE)
		glVertexAttribPointer(self.eUV, 2, GL_FLOAT, GL_FALSE, 0, None)

		# bind background texture
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)

		# draw
		glDrawArrays(GL_TRIANGLES, 0, 6)

		# disable attribute arrays
		glDisableVertexAttribArray(self.eVert)
		glDisableVertexAttribArray(self.eUV)

	def init_GazePoints(self):
		"""
		Adapted from vispy point cloud example
		SOURCE github.com/vispy/vispy/blob/master/examples/demo/gloo/cloud.py
		"""

		vs = compileShader(self.vertexShaderGP, GL_VERTEX_SHADER)
		fs = compileShader(self.fragmentShaderGP, GL_FRAGMENT_SHADER)
		self.gp_program = compileProgram(vs, fs)

		self.gp_a_position = glGetAttribLocation(self.gp_program, "a_position")
		self.gp_a_bg_color = glGetAttribLocation(self.gp_program, "a_bg_color")
		self.gp_a_size = glGetAttribLocation(self.gp_program, "a_size")

		self.gp_buff_position = glGenBuffers(1)
		self.gp_buff_bg_color = glGenBuffers(1)
		self.gp_buff_size = glGenBuffers(1)

	def set_GazePointSize(self):
		a_size = np.ones(self.gp_n, dtype=np.float32)
		a_size *= self.parent.sceneOption["GP.size"] if self.showGP else self.parent.sceneOption["FP.size"]

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_size)
		glBufferData(GL_ARRAY_BUFFER, 4 * a_size.size, a_size, GL_STATIC_DRAW)

	def set_GazePointColour(self):
		a_bg_color = np.zeros([0, 4], dtype=np.float32)
		colour = None

		type_ = None
		if self.showGP and self.parent.gazeData.raw_gaze is not None:
			colour = self.parent.sceneOption["GP.colour"]
			type_ = self.parent.sceneOption["GP.type"]
		elif self.showSP and self.parent.gazeData.fix_list is not None:
			colour = self.parent.sceneOption["FP.colour"]
			type_ = self.parent.sceneOption["FP.type"]

		if colour is not None:
			a_bg_color = np.zeros([self.gp_n, 4], dtype=np.float32)

			if type_ in ["Auto", "Label S/F"]:
				if self.parent.gazeData.gaze_marker is not None:
					markers = self.parent.gazeData.gaze_marker.copy()
					if self.gp_n < markers.shape[0]:
						markers = markers[np.linspace(0, markers.shape[0]-1, self.gp_n, dtype=int)]
					a_bg_color[self.parent.gazeData.gaze_marker==0, :] = [.2, .71, .06, 1]
					a_bg_color[self.parent.gazeData.gaze_marker==1, :] = [.06, .25, .7, 1]
				else:
					type_ = "Gradient" # Fallback

			if type_ == "Solid":
				a_bg_color[:] = colour

			elif type_ == "Gradient":
				a_bg_color[:] = colour * np.linspace(.5, 1.5, self.gp_n, dtype=np.float32)[:,None]
				a_bg_color[:, -1] = 1

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_bg_color)
		glBufferData(GL_ARRAY_BUFFER, 4 * a_bg_color.size, a_bg_color, GL_STATIC_DRAW)

	def set_GazePointPosition(self):
		data = np.zeros([0, 2])

		if self.showGP and self.parent.gazeData.raw_gaze is not None:
			data = self.parent.gazeData.raw_gaze.copy()
		elif self.showSP and self.parent.gazeData.fix_list is not None:
			data = self.parent.gazeData.fix_list.copy()

		a_position = data[:, :2].astype(np.float32) * 2 - 1
		a_position[:, 1] = -a_position[:, 1]

		if self.gp_n < a_position.shape[0]:
			a_position = a_position[np.linspace(0, a_position.shape[0]-1, self.gp_n, dtype=int)]

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_position)
		glBufferData(GL_ARRAY_BUFFER, 4 * a_position.size, a_position, GL_STATIC_DRAW)

	def set_GazePointData(self):
		"""
		Gaze point data: calls setters for size, colour and position
		"""

		self.gp_n = 0
		if self.showGP and self.parent.gazeData.raw_gaze is not None:
			self.gp_n = min(self.parent.gazeData.raw_gaze.shape[0], self.max_fixpts)
		elif self.showSP and self.parent.gazeData.fix_list is not None:
			self.gp_n = min(self.parent.gazeData.fix_list.shape[0], self.max_fixpts)

		if self.gp_n > self.max_fixpts:
			printWarning("Parsing error", "Maximum number of gaze point reached. Only {} points will be displayed.".format(self.max_fixpts))

		self.gp_n = int(min(self.gp_n, self.max_fixpts))

		self.set_GazePointSize()
		self.set_GazePointColour()
		self.set_GazePointPosition()

	def draw_FixationPts(self):
		"""
		Draw raw gaze or fixation points.
		Can easily draw hundreds of thousands of points on integrated graphics.
		"""

		glUseProgram(self.gp_program)

		glEnable(GL_PROGRAM_POINT_SIZE)
		glEnable(GL_POINT_SPRITE)

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()

		glOrtho(0, self.width(), 0, self.height(), 0, 1);

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		glEnableVertexAttribArray(self.gp_a_position)
		glEnableVertexAttribArray(self.gp_a_bg_color)
		glEnableVertexAttribArray(self.gp_a_size)

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_position)
		glVertexAttribPointer(self.gp_a_position, 2, GL_FLOAT, GL_FALSE, 0, None)

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_bg_color)
		glVertexAttribPointer(self.gp_a_bg_color, 4, GL_FLOAT, GL_FALSE, 0, None)

		glBindBuffer(GL_ARRAY_BUFFER, self.gp_buff_size)
		glVertexAttribPointer(self.gp_a_size, 1, GL_FLOAT, GL_FALSE, 0, None)

		glDrawArrays(GL_POINTS, 0, self.gp_n)
		
		glDisableVertexAttribArray(self.gp_a_position)
		glDisableVertexAttribArray(self.gp_a_bg_color)
		glDisableVertexAttribArray(self.gp_a_size)

		glDisable(GL_BLEND)

	def init_Viewport(self):
		"""
		Initialize openGL elements necessary to display the viewport view (lower right corner).
		Viewport view position is dictated by vertices positions in `backgroundVertices`.
		"""

		# create shader program
		vs = compileShader(self.vertexShader, GL_VERTEX_SHADER)
		fs = compileShader(self.fragmentShaderVP, GL_FRAGMENT_SHADER)
		self.vprogram = compileProgram(vs, fs)

		# obtain uniforms and attributes
		self.vVert = glGetAttribLocation(self.vprogram, "vert")
		self.vUV = glGetAttribLocation(self.vprogram, "uV")

		self.vMVMatrix = glGetUniformLocation(self.vprogram, "mvMatrix")
		self.vangleSpan = glGetUniformLocation(self.vprogram, "angleSpan")
		self.vcamAngle = glGetUniformLocation(self.vprogram, "camAngle")

		self.vtvbackgroundTexture = glGetUniformLocation(self.vprogram, "tvbackgroundTexture")

		self.vvertexBuffer = glGenBuffers(1)

		self.uvBufferV = glGenBuffers(1)

	def set_Viewport(self):
		
		self.VPbackgroundVertices = self.getViewportVertexPos(
			*([self.width(), self.height()] - (self.angleSpan()*self.parent.sceneOption["VP.mult"])/2) /
			   [self.width(), self.height()] -
			   [.01, .01 * self.width()/self.height()]
			)

		glBindBuffer(GL_ARRAY_BUFFER, self.vvertexBuffer)
		glBufferData(GL_ARRAY_BUFFER, 4 * len(self.VPbackgroundVertices), self.VPbackgroundVertices, GL_STATIC_DRAW)

		# set background UV
		backgroundUV = [
			0.0, 0.0,
			0.0, 1.0,
			1.0, 0.0,
			1.0, 0.0,
			0.0, 1.0,
			1.0, 1.0]

		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferV)
		uvData = np.array(backgroundUV, np.float32)
		glBufferData(GL_ARRAY_BUFFER, 4 * len(uvData), uvData, GL_STATIC_DRAW)

	def draw_Viewport(self):
		"""
		Draw viewport view.
		"""

		# use shader program
		glUseProgram(self.vprogram)

		# set uniforms
		glUniformMatrix4fv(self.vMVMatrix, 1, GL_FALSE, np.eye(4))

		glUniform3f(self.vcamAngle, *self.camAngle)
		glUniform2f(self.vangleSpan, *np.deg2rad(self.angleSpan()))

		glUniform1i(self.vtvbackgroundTexture, 0)

		# enable attribute arrays
		glEnableVertexAttribArray(self.vVert)
		glEnableVertexAttribArray(self.vUV)

		# set vertex and UV buffers
		glBindBuffer(GL_ARRAY_BUFFER, self.vvertexBuffer)
		glVertexAttribPointer(self.vVert, 3, GL_FLOAT, GL_FALSE, 0, None)
		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferV)
		glVertexAttribPointer(self.vUV, 2, GL_FLOAT, GL_FALSE, 0, None)

		# bind background texture
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)

		# draw
		glDrawArrays(GL_TRIANGLES, 0, 6)

		# disable attribute arrays
		glDisableVertexAttribArray(self.vVert)
		glDisableVertexAttribArray(self.vUV)

	def init_Sphere(self, subdivids=50):
		self.circle_geom = [0, 0]
		for i in range(subdivids+1):
			angle = i * ((2*np.pi)/subdivids)
			self.circle_geom.extend([np.cos(angle), np.sin(angle)])

		self.circle_geom = np.array(self.circle_geom, dtype=np.float32)

	def draw_Sphere(self):

		glUseProgram(0)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()

		glOrtho(-10., 30., -10., 10., -10., 10.)

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		# Red background circle acting as sphere border
		glEnableClientState(GL_VERTEX_ARRAY)

		vbo = glGenBuffers(1)
		glBindBuffer (GL_ARRAY_BUFFER, vbo)
		glBufferData(GL_ARRAY_BUFFER, len(self.circle_geom)*4, self.circle_geom, GL_STATIC_DRAW)

		glPushMatrix()

		glTranslatef(-7.5, -7.5, 0)
		glScalef(2.05, 2.05, 0.)
		glColor3ub(255, 0, 0)

		glBindBuffer(GL_ARRAY_BUFFER, vbo)
		glVertexPointer(2, GL_FLOAT, 0, None)

		glDrawArrays(GL_TRIANGLE_FAN, 0, len(self.circle_geom)//2)
		glPopMatrix()

		glDisableClientState(GL_VERTEX_ARRAY)

		# Sphere with texture
		glEnable(GL_CULL_FACE)
		glCullFace(GL_FRONT)
		# glCullFace(GL_BACK)
		glTranslatef(-7.5, -7.5, 0)

		glShadeModel(GL_SMOOTH)

		glMatrixMode(GL_MODELVIEW)
		gluLookAt(
				-np.cos(2*np.pi-self.camAngle[1]) * np.cos(self.camAngle[0] - np.pi/2) * 4, # cam X
				-np.cos(2*np.pi-self.camAngle[1]) * np.sin(self.camAngle[0] - np.pi/2) * 4, # cam Y
				-np.sin(2*np.pi-self.camAngle[1]) * 4, # cam Z
				 0, 0, 0, # Center (x,y,z)
				 0, 0, 1) # Up vector
		glRotatef(180, 0, 1, 0) # Flip latitude
		glRotatef(180, 0, 0, 1) # Flip longitude

		qobj = gluNewQuadric()
		gluQuadricDrawStyle(qobj, GLU_FILL)
		gluQuadricTexture(qobj, GL_TRUE)
		gluQuadricNormals(qobj, GLU_SMOOTH)
		# gluQuadricOrientation(qobj, GLU_OUTSIDE)

		glEnable(GL_NORMALIZE)
		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)

		gluSphere(qobj, 2, 100, 50)

		glDisable(GL_NORMALIZE)
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_CULL_FACE)

		gluDeleteQuadric(qobj)

		glUseProgram(0)

	def paintGL(self):
		glClearColor(1., .5, .5, 1.)
		glClear(GL_COLOR_BUFFER_BIT)# | GL_DEPTH_BUFFER_BIT)

		self.draw_Equirect()
		self.draw_FixationPts()
		if self.parent.sceneOption["Tog.viewp"]: self.draw_Viewport()
		if self.parent.sceneOption["Tog.sph"]: self.draw_Sphere()
		
	def initializeGL(self):
		glViewport(0, 0, self.width(), self.height())

		glClearDepth(1.0)

		# Read and store shader scripts
		self.vertexShader = self.openShaderScipt(EquirectView.vertexShader)
		self.fragmentShaderEqui = self.openShaderScipt(EquirectView.fragmentShaderEqui)
		self.fragmentShaderVP = self.openShaderScipt(EquirectView.fragmentShaderVP)
		self.vertexShaderGP = self.openShaderScipt(EquirectView.vertexShaderGP)
		self.fragmentShaderGP = self.openShaderScipt(EquirectView.fragmentShaderGP)

		self.init_complete = True
		self.init_texture()

		self.init_Equirect()
		self.init_GazePoints()
		self.set_GazePointData()
		self.init_Viewport()
		self.set_Viewport()
		self.init_Sphere()

		self.setMouseTracking(True)
		self.setAcceptDrops(True)

		self.parent.DisplayInterface.GPImg.toggleSelected(False)
		self.parent.DisplayInterface.SMImg.toggleSelected(False)

	def resizeGL(self, w, h):
		glViewport(0, 0, w, h)

		self.set_Viewport()

	def addData(self, paths, concatenate=False):
		def feedback(message): self.mainWin.setStatusBar(message, perma=False)

		if type(paths) not in [list, tuple]: paths = [paths]
		concatenate = concatenate or QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier
		
		# If list of paths contains at least one folder we process them as gaze files only
		if sum([os.path.isdir(p) for p in paths]) > 0:
			from glob import glob
			t_paths = []
			for p in paths:
				if os.path.isdir(p):
					t_paths.extend([file for file in glob(p+os.sep+"*") if os.path.isfile(file)])
				else:
					t_paths.append(p)
			paths = t_paths

		p_img = []
		p_vid = []
		p_data = []
		p_sett = None

		for path in paths:
			if os.path.exists(path):
				from mimetypes import MimeTypes

				mt = MimeTypes().guess_type(path)[0]
				if mt is not None:
					mt = mt.split("/")[0]
					if mt == "image":
						p_img.append(path)
					elif mt == "video":
						p_vid.append(path)
					elif mt == "text":
						p_data.append(path)
				else:
					if path[-4:] == ".set":
						p_sett = path

		# Load up first image found
		for path in p_img:
			self.BACKGROUND_IMAGE = path
			self.setBackgroundTexture(self.BACKGROUND_IMAGE)
			
			self.parent.gazeData.genBlendSalmap()

			if len(p_img) > 1:
				feedback("New background image: "+path+". Ignored other images in path list")
			else:
				feedback("New background image: "+path)
			break

		# Load up first video found (could replace an image in the same path list)
		for path in p_vid:
			from ..utils.misc import getVideoFrame
			frame = getVideoFrame(path, .05)
			# Rearrange channels and force data to be contiguous (else QtGui.QImage will fail)
			frame = np.ascontiguousarray(frame[:, :, [2, 1, 0]])
			frame = cv2.resize(frame, (1000, 2000))
			self.BACKGROUND_IMAGE = "data"
			self.setBackgroundTexture(frame)

			self.parent.gazeData.genBlendSalmap()

			if len(p_vid) > 1:
				feedback("New background video frame: "+path+". Ignored other video files in path list")
			else:
				feedback("New background video frame: "+path)
			break

		if len(p_data) > 0:
			stats = self.parent.gazeData.process(p_data, concatenate)
			feedback("Parsed {} raw gaze files and {} fixation files.".format(*stats))

		if p_sett is not None:
			settings = {}
			if self.parent.sceneOption.loadFromFile(p_sett, settings):
				self.parent.applySettings(settings)

	# EVENTS
	# Keyboard shortcuts
	def save_frame(self, out=False):
		"""Save a screenshot of the current openGL scene
		"""
		# Clear content so that it doesn't appear on the screenshot
		self.mainWin.setStatusBar("", perma=False)
		if not os.path.exists(self.path): os.mkdir(self.path)
		# Read and output window content
		glReadBuffer(GL_FRONT)
		buffer = glReadPixels(0, 0, self.width(), self.height(),
							  GL_RGB, GL_UNSIGNED_BYTE)
		# save to jpg
		image = np.fromstring(buffer, dtype=np.uint8)
		image = image.reshape([self.height(), self.width(), 3])
		image = np.flip(image, axis=0)
		path = "{}out_{:05}.jpg".format(self.path, self.frame_idx)
		imsave(path, image)
		self.mainWin.setStatusBar("Screenshot saved: {}".format(path), perma=False)
		self.frame_idx += 1

	# Drag and drop events
	def dragEnterEvent(self, event):
		self.drag_in = True
		event.acceptProposedAction()
		self.update()

	def dropEvent(self, event):
		self.drag_in = False
		event.acceptProposedAction()

		path = []
		mimeData = event.mimeData()
		if mimeData.hasUrls():
			path = mimeData.urls()
		elif mimeData.hasText():
			path = mimeData.text()

		path = [el.toLocalFile() for el in path]

		self.addData(path)

		self.update()

	def dragMoveEvent(self, event):
		event.accept()

	def dragLeaveEvent(self, event):
		self.drag_in = False
		event.accept()
		self.update()

	# Mouse events
	def leaveEvent(self, event):
		"""Force mouse to loop horizontally.
		When the mouse leaves the openGL scene from an horizontal border, it is relocated to the opposite side.
		"""
		modifCtrl = QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier
		if not self.mainWin.isActiveWindow() or modifCtrl or not self.parent.sceneOption["Tog.loop"]: return

		mouse = self.cursor()
		pos = self.mapFromGlobal(mouse.pos())

		if pos.x() >= self.width():
			mouse.setPos(self.mapToGlobal(QtCore.QPoint(0, pos.y())))
		elif pos.x() <= 0:
			mouse = self.cursor()
			mouse.setPos(self.mapToGlobal(QtCore.QPoint(self.width()-1, pos.y())))

	def mouseMoveEvent(self, event):
		"""Catch the mouse position and interpret it as the camera positon for the viewport
		"""
		modifCtrl = QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier

		if not modifCtrl:

			mx = self.width() - event.localPos().x()
			my = event.localPos().y()

			self.camAngle = np.array([(mx/self.width()) * (2*np.pi),# - np.deg2rad(self.angleSpan()[0]/2),
								 (my/self.height() - .5) * np.pi,
								 np.pi])
			self.sphereRotTimer = time.time()

			self.update()

	# Custom loop
	def timerEvent (self, event):
		modifCtrl = QtGui.QGuiApplication.keyboardModifiers() == QtCore.Qt.ControlModifier

		# Idle animation conditions
		if    not self.parent.gazeData.processing and\
			  not self.underMouse() and\
			  self.parent.sceneOption["Tog.idle"] and\
			  (time.time() - self.sphereRotTimer) >= 2:

			elapsedTime = self.TimeSinceLastUpdate.elapsed()/1000

			self.camAngle[0] = (self.camAngle[0] + elapsedTime/4) % (2*np.pi)
			self.update()

		self.TimeSinceLastUpdate.restart()
