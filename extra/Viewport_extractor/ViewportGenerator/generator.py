#! /usr/bin/env python3
# ---------------------------------
# Author: Erwan DAVID
# Year: 2018
# Lab: IPI, LS2N, Nantes, France
# Comment: Salient360° saliency and scanpath prediction model
# Cite: E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (2018). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
# ---------------------------------

try:
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GLUT import *
	from OpenGL.GL.shaders import *
except:
	print("OpenGL wrapper for python not found")
	print("\trun `pip install PyOpenGL`")
	exit()

import numpy as np
import cv2
import sys, time

path = os.path.dirname(os.path.abspath(__file__))

class ViewportView:
	
	BACKGROUND_IMAGE = None

	vertexShader = path+"/shaders/quadProj.vs"
	fragmentShaderVP = path+"/shaders/Viewport.fs"

	def __init__(self,
		path_img=None,
		camAngle=[np.pi, 0, np.pi],
		angle_dim=[110, 110],
		pixel_dim=[512, 512],
		max_pix_dim=(3000, 1500)):
		
		if path_img is not None:
			ViewportView.BACKGROUND_IMAGE = path_img

		if not os.path.exists(ViewportView.BACKGROUND_IMAGE):
			path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])
			ViewportView.BACKGROUND_IMAGE = path+os.sep+ViewportView.BACKGROUND_IMAGE

		self.vertexShader = ViewportView.openShaderScipt(ViewportView.vertexShader)
		self.fragmentShaderVP = ViewportView.openShaderScipt(ViewportView.fragmentShaderVP)

		self.camAngle = camAngle

		self.width = int(pixel_dim[0])
		self.height = int(pixel_dim[1])

		self.angleSpan = np.array(angle_dim)

		self.backgroundImageData = cv2.imread(ViewportView.BACKGROUND_IMAGE)[:,:,[2,1,0]]
		self.backgroundImageData = cv2.resize(self.backgroundImageData, max_pix_dim)

		dim = self.backgroundImageData.shape
		self.backgroundImageData = self.backgroundImageData.reshape([dim[1], dim[0], dim[2]])

		self.backgroundTextureRef = glGenTextures(1)
		
		self._init_opengl()

	@staticmethod
	def openShaderScipt(path):
		with open(path, "r") as file:
			content = file.read()
		return content

	def _init_opengl(self):
		glutInit()
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)

		glutInitWindowSize(self.width, self.height)
		glutInitWindowPosition(0, 0)
		glutCreateWindow('viewport')

		glutSetWindow(1)

		self.init_texture()
		self.init_Viewport()

	def init_Viewport(self):
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

		backgroundVertices = [
			-1.0,  1, 0.0, 
			-1.0, -1, 0.0,
			 1.0,  1, 0.0, 
			 1.0,  1, 0.0, 
			-1.0, -1, 0.0, 
			 1.0, -1, 0.0]

		self.vvertexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vvertexBuffer)
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

		self.uvBufferV = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferV)
		uvData = np.array(backgroundUV, np.float32)
		glBufferData(GL_ARRAY_BUFFER, 4 * len(uvData), uvData, GL_STATIC_DRAW)

	def init_texture(self):
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)
		# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

		dim = self.backgroundImageData.shape[:2]
		data = self.backgroundImageData.flatten()
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, dim[0], dim[1], 0, GL_RGB, GL_UNSIGNED_BYTE, data)

	def update_texture(self, data):
		glutSetWindow(1)

		self.backgroundImageData = data
		dim = self.backgroundImageData.shape
		self.backgroundImageData = self.backgroundImageData.reshape([dim[1], dim[0], dim[2]])

		glActiveTexture(GL_TEXTURE1);

		self.init_texture()

	def draw_Viewport(self):
		# use shader program
		glUseProgram(self.vprogram)

		# set uniforms
		glUniformMatrix4fv(self.vMVMatrix, 1, GL_FALSE, np.eye(4))

		glUniform3f(self.vcamAngle, *self.camAngle)
		glUniform2f(self.vangleSpan, *np.deg2rad(self.angleSpan))

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
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)

		# draw
		glDrawArrays(GL_TRIANGLES, 0, 6)

		# disable attribute arrays
		glDisableVertexAttribArray(self.vVert)
		glDisableVertexAttribArray(self.vUV)

	def _draw_frame(self):
		self.draw_Viewport()

	@staticmethod
	def swap_buffer():
		glutSwapBuffers()

	def get_frame(self, out=False):
		glutSetWindow(1)
		self._draw_frame()
		# glReadBuffer(GL_FRONT)
		buffer = glReadPixels(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 
							  GL_RGB, GL_UNSIGNED_BYTE)

		image = np.fromstring(buffer, dtype=np.uint8)
		image = image.reshape([glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), 3])
		image = np.flip(image, axis=0)
		
		return image
