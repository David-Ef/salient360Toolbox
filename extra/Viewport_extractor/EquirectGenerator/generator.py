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

path = os.path.dirname(os.path.abspath(__file__))

class EquirectView:
	
	BACKGROUND_IMAGE = None

	vertexShader = path+"/shaders/quadProj.vs"
	fragmentShaderEqui = path+"/shaders/EquiRec.fs"

	def __init__(self,
		BGimg_data=None,
		VPimg_data=None,
		path_VP=None,
		camAngle=[np.pi, 0, np.pi],
		angle_dim=[110, 110],
		pixel_dim=[500, 250]):

		self.vertexShader = EquirectView.openShaderScipt(EquirectView.vertexShader)
		self.fragmentShaderEqui = EquirectView.openShaderScipt(EquirectView.fragmentShaderEqui)

		self.camAngle = camAngle

		self.width = int(pixel_dim[1])
		self.height = int(pixel_dim[0])

		self.angleSpan = np.array(angle_dim)

		self.backgroundImageData = BGimg_data
		dim = self.backgroundImageData.shape
		self.backgroundImageData = self.backgroundImageData.reshape([dim[1], dim[0], 3])

		self.VPimgData = VPimg_data
		dim = self.VPimgData.shape
		self.VPimgData = self.VPimgData.reshape([dim[1], dim[0], 3])
		
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
		glutCreateWindow('equiretc')
		glutSetWindow(2)

		self.init_textures()
		self.init_Equirect()

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

		self.eMVMatrix = glGetUniformLocation(self.evProgram, "mvMatrix")
		self.eangleSpan = glGetUniformLocation(self.evProgram, "angleSpan")
		self.ecamAngle = glGetUniformLocation(self.evProgram, "camAngle")

		# self.ugazePos = glGetUniformLocation(self.evProgram, "gazepos")
		self.eBackgroundTexture = glGetUniformLocation(self.evProgram, "backgroundTexture")
		self.eVPTexture = glGetUniformLocation(self.evProgram, "VPTexture")

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

	def init_textures(self):
		self.backgroundTextureRef = glGenTextures(1)
		glActiveTexture(GL_TEXTURE1);
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

		dim = self.backgroundImageData.shape
		data = self.backgroundImageData.flatten()
		glTexImage2D(GL_TEXTURE_2D, 0,
				GL_RGB,
				dim[0], dim[1], 0,
				GL_RGB,
				GL_UNSIGNED_BYTE,
				data)

		# Viewport saliency texture
		self.VPTextureRef = glGenTextures(1)
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, self.VPTextureRef)
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

		dim = self.VPimgData.shape
		data = self.VPimgData.flatten()
		glTexImage2D(GL_TEXTURE_2D, 0,
				GL_RGB,
				dim[0], dim[1], 0,
				GL_RGB,
				GL_UNSIGNED_BYTE,
				data)

	def draw_Equirect(self):
		"""
		Draw equirectangular view.
		"""

		# use shader evProgram
		glUseProgram(self.evProgram)

		# set uniforms
		glUniform1i(self.eBackgroundTexture, 1)
		glUniform1i(self.eVPTexture, 2)

		glUniformMatrix4fv(self.eMVMatrix, 1, GL_FALSE, np.eye(4))

		glUniform3f(self.ecamAngle, *self.camAngle)
		glUniform2f(self.eangleSpan, *np.deg2rad(self.angleSpan))

		# enable attribute arrays
		glEnableVertexAttribArray(self.eVert)
		glEnableVertexAttribArray(self.eUV)

		# set vertex and UV buffers
		glBindBuffer(GL_ARRAY_BUFFER, self.evertexBuffer)
		glVertexAttribPointer(self.eVert, 3, GL_FLOAT, GL_FALSE, 0, None)
		glBindBuffer(GL_ARRAY_BUFFER, self.uvBufferE)
		glVertexAttribPointer(self.eUV, 2, GL_FLOAT, GL_FALSE, 0, None)

		# bind background texture
		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)
		glActiveTexture(GL_TEXTURE2)
		glBindTexture(GL_TEXTURE_2D, self.VPTextureRef)

		# glUniform1i(self.eBackgroundTexture, 0)
		# glUniform1i(self.eVPTexture, 1)

		# draw
		glDrawArrays(GL_TRIANGLES, 0, 6)

		# disable attribute arrays
		glDisableVertexAttribArray(self.eVert)
		glDisableVertexAttribArray(self.eUV)

	def update_textures(self, BGdata=None, VPdata=None):
		glutSetWindow(2)

		if BGdata is not None:
			self.backgroundImageData = BGdata

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, self.backgroundTextureRef)
			dim = self.backgroundImageData.shape
			data = self.backgroundImageData.flatten()

			glTexImage2D(GL_TEXTURE_2D, 0,
					GL_RGB,
					dim[0], dim[1], 0,
					GL_RGB,
					GL_UNSIGNED_BYTE,
					data)

		if VPdata is not None:
			self.VPimgData = VPdata

			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, self.VPTextureRef)
			dim = self.VPimgData.shape
			data = self.VPimgData.flatten()
			glTexImage2D(GL_TEXTURE_2D, 0,
					GL_RGB,
					dim[0], dim[1], 0,
					GL_RGB,
					GL_UNSIGNED_BYTE,
					data)

	def update_campos(self, pos):
		self.camAngle = pos

	def _draw_frame(self):
		self.draw_Equirect()

	@staticmethod
	def swap_buffer():
		glutSetWindow(2)
		glutSwapBuffers()

	def get_frame(self, out=False):
		glutSetWindow(2)
		self._draw_frame()

		# glReadBuffer(GL_FRONT)
		buffer = glReadPixels(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 
							  GL_RGB,
							  GL_UNSIGNED_BYTE)

		image = np.fromstring(buffer,
			dtype=np.uint8)
			# dtype=np.float32)
		image = image.reshape([glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), 3])
		image = np.flip(image, axis=0)
		
		return image
