################################################################################
# MIT License
# 
# Copyright (c) 2018 Adina Wagner
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Functions taken/adapted from "Reimplementation of MultiMatch toolbox (Dewhurst et al., 2012) in Python."
# Original author: Adina Wagner (https://github.com/adswa)
# Source git repo: https://github.com/adswa/multimatch_gaze
# Source file: https://github.com/adswa/multimatch_gaze/blob/master/multimatch_gaze/multimatch_gaze.py
################################################################################

import numba
import numpy as np
import scipy.sparse as sp

@numba.jit
def createdirectedgraph(scanpath_dim, M):
	rows = []
	cols = []
	weight = []

	# loop through every node rowwise
	for i in range(0, scanpath_dim[0]):
		# loop through every node columnwise
		for j in range(0, scanpath_dim[1]):
			currentNode = i * scanpath_dim[1] + j

			# if in the last (bottom) row, only go right
			if (i == scanpath_dim[0] - 1) and (j < scanpath_dim[1] - 1):
				rows.append(currentNode)
				cols.append(currentNode+1)
				weight.append(M[i,j+1])

			# if in the last (rightmost) column, only go down
			elif (i < scanpath_dim[0] - 1) and (j == scanpath_dim[1] - 1):
				rows.append(currentNode)
				cols.append(currentNode + scanpath_dim[1])
				weight.append(M[i+1,j])

			# if in the last (bottom-right) vertex, do not move any further
			elif (i == scanpath_dim[0] - 1) and (j == scanpath_dim[1] - 1):
				rows.append(currentNode)
				cols.append(currentNode)
				weight.append(0)

			# anywhere else, move right, down and down-right.
			else:
				rows.append(currentNode) 
				rows.append(currentNode)
				rows.append(currentNode)
				cols.append(currentNode+1)
				cols.append(currentNode+scanpath_dim[1])
				cols.append(currentNode+scanpath_dim[1]+1)
				weight.append(M[i,j+1])
				weight.append(M[i+1,j])
				weight.append(M[i+1,j+1])

	rows = np.asarray(rows)
	cols = np.asarray(cols)
	weight = np.asarray(weight)
	numVert = scanpath_dim[0]*scanpath_dim[1]

	return numVert, rows, cols, weight

@numba.jit
def dijkstra(numVert, rows, cols, data, start, end, dim):
	#Create a scipy csr matrix from the rows,cols and append. This saves on memory.
	arrayWeightedGraph = sp.coo_matrix((data, (rows, cols)), shape=(numVert, numVert)).tocsr()

	#Run scipy's dijkstra and get the distance matrix and predecessors 
	dist_matrix, predecessors = sp.csgraph.dijkstra(csgraph=arrayWeightedGraph, directed=True, indices=0, return_predecessors=True)
	
	#Backtrack through the predecessors to get the reverse path
	patha = [end]
	dist = float(dist_matrix[end])
	#If the predecessor is -9999, that means the index has no parent and thus we have reached the start node
	while end != -9999:
		patha.append(predecessors[end])
		end = predecessors[end]

	patha = np.array(patha) 
	path = np.zeros([patha.shape[0], 2])
	path[:, 0] = patha // dim[1]
	path[:, 1] = patha % dim[1]

	# Return the path in ascending order and return the distance
	return path[-2::-1], dist
