# exec macro ctrl f6
import FreeCAD
import Points
import Part
import os
import time
import numpy as np
import random
from scipy.spatial import distance
from collections import Counter
from pathlib import Path

dir_path = os.path.dirname(os.path.abspath(__file__))


def icompute_distances(shape, pts):
	dsts = []
	n_pts = len(pts)
	# start = time.time()

	for i in range(n_pts):
		pt = Part.Vertex(pts[i])
		dst = pt.distToShape(shape)
		dsts.append(dst)

	# end = time.time()
	# print(" comp : " + str(n_pts) + " , " + str(end - start))
	return dsts
def idistance_map(shape, pts):
	dsts = icompute_distances(shape, pts)
	avg = sum([dst[0] for dst in dsts]) / len(dsts)
	n_pts = len(dsts)
	max_tol = shape.getTolerance(1)

	on = []
	close = []
	medium = []
	far = []
	for i in range(n_pts):
		d = [dst[0] for dst in dsts][i]
		if d < max_tol:
			on.append(pts[i])
		elif d > avg * 2:
			far.append(pts[i])
		elif d > avg:
			medium.append(pts[i])
		elif d > avg / 2:
			close.append(pts[i])
		else:
			on.append(pts[i])

	return on, close, medium, far

def idistance_map_knn(shape, pts, k=3):
	if shape is None or pts is None:
		return None
	dsts = icompute_distances(shape, pts)
	avg = sum([dst[0] for dst in dsts]) / len(dsts)
	n_pts = len(dsts)

	labels = []
	for i in range(n_pts):
		d = [dst[0] for dst in dsts][i]
		if d > avg * 2:
			labels.append('far')
		elif d > avg:
			labels.append('medium')
		elif d > avg / 2:
			labels.append('close')
		else:
			labels.append('on')

	labels = knn(pts, labels, k)
	on_i = [i for i, label in enumerate(labels) if label == 'on']
	on = [pts[i] for i in on_i]
	close_i = [i for i, label in enumerate(labels) if label == 'close']
	close = [pts[i] for i in close_i]
	medium_i = [i for i, label in enumerate(labels) if label == 'medium']
	medium = [pts[i] for i in medium_i]
	far_i = [i for i, label in enumerate(labels) if label == 'far']
	far = [pts[i] for i in far_i]

	return on, close, medium, far

def knn(pts, labels, k= 3):
	print("knn with k =", k)
	# computes distance between each point, then sort by closest
	D = distance.squareform(distance.pdist(pts))
	closest = np.argsort(D, axis=1)
	kclosest = closest[:, 1:k + 1]
	for p_i in range(len(pts)):
		# get the labels of all nearest neighbours
		neighbour_labels = [labels[i] for i in kclosest[p_i, :]]
		c = Counter(neighbour_labels)
		true_label = c.most_common(1)[0][0]  # get most common label
		labels[p_i] = true_label
	return labels

def ifeature_matching_growing_bb(shape, pts):
	if shape is None:
		return None
	pt_matched = [False for p in range(len(pts))]
	faceIdx_pts_dict = {}
	scale = 1.0
	while False in pt_matched:
		for face_index in range(len(shape.Faces)):
			for pt_index in range(len(pts)):
				bb = shape.Faces[face_index].BoundBox
				bb.enlarge(scale)
				if bb.isInside(pts[pt_index]):
					# and not pt_matched[pt_index]
					pt_matched[pt_index] = True
					if face_index in faceIdx_pts_dict:
						if pts[pt_index] not in faceIdx_pts_dict[face_index]:
							faceIdx_pts_dict[face_index].append(pts[pt_index])
					else:
						faceIdx_pts_dict[face_index] = [pts[pt_index]]
		scale += 0.001

	return faceIdx_pts_dict


def ifeature_matching_bb(shape, pts):
	if shape is None:
		return None

	pt_matched = [False for p in range(len(pts))]
	faceIdx_pts_dict = {}
	for face_index in range(len(shape.Faces)):
		for pt_index in range(len(pts)):
			bb = shape.Faces[face_index].BoundBox
			if bb.isInside(pts[pt_index]):
				# and not pt_matched[pt_index]
				pt_matched[pt_index] = True
				if face_index in faceIdx_pts_dict:
					if pts[pt_index] not in faceIdx_pts_dict[face_index]:
						faceIdx_pts_dict[face_index].append(pts[pt_index])
				else:
					faceIdx_pts_dict[face_index] = [pts[pt_index]]
	not_matched_points = [pts[i] for i in range(len(pts)) if not pt_matched[i]]
	faceIdx_pts_dict[-1] = not_matched_points
	return faceIdx_pts_dict



def ifeature_matching_to_closest_bb(shape, pts):
	if shape is None or pts is None:
		return None
	pt_matched = [False for p in range(len(pts))]
	faceIdx_pts_dict = {}
	closest_face = {}
	for face_index in range(len(shape.Faces)):
		for pt_index in range(len(pts)):
			bb = shape.Faces[face_index].BoundBox
			if bb.isInside(pts[pt_index]) and not pt_matched[pt_index]:
				pt_matched[pt_index] = True
				if (pt_index in closest_face):
					closest_face.pop(pt_index)
				if face_index in faceIdx_pts_dict:
					if pts[pt_index] not in faceIdx_pts_dict[face_index]:
						faceIdx_pts_dict[face_index].append(pts[pt_index])
				else:
					faceIdx_pts_dict[face_index] = [pts[pt_index]]
			else:
				pt = pts[pt_index]
				dist = (pt - bb.Center).Length
				if pt_index in closest_face:
					if dist < closest_face[pt_index][1]:
						closest_face[pt_index] = (face_index, dist)
				else:
					closest_face[pt_index] = (face_index, dist)

	for pt_index in closest_face:
		face_id = closest_face[pt_index][0]
		if face_id in faceIdx_pts_dict:
			faceIdx_pts_dict[face_id].append(pts[pt_index])
		else:
			faceIdx_pts_dict[face_id] = [pts[pt_index]]

	return faceIdx_pts_dict

def bruitage(shp):

	tmp = [0 for x in range(shp.CountPoints)]
	for i in range(shp.CountPoints):
		rand = np.array([random.random(), random.random(),
						 random.random()]) * np.sign(cloud.Normal[i]) * 2
		displacementVector = FreeCAD.Vector(rand[0], rand[1], rand[2])
		tmp[i] = shp.Points[i] + displacementVector
	noise = Points.Points()
	noise.addPoints(shp.Points)
	noise.addPoints(tmp)

	doc = App.ActiveDocument
	noiseObj = doc.addObject("Points::Feature", "noisyObj")
	noiseObj.Points = noise


def bruit_gaussien(shp):

	gaussNoise = np.random.randn(shp.CountPoints, 3)
	tmp = [0 for x in range(shp.CountPoints)]
	for i in range(shp.CountPoints):
		tmp[i] = shp.Points[i] + \
				 FreeCAD.Vector(gaussNoise[i][0], gaussNoise[i][1], gaussNoise[i][2])
	noise = Points.Points()
	noise.addPoints(shp.Points)
	noise.addPoints(tmp)

	doc = App.ActiveDocument
	noiseObj = doc.addObject("Points::Feature", "noisyObj")
	noiseObj.Points = noise


def feature_matching_dst(shape, pts):
	if shape is None:
		return None
	pt_dst = [float("inf") for p in range(len(pts))]
	pt_face = [Part.Face() for p in range(len(pts))]
	faceIdx_pts_dict = {}

	for face_index in range(len(shape.Faces)):
		for pt_index in range(len(pts)):
			face = shape.Faces[face_index]
			pt = Part.Vertex(pts[pt_index])
			dst = pt.distToShape(face)
			if dst[0] < pt_dst[pt_index]:
				pt_face[pt_index] = face_index
				pt_dst[pt_index] = dst[0]

	for pt_index in range(len(pts)):
		face_index = pt_face[pt_index]
		if face_index in faceIdx_pts_dict:
			faceIdx_pts_dict[face_index].append(pts[pt_index])
		else:
			faceIdx_pts_dict[face_index] = [pts[pt_index]]

	doc = App.ActiveDocument
	matches = doc.addObject('App::Part', 'features_matches')
	for face_index, f_pts in faceIdx_pts_dict.items():
		feature_pts = Points.Points()
		feature_pts.addPoints(f_pts)
		fm_pts = doc.addObject("Points::Feature", "Face" + str(face_index))
		fm_pts.adjustRelativeLinks(matches)
		matches.addObject(fm_pts)
		fm_pts.Points = feature_pts
	return faceIdx_pts_dict


def feature_matching_optimized(shape, pts):
	if shape == None:
		return None
	faceIdx_pts_dict = pfe.feature_matching_bb()

	pt_face = [None for i in range(pts)]
	for face_index in range(len(faceIdx_pts_dict)):
		f_pts = faceIdx_pts_dict[face_index]
		if face_index >= 0:
			face = shape.Faces[face_index]
			neighbouring_faces = set()
			face_edges = shape.ancestorsOfType(face, Part.Edge)
			for edge in face_edges:
				n_faces = shape.ancestorsOfType(edge, Part.Face)
				for f in n_faces:
					neighbouring_faces.add(f)
			for pt in faceIdx_pts_dict[face_index]:
				pt_idx = pts.index(pt)
				neigh_contains = False
				for nf in neighbouring_faces:
					if nf != face:
						nf_index = shape.Faces.index(nf)

						if pt_face[pt_idx] is None:
							if pt in faceIdx_pts_dict[nf_index]:
								neigh_contains = True
								break
						else:
							break

				if not neigh_contains:
					pt_face[pt_idx] = face

			pt_dst = [float("inf") for p in range(len(f_pts))]
			pt_face = [Part.Face() for p in range(len(f_pts))]
			for nf in neighbouring_faces:
				for pt_index in range(len(f_pts)):
					pt = Part.Vertex(f_pts[pt_index])
					dst = pt.distToShape(nf)
					if dst[0] < pt_dst[pt_index]:
						pt_face[pt_index] = face_index
						pt_dst[pt_index] = dst[0]
		# WIP NOT WORKING yet

	# isoler face dans un compound
	# doc.addObject("Part::Compound","as")
	# ttt = Part.makeCompound(sub)
	# obj.Shape = ttt
	# distances


def fit_mesh_to_part(mesh, part):
	if mesh is None:
		return
	n_pts = len(mesh.Points)
	for pt_index in range(n_pts):
		pt = mesh.Points[pt_index]
		part_vertex = Part.Vertex(pt.Vector)
		d = part_vertex.distToShape(part)
		pt.move(d[1][0][1] - d[1][0][0])



'''
dsts = []
n_pts = len(pts)

for i in range(n_pts):
	pt = Part.Vertex(pts[i])
	dst = pt.distToShape(cube)
	dsts.append(dst)
#	print (dst[0])
avg = sum([dst[0] for dst in dsts]) / len(dsts)
print("average distance : ",avg)

out_pts = Points.Points()
on_pts = Points.Points()

on = []
out = []
for i in range(n_pts):
	if ([dst[2][0][3] for dst in dsts][i] == 'Face'):
		if ([dst[0] for dst in dsts][i] > avg):
			out.append(pts[i])
		else :
			on.append(pts[i])

# vec 0 [dst[1][0][1] for dst in dsts]
# vec 1 [dst[1][0][1] for dst in dsts]
out_pts.addPoints(out)
on_pts.addPoints(on)

out = doc.addObject("Points::Feature","out")
on = doc.addObject("Points::Feature","on")

out.Points = out_pts
on.Points = on_pts

# ptts = doc.addObject("Points::Feature","ptss")
#pts.ViewObject.ShapeColor = (1.0,1.0,1.0)
### End command Std_Import
# Macro End: /home/tom/.var/app/org.freecadweb.FreeCAD/data/FreeCAD/Macro/load_pfe.FCMacro +++++++++++++++++++++++++++++++++++++++++++++++++
'''
