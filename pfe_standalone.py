# exec macro ctrl f6
import FreeCAD, Mesh, Part, Points
import MeshPart
import os
import time
import numpy as np
import random
from scipy.spatial import distance
from collections import Counter
from pathlib import Path
from probreg import cpd
import open3d as o3d

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
	start = time.time()
	if shape is None:
		return None

	pt_matched = [False for p in range(len(pts))]
	face_idx_pts_idx = [[] for i in range(len(shape.Faces))]
	for face_index in range(len(shape.Faces)):
		for pt_index in range(len(pts)):
			bb = shape.Faces[face_index].BoundBox
			if bb.isInside(pts[pt_index]):
				# and not pt_matched[pt_index]
				pt_matched[pt_index] = True
				face_idx_pts_idx[face_index].append(pt_index)

	not_matched_points = [pt_index for pt_index in range(len(pts)) if not pt_matched[pt_index]]

	end = time.time()
	print("feature_matching_bb on", len(pts), " points : ", str(end - start))
	return face_idx_pts_idx, not_matched_points

def ifeature_matching_bb_bis(part, pts):
	if part is None:
		return None

	start = time.time()

	pt_index_faces_indexes = [[] for pt in pts]
	for face_index in range(len(part.Faces)):
		for pt_index in range(len(pts)):
			bb = part.Faces[face_index].BoundBox
			if bb.isInside(pts[pt_index]):
				pt_index_faces_indexes[pt_index].append(face_index)

	end = time.time()
	print("feature_matching_bb_bis on", len(pts), " points : ", str(end - start))

	return pt_index_faces_indexes



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




def ifeature_matching_dst(shape, pts):
	if shape is None:
		return None

	start = time.time()
	pt_dst = [float("inf") for p in range(len(pts))]
	pt_face = [Part.Face() for p in range(len(pts))]
	face_idx_pts_idx = [[] for i in range(len(shape.Faces))]

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
		face_idx_pts_idx[face_index].append(pt_index)

	end = time.time()
	print("feature_matching_dst on", len(pts), " points : ", str(end - start))

	return face_idx_pts_idx, []


def ifeature_matching_optimized(part, pts):
	if part == None:
		return None
	start = time.time()
	face_idx_pts_idx = [[] for i in range(len(part.Faces))]
	pt_idx_faces_idx = ifeature_matching_bb_bis(part, pts)
	lost_pts_idx = []

	pt_dst = [float("inf") for p in range(len(pts))]
	for pt_idx in range(len(pt_idx_faces_idx)):
		faces_idx = pt_idx_faces_idx[pt_idx]
		if len(faces_idx) == 0:
			# full search
			lost_pts_idx.append(pt_idx)
		elif len(faces_idx) == 1:
			face_idx_pts_idx[faces_idx[0]].append(pt_idx)
		elif len(faces_idx) > 1:
			face_dst = float("inf")
			closest_face_idx = faces_idx[0]
			for face_idx in faces_idx:
				face = part.Faces[face_idx]
				pt = Part.Vertex(pts[pt_idx])
				dst = pt.distToShape(face)
				if dst[0] < face_dst:
					closest_face_idx = face_idx
					face_dst = dst[0]
			# print("a : ", closest_face_idx)
			face_idx_pts_idx[closest_face_idx].append(pt_idx)

	for face_index in range(len(part.Faces)):
		for pt_index in lost_pts_idx:
			face = part.Faces[face_index]
			pt = Part.Vertex(pts[pt_index])
			dst = pt.distToShape(face)
			if dst[0] < pt_dst[pt_index]:
				face_idx_pts_idx[face_index].append(pt_index)
				pt_dst[pt_index] = dst[0]

	end = time.time()
	print("feature_matching_optimized on", len(pts), " points : ", str(end - start))

	return face_idx_pts_idx, []


	# isoler face dans un compound
	# doc.addObject("Part::Compound","as")
	# ttt = Part.makeCompound(sub)
	# obj.Shape = ttt
	# distances


def bruitage(shp):
	tmp = [0 for x in range(shp.CountPoints)]
	for i in range(shp.CountPoints):
		rand = np.array([random.uniform(-1,1), random.uniform(-1,1),
						 random.uniform(-1,1)])
		displacementVector = FreeCAD.Vector(rand[0], rand[1], rand[2])
		tmp[i] = shp.Points[i] + displacementVector
	noise = Points.Points()
	noise.addPoints(shp.Points)
	noise.addPoints(tmp)
	return noise


def bruit_gaussien(shp):
	gaussNoise = np.random.randn(shp.CountPoints, 3)
	tmp = [0 for x in range(shp.CountPoints)]
	for i in range(shp.CountPoints):
		tmp[i] = shp.Points[i] + \
				 FreeCAD.Vector(gaussNoise[i][0], gaussNoise[i][1], gaussNoise[i][2])
	noise = Points.Points()
	noise.addPoints(shp.Points)
	noise.addPoints(tmp)
	return noise

def fit_mesh_to_part(mesh, part):
	if mesh is None:
		return
	n_pts = len(mesh.Points)
	for pt_index in range(n_pts):
		pt = mesh.Points[pt_index]
		part_vertex = Part.Vertex(pt.Vector)
		d = part_vertex.distToShape(part)
		pt.move(d[1][0][1] - d[1][0][0])

def cloud_to_numpy(cloud):
	return np.array(cloud.Points.Points)

def numpy_to_o3d_cloud(array):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(array)
	return pcd

def freecad_to_o3d_cloud(cloud):
	array = cloud_to_numpy(cloud)
	return numpy_to_o3d_cloud(array)

def icpd(source, target):
	source_o3d = freecad_to_o3d_cloud(source)
	source_o3d.remove_non_finite_points()
	target_o3d = freecad_to_o3d_cloud(target)
	target_o3d.remove_non_finite_points()
	print("Computing registration...")
	tf_param, _, _ = cpd.registration_cpd(source_o3d, target_o3d)
	print("Finished registration with:\nrotation=", tf_param.rot, "\ntranslation=", tf_param.t,"\nscale=", tf_param.scale);
	source_o3d.points = tf_param.transform(source_o3d.points)
	pcl = Points.Points()
	pcl_points = np.asarray(source_o3d.points).tolist()
	pcl.addPoints([tuple(x) for x in pcl_points])
	return pcl

def part_to_mesh(part, max_length=1):
	shp = Part.getShape(part)
	return shape_to_mesh(shp, max_length)

def shape_to_mesh(shp, max_length=1):
	mesh = MeshPart.meshFromShape(shp, MaxLength=max_length)
	return mesh

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
