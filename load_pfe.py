# Macro Begin: /home/tom/.var/app/org.freecadweb.FreeCAD/data/FreeCAD/Macro/load_pfe.FCMacro +++++++++++++++++++++++++++++++++++++++++++++++++
# exec macro ctrl f6
import FreeCAD
import Points
import ImportGui
import Part
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
class pfe:
	def load_example():
		docName = "test"
		try:
			App.closeDocument(docName)
		except Exception:
			pass

		App.newDocument(docName)
		doc = App.getDocument(docName)

		git_dirpath = dir_path
		print(git_dirpath)

		Points.insert(os.path.join(git_dirpath, 'step_files/nuage_pts_test_cube.ply'), docName)
		ImportGui.insert(os.path.join(git_dirpath, 'step_files/test_cube.step'), docName)

		pts = doc.getObject("nuage_pts_test_cube").Points.Points
		cube = doc.getObject("Part__Feature").Shape

	def select_part_cloud():
		selection = Gui.Selection.getSelection()
		if len(selection) == 2:
			if type(selection[0]) is App.GeoFeature and type(selection[1]) is Part.Feature:
				cloud = selection[0]
				object = selection[1]
			elif type(selection[1]) is App.GeoFeature and type(selection[0]) is Part.Feature:
				cloud = selection[1]
				object = selection[0]
			else:
				print("WRONG ARGUMENTS should be App.GeoFeature and Part.Feature")
				return None, None
		else:
			print("TOO FEW ARGUMENTS should be App.GeoFeature and Part.Feature")
			return None, None
		shape = object.Shape
		pts = cloud.Points.Points
		return shape, pts

	def select_part_mesh():
		selection = Gui.Selection.getSelection()
		if len(selection) == 2:
			if type(selection[0]) is Part.Feature and selection[1].TypeId == 'Mesh::Feature':
				part_obj = selection[0]
				object = selection[1]
			elif type(selection[1]) is Part.Feature and selection[0].TypeId == 'Mesh::Feature':
				part_obj = selection[1]
				object = selection[0]
			else:
				print("WRONG ARGUMENTS should be Mesh.Feature and Part.Feature")
				return None, None
		else:
			print("TOO FEW ARGUMENTS should be Mesh.Feature and Part.Feature")
			return None, None
		mesh = object.Mesh
		part = part_obj.Shape
		return mesh, part

	@staticmethod
	def compute_distances():
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None
		return pfe.icompute_distances(shape, pts)

	@staticmethod
	def icompute_distances(shape, pts):
		dsts = []
		n_pts = len(pts)

		start = time.time()

		for i in range(n_pts):
			pt = Part.Vertex(pts[i])
			dst = pt.distToShape(shape)
			dsts.append(dst)

		end = time.time()
		print(" comp : " + str(n_pts) + " , " + str(end - start))
		return dsts

	@staticmethod
	def distance_map():
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None
		pfe.idistance_map(shape, pts)

	@staticmethod
	def idistance_map(shape, pts):

		dsts = pfe.icompute_distances(shape, pts)
		avg = sum([dst[0] for dst in dsts]) / len(dsts)
		n_pts = len(dsts)

		on = []
		close = []
		medium = []
		far = []
		for i in range(n_pts):
			d = [dst[0] for dst in dsts][i]
			if d > avg * 2:
				far.append(pts[i])
			elif d > avg:
				medium.append(pts[i])
			elif d > avg / 2:
				close.append(pts[i])
			else:
				on.append(pts[i])

		doc = App.ActiveDocument
		matches = doc.addObject('App::Part', 'distance_map')
		on_pts = Points.Points()
		close_pts = Points.Points()
		medium_pts = Points.Points()
		far_pts = Points.Points()
		on_pts.addPoints(on)
		close_pts.addPoints(close)
		medium_pts.addPoints(medium)
		far_pts.addPoints(far)

		# TODO mettre dans un part
		on = doc.addObject("Points::Feature", "on")
		on.Points = on_pts
		close = doc.addObject("Points::Feature", "close")
		close.Points = close_pts
		medium = doc.addObject("Points::Feature", "medium")
		medium.Points = medium_pts
		far = doc.addObject("Points::Feature", "far")
		far.Points = far_pts

		on.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
		close.ViewObject.ShapeColor = (0.0, 0.0, 1.00)
		medium.ViewObject.ShapeColor = (1.0, 1.0, 0.15)
		far.ViewObject.ShapeColor = (1.0, 0.0, 0.0)

	def distance_map_knn(k=3):
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None

		dsts = pfe.compute_distances()
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

		labels = pfe.knn(pts, labels, k);
		on_i = [i for i, label in enumerate(labels) if label == 'on']
		on = [pts[i] for i in on_i]
		close_i = [i for i, label in enumerate(labels) if label == 'close']
		close = [pts[i] for i in close_i]
		medium_i = [i for i, label in enumerate(labels) if label == 'medium']
		medium = [pts[i] for i in medium_i]
		far_i = [i for i, label in enumerate(labels) if label == 'far']
		far = [pts[i] for i in far_i]

		doc = App.ActiveDocument
		matches = doc.addObject('App::Part', 'distance_map')
		on_pts = Points.Points()
		close_pts = Points.Points()
		medium_pts = Points.Points()
		far_pts = Points.Points()
		on_pts.addPoints(on)
		close_pts.addPoints(close)
		medium_pts.addPoints(medium)
		far_pts.addPoints(far)

		# TODO mettre dans un part
		on = doc.addObject("Points::Feature", "on")
		on.Points = on_pts
		close = doc.addObject("Points::Feature", "close")
		close.Points = close_pts
		medium = doc.addObject("Points::Feature", "medium")
		medium.Points = medium_pts
		far = doc.addObject("Points::Feature", "far")
		far.Points = far_pts

		on.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
		close.ViewObject.ShapeColor = (0.0, 0.0, 1.00)
		medium.ViewObject.ShapeColor = (1.0, 1.0, 0.15)
		far.ViewObject.ShapeColor = (1.0, 0.0, 0.0)

	def knn(pts, labels, k = 3):
		print("knn with k =", k)
		# computes distance between each point, then sort by closest
		D = distance.squareform(distance.pdist(pts))
		closest = np.argsort(D, axis=1)
		kclosest = closest[:, 1:k+1]
		for p_i in range(len(pts)):
			neighbour_labels = [labels[i] for i in kclosest[p_i, :]] # get the labels of all nearest neighbours
			c = Counter(neighbour_labels)
			true_label = c.most_common(1)[0][0] # get most common label
			labels[p_i] = true_label
		return labels
			
	def feature_matching_growing_bb():
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None

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

	def feature_matching_bb():
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None

		pt_matched = [False for p in range(len(pts))]
		faceIdx_pts_dict = {}
		scale = 1.0
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
		not_matched = [pts[i] for i in range(len(pts)) if not pt_matched[i]]
		faceIdx_pts_dict[-1] = not_matched

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

	def feature_matching_to_closest_bb():
		shape, pts = pfe.select_part_cloud()
		if shape == None: return None
		
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

	def bruitage():
		selection = Gui.Selection.getSelection()
		if len(selection) == 1:
			if type(selection[0]) is App.GeoFeature:
				cloud = selection[0]
			else:
				print("WRONG ARGUMENT should be App.GeoFeature")
				return
		else:
			print("TOO MANY ARGUMENTS should be App.GeoFeature")
			return

		shp = cloud.Points

		tmp = [0 for x in range(shp.CountPoints)]
		for i in range(shp.CountPoints):
			rand = np.array([random.random(), random.random(), random.random()]) * np.sign(cloud.Normal[i]) * 2
			displacementVector = FreeCAD.Vector(rand[0], rand[1], rand[2])
			tmp[i] = shp.Points[i] + displacementVector
		noise = Points.Points()
		noise.addPoints(shp.Points)
		noise.addPoints(tmp)

		doc = App.ActiveDocument
		noiseObj = doc.addObject("Points::Feature", "noisyObj")
		noiseObj.Points = noise

	def bruit_gaussien():
		selection = Gui.Selection.getSelection()
		if len(selection) == 1:
			if type(selection[0]) is App.GeoFeature:
				cloud = selection[0]
			else:
				print("WRONG ARGUMENT should be App.GeoFeature")
				return
		else:
			print("TOO MANY ARGUMENTS should be App.GeoFeature")
			return

		shp = cloud.Points
		gaussNoise = np.random.randn(shp.CountPoints, 3)
		tmp = [0 for x in range(shp.CountPoints)]
		for i in range(shp.CountPoints):
			tmp[i] = shp.Points[i] + FreeCAD.Vector(gaussNoise[i][0], gaussNoise[i][1], gaussNoise[i][2])
		noise = Points.Points()
		noise.addPoints(shp.Points)
		noise.addPoints(tmp)

		doc = App.ActiveDocument
		noiseObj = doc.addObject("Points::Feature", "noisyObj")
		noiseObj.Points = noise

	def feature_matching_dst():
		shape, pts = pfe.select_part_cloud()
		if shape is None: return None

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

	# isoler face dans un compound
	# doc.addObject("Part::Compound","as")
	# ttt = Part.makeCompound(sub)
	# obj.Shape = ttt
	# distances
	def fit_mesh_to_part():
		mesh, part = pfe.select_part_mesh()
		if mesh is None: return
		n_pts = len(mesh.Points)
		for pt_index in range(n_pts):
			pt = mesh.Points[pt_index]
			part_vertex = Part.Vertex(pt.Vector)
			d = part_vertex.distToShape(part)
			pt.move(d[1][0][1] - d[1][0][0])

	@staticmethod
	def cloud_to_numpy(cloud):
		return np.array(cloud.Points.Points)

	@staticmethod
	def numpy_to_open3d_cloud(array):
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(array)
		return pcd


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
