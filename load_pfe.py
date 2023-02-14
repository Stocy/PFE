# -*- coding: utf-8 -*-

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
from pathlib import Path

class pfe:
	def load_example():
		docName = "test"
		try:
			App.closeDocument(docName)
		except Exception:
			pass

		App.newDocument(docName)
		doc = App.getDocument(docName)

		git_dirpath = "/home/tom/Documents/cours/pfe/project/PFE"
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

	def compute_distances():
		shape, pts = select_part_cloud()
		if shape == None: return None
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

	def distance_map():
		shape, pts = select_part_cloud()
		if shape == None: return None

		dsts = compute_distances()
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

	def feature_matching_bb():
		shape, pts = select_part_cloud()
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
		shape, pts = select_part_cloud()
		if shape is None: return None

		shape = object.Shape
		pts = cloud.Points.Points

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
		mesh, part = select_part_mesh()
		if mesh is None: return
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
