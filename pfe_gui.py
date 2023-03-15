# exec macro ctrl f6
import FreeCAD
import Points
import ImportGui
import Part
import os
import sys
from PySide import QtGui

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(dir_path))

try:
    del sys.modules['pfe_standalone']
except:
    pass

from pfe_standalone import *
class pfe:
	@staticmethod
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

		# pts = doc.getObject("nuage_pts_test_cube").Points.Points
		# cube = doc.getObject("Part__Feature").Shape

	@staticmethod
	def select_cloud_obj():
		cloud_obj = Gui.Selection.getSelection()
		if len(cloud_obj) == 1:
			if type(cloud_obj[0]) is App.GeoFeature:
				cloud = cloud_obj[0]
			else:
				print("WRONG ARGUMENT should be App.GeoFeature")
				return None
		else:
			print("TOO MANY OR NO ARGUMENTS : ", len(cloud_obj), " should be App.GeoFeature")
			return None
		return cloud_obj

	@staticmethod
	def select_cloud():
		return pfe.select_cloud_obj().Points

	@staticmethod
	def select_part_obj():
		part_obj = Gui.Selection.getSelection()
		if len(part_obj) == 1:
			if type(part_obj[0]) is Part.Feature:
				cloud = part_obj[0]
			else:
				print("WRONG ARGUMENT should be Part.Feature")
				return None
		else:
			print("TOO MANY OR NO ARGUMENTS : ", len(part_obj), " should be Part.Feature")
			return None
		return part_obj

	@staticmethod
	def select_part():
		return pfe.select_part_obj().Shape
	@staticmethod
	def select_part_cloud_objs():
		selection = Gui.Selection.getSelection()
		if len(selection) == 2:
			if type(selection[0]) is App.GeoFeature and type(selection[1]) is Part.Feature:
				cloud = selection[0]
				part = selection[1]
			elif type(selection[1]) is App.GeoFeature and type(selection[0]) is Part.Feature:
				cloud = selection[1]
				part = selection[0]
			else:
				print("WRONG ARGUMENTS should be App.GeoFeature and Part.Feature")
				return None, None
		else:
			print("TOO FEW OR TOO MUCH ARGUMENTS : ", len(selection), " should be 2 of type App.GeoFeature and Part.Feature")
			return None, None
		return part, cloud

	@staticmethod
	def select_part_cloud():
		part, cloud = pfe.select_part_cloud_objs()
		part = part.Shape
		pts = cloud.Points.Points
		return part, pts

	@staticmethod
	def select_part_mesh_objs():
		selection = Gui.Selection.getSelection()
		if len(selection) == 2:
			if type(selection[0]) is Part.Feature and selection[1].TypeId == 'Mesh::Feature':
				part_obj = selection[0]
				mesh_obj = selection[1]
			elif type(selection[1]) is Part.Feature and selection[0].TypeId == 'Mesh::Feature':
				part_obj = selection[1]
				mesh_obj = selection[0]
			else:
				print("WRONG ARGUMENTS should be Mesh.Feature and Part.Feature")
				return None, None
		else:
			print("TOO FEW OR TOO MUCH ARGUMENTS : ", len(selection), " should be Mesh.Feature and Part.Feature")
			return None, None
		return part_obj, mesh_obj

	@staticmethod
	def select_part_mesh():
		part_obj, mesh_obj = pfe.select_part_mesh_objs()
		mesh = mesh_obj.Mesh
		part = part_obj.Shape
		return mesh, part

	@staticmethod
	def compute_distances():
		part, pts = pfe.select_part_cloud()
		if part is None:
			return None

		return icompute_distances(part, pts)

	@staticmethod
	def distance_map_base(idistance_map_fct):
		shape_obj, pts_obj = pfe.select_part_cloud_objs()
		part, pts = pfe.select_part_cloud()
		if part is None or pts is None:
			return None
		on, close, medium, far = idistance_map_fct(part, pts)

		pts_obj.ViewObject.Visibility = False  # hide previous point cloud
		doc = App.ActiveDocument

		on_pts = Points.Points()
		close_pts = Points.Points()
		medium_pts = Points.Points()
		far_pts = Points.Points()
		on_pts.addPoints(on)
		close_pts.addPoints(close)
		medium_pts.addPoints(medium)
		far_pts.addPoints(far)

		dmap = doc.addObject('App::Part', 'distance_map')
		on = doc.addObject("Points::Feature", "on")
		on.Points = on_pts
		on.adjustRelativeLinks(dmap)
		dmap.addObject(on)

		close = doc.addObject("Points::Feature", "close")
		close.Points = close_pts
		close.adjustRelativeLinks(dmap)
		dmap.addObject(close)

		medium = doc.addObject("Points::Feature", "medium")
		medium.Points = medium_pts
		medium.adjustRelativeLinks(dmap)
		dmap.addObject(medium)

		far = doc.addObject("Points::Feature", "far")
		far.Points = far_pts
		far.adjustRelativeLinks(dmap)
		dmap.addObject(far)

		on.ViewObject.ShapeColor = (0.0, 1.0, 0.0)
		close.ViewObject.ShapeColor = (0.0, 0.0, 1.00)
		medium.ViewObject.ShapeColor = (1.0, 1.0, 0.15)
		far.ViewObject.ShapeColor = (1.0, 0.0, 0.0)
	@staticmethod
	def distance_map():
		pfe.distance_map_base(idistance_map)
	@staticmethod
	def distance_map_knn():
		pfe.distance_map_base(idistance_map_knn)

	@staticmethod
	def feature_matching_base(ifeature_matching_fct):
		part, pts = pfe.select_part_cloud()
		if part is None or pts is None:
			return None

		faceIdx_pts_dict = ifeature_matching_fct(part, pts)

		doc = App.ActiveDocument
		matches = doc.addObject('App::Part', 'features_matches')
		for face_index, points in faceIdx_pts_dict.items():
			feature_pts = Points.Points()
			feature_pts.addPoints(points)
			fm_pts = doc.addObject("Points::Feature", "Face" + str(face_index))
			fm_pts.adjustRelativeLinks(matches)
			matches.addObject(fm_pts)
			fm_pts.Points = feature_pts

		return faceIdx_pts_dict

	@staticmethod
	def feature_matching_base_bis(ifeature_matching_fct):
		part, pts = pfe.select_part_cloud()
		if part is None or pts is None:
			return None

		face_idx_pts_idx, not_matched_pt_idx = ifeature_matching_fct(part, pts)

		doc = App.ActiveDocument
		matches = doc.addObject('App::Part', 'features_matches')
		for face_index in range(len(face_idx_pts_idx)):
			points = [pts[i] for i in face_idx_pts_idx[face_index]]
			feature_pts = Points.Points()
			feature_pts.addPoints(points)
			fm_pts = doc.addObject("Points::Feature", "Face" + str(face_index))
			fm_pts.adjustRelativeLinks(matches)
			matches.addObject(fm_pts)
			fm_pts.Points = feature_pts

		return face_idx_pts_idx, not_matched_pt_idx

	@staticmethod
	def feature_matching_bb():
		return pfe.feature_matching_base_bis(ifeature_matching_bb)

	@staticmethod
	def feature_matching_growing_bb():
		return pfe.feature_matching_base(ifeature_matching_growing_bb)

	@staticmethod
	def feature_matching_to_closest_bb():
		return pfe.feature_matching_base(ifeature_matching_to_closest_bb)

	@staticmethod
	def feature_matching_dst():
		return pfe.feature_matching_base_bis(ifeature_matching_dst)

	@staticmethod
	def feature_matching_optimized():
		return pfe.feature_matching_base_bis(ifeature_matching_optimized)
	@staticmethod
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

		noise = bruitage(cloud.Points)
		doc = App.ActiveDocument
		noiseObj = doc.addObject("Points::Feature", "noisyObj")
		noiseObj.Points = noise

	@staticmethod
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

		noise = bruit_gaussien(cloud.Points)
		doc = App.ActiveDocument
		noiseObj = doc.addObject("Points::Feature", "noisyObj")
		noiseObj.Points = noise

	@staticmethod
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
	def cpd():
		selection = Gui.Selection.getSelection()
		if len(selection) == 2:
			if type(selection[0]) is App.GeoFeature and type(selection[1]) is App.GeoFeature:
				source = selection[0]
				target = selection[1]
				pcl = icpd(source, target)
			else:
				print("WRONG ARGUMENTS should be App.GeoFeature and App.GeoFeature")
				return None
		else:
			print("TOO FEW ARGUMENTS should be App.GeoFeature and App.GeoFeature")
			return None

		doc = App.ActiveDocument
		result = doc.addObject("Points::Feature", "result")
		result.Points = pcl

	@staticmethod
	def cmp_fmap(true_fmatch, fmatch):
		sum = 0
		for i in range(len(true_fmatch)):
			sum += abs(len(true_fmatch[i]) - len(fmatch[i]))
		print(sum)
