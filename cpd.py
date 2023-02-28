import copy
import numpy as np
import open3d as o3
from probreg import cpd

print("Reading source point cloud.")
source = o3.io.read_point_cloud('mov.pcd')
source.remove_non_finite_points()
print("Reading target point cloud.")
target = o3.io.read_point_cloud('fix.pcd')
target.remove_non_finite_points()

print("Starting registration.")
tf_param, _, _ = cpd.registration_cpd(source, target)
print("Registration done.")
print("Copying result.")
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.visualization.draw_geometries([source, target, result])
