import copy
import numpy as np
import open3d as o3
from probreg import cpd
"""
plydata_fix = PlyData.read('fix.ply')
plydata_mov = PlyData.read('mov.ply')

X_fix = np.array([plydata_fix['vertex']['x'], plydata_fix['vertex']['y'], plydata_fix['vertex']['z']])
X_mov = np.array([plydata_mov['vertex']['x'], plydata_mov['vertex']['y'], plydata_mov['vertex']['z']])
X_fix = np.transpose(X_fix)
X_mov = np.transpose(X_mov)

reg = RigidRegistration(X=X_fix, Y=X_mov)
result, (s_reg, R_reg, t_reg) = reg.register()
"""
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