from mayavi import mlab
import numpy as np
import pyvista as pv
import open3d as o3d
import numpy as np


point = np.loadtxt('cmake-build-debug/point.txt', delimiter=' ')
#surface = np.loadtxt('cmake-build-debug/surface.txt', dtype=np.int_, delimiter=' ')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point)

pcd.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries([pcd])






