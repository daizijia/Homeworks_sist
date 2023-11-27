import numpy as np
import struct
import open3d as o3d

def bin_to_pcd(indir,outdir):
    size_float = 4
    list_pcd = []
    with open(indir, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    o3d.io.write_point_cloud(outdir, pcd)

indir = '/home/daizj/Homeworks/cg/Proj/proj/data/training/velodyne/000007.bin'
outdir = 'visual/000007.pcd'

bin_to_pcd(indir, outdir)