import open3d as o3d
import numpy as np


class Visualizer():
    def __init__(self, data_path) -> None:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            points = []
            for line in lines:
                value = line.strip().split(',')
                point = [value[i] for i in range(3)]
                points.append(point)  
            points = np.array(points)
            print(points.shape)
            # print(points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    data_path = './dataset/object_training_datasets/vehicle/test/020533.txt'
    visualizer = Visualizer(data_path)
