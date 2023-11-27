import numpy as np
import time
from collections import Counter
import open3d as o3d
import glob
import os

class Node():
    def __init__(self, data, label, depth=0, lchild=None, rchild=None):
        self.data, self.label, self.depth, self.lchild, self.rchild = \
            (data, label, depth, lchild, rchild)


class KdTree():
    def __init__(self, X, y):
        y = y[np.newaxis, :]
        self.data = np.hstack([X, y.T])

        self.rootNode = self.buildTree(self.data)

    def buildTree(self, data, depth=0):
        if(len(data) <= 0):
            return None

        m, self.n = np.shape(data)
        aim_axis = depth % (self.n-1)

        sorted_data = sorted(data, key=lambda item: item[aim_axis])
        mid = m // 2
        node = Node(sorted_data[mid][:-1], sorted_data[mid][-1], depth=depth)

        if(depth == 0):
            self.kdTree = node

        node.lchild = self.buildTree(sorted_data[:mid], depth=depth+1)
        node.rchild = self.buildTree(sorted_data[mid+1:], depth=depth+1)
        return node

    def preOrder(self, node):
        if node is None:
            return
        print("(", node.data, node.label, ')', ':', node.depth)
        self.preOrder(node.lchild)
        self.preOrder(node.rchild)

    def search(self, x, count=1):
        nearest = []
        assert count >= 1 and count <= len(self.data), '错误的k近邻值'

        self.nearest = nearest

        def recurve(node):
            if node is None:
                return
            now_axis = node.depth % (self.n - 1)

            if(x[now_axis] < node.data[now_axis]):
                recurve(node.lchild)
            else:
                recurve(node.rchild)
            dist = np.linalg.norm(x - node.data, ord=2)

            if(len(self.nearest) < count):
                self.nearest.append([dist, node])
            else:
                aim_index = -1
                for i, d in enumerate(self.nearest):
                    if(d[0] < 0 or dist < d[0]):
                        aim_index = i
                if(aim_index != -1):
                    self.nearest[aim_index] = [dist, node]

            max_index = np.argmax(np.array(self.nearest)[:, 0])
           
            if(self.nearest[max_index][0] > abs(x[now_axis] - node.data[now_axis])):
                if(x[now_axis] - node.data[now_axis] < 0):
                    recurve(node.rchild)
                else:
                    recurve(node.lchild)

        recurve(self.rootNode)
        poll = [int(item[-1].label) for item in self.nearest]

        return self.nearest, poll#Counter(poll).most_common()[0][0]


class KNNKdTree():
    def __init__(self, n_neighbors=3):
        self.k = n_neighbors

    def fit(self, X_train, y_train = None):
        self.X_train = np.array(X_train)
        if y_train is None:
            self.y_train = np.array([i for i in range(X_train.shape[0])])
        self.kdTree = KdTree(self.X_train, self.y_train)

    def predict(self, x):
        nearest, label = self.kdTree.search(x, self.k)
        return nearest, label

    def kneighbors(self,X):
        labels = []
        nearests = []
        for i in range(X.shape[0]):
            nearest, label = self.kdTree.search(X[i], self.k)
            labels.append(label)
            nearests.append(nearest[0][0])
        return np.asarray(nearests),np.asarray(labels).T


def main():
    filepath = "/home/mpl/Desktop/slam_hw/hw1/datas/voxel_0.3"
    files = sorted(glob.glob(os.path.join(filepath, "*.xyz")))
    s = o3d.io.read_point_cloud(files[4])
    d = o3d.io.read_point_cloud(files[5])
    src = np.asarray(s.points)
    dst = np.asarray(d.points)
    num = min(src.shape[0], dst.shape[0])
    src = src[:num,:3]
    dst = dst[:num,:3]

    knn_kdTree = KNNKdTree(n_neighbors=4)
    knn_kdTree.fit(dst)
    nn,labels = knn_kdTree.kneighbors(src)
    
    print(labels)
    print(np.asarray(nn))




if __name__ == "__main__":
    main()