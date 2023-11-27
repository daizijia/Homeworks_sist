import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import Transforms
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
"""
problem:
    1. not converge
    2. do not know exactly the relation between delta_x and g
possible solution:
    1. change to minus -? meiyong
    2. use rotation vector 
"""
class BA:
    def __init__(self, K, points3D, points2D, r, t):
        """Intializes all the class attributes and instance variables.
            K is the intrinsic
            points_3d with shape (3, n_points)
                    contains initial estimates of point coordinates in the world frame.
            points_2d with shape (n_observations, 2, n_points)
                    contains measured 2-D coordinates of points projected on images in each observations.
            r,t is the rotation and transition of camera 
        """

        self.points3D = points3D
        self.points2D = points2D
        self.r = r
        self.t = t
        self.K = K
        self.m = 9
        self.n = 25
        self.Z = 2


    def project(self, r, t, points):
        """Convert 3-D points to 2-D by projecting onto images."""
        # Puv = K(RPw+t)
        R = Rotation.from_euler('xyz',r)
        R = R.as_matrix()
        world = R @ points.T + np.asarray(t).reshape(3,-1) 
        uv = self.K *(1/self.Z) @ world
        uv = uv[0:2]
        
        return uv.T

    def get_residual(self, xp, r, t):
        """Compute residuals.
        """
        point_imgs = self.points2D

        residual = []
        for i in range(self.m):#number of observations
            point_proj = self.project(r[i], t[i], xp)
            # print(r[i], t[i])
            point_imgs[i] = point_imgs[i].T
            for j in range(self.n):#number of points
                # print(point_imgs[i][j])
                # print(point_proj[j])
                res = np.square(point_proj[j][0] - point_imgs[i][j][0]) + \
                                 np.square(point_proj[j][1] - point_imgs[i][j][1])
                res = np.sqrt(res) 
                residual.append(res)
            point_imgs[i] = point_imgs[i].T
        residual = np.asarray(residual)
        print(np.sum(residual)/(25*9))
        return residual

    def get_FE(self,xpc,r):
        """
        the x,y,z is after transformation extrinc R,t ????
        
        F is 2x6
        E is 2x3
        """
        x, y, z = [i for i in xpc]
        fx = self.K[0][0]
        fy = self.K[1][1]
        R = Rotation.from_euler('xyz',r)
        R = R.as_matrix()
        #TODO:if error check here F need minus? 
        F = np.array([[fx/z, 0, -(fx*x)/(z*z), -(fx*x*y)/(z*z), fx + (fx*x*x)/(z*z), -(fx*y)/z],
                      [0, fy/z, -(fy*y)/(z*z), -fy-(fy*y*y)/(z*z), (fy*x*y)/(z*z), (fy*x)/z]])
        E = -np.array([[fx/z, 0, -(fx*x)/(z*z)],
                       [0, fy/z, -(fy*y)/(z*z)]]) @ R
        
        return F,E
    
    def get_Jacobian(self, xc, xp):
        """
        input:
        xc is camera pose 6xm m = 9
        xp is point in space 3xn n = 25
        output:
        Jij is 2x(9*6 + 25*3) only i and 9+j blocks is not none(F and E)
        J append Jij in order
        """
        J = []
        for i in range(self.m):
            r = xc[i][0:3]
            t = xc[i][3:]
            R = Rotation.from_euler('xyz',r)
            R = R.as_matrix()
            for j in range(self.n):
                Jij = np.zeros((2, (6*self.m+3*self.n)))
                xpc = R @ xp[j].T + t.T
                Fij,Eij = self.get_FE(xpc,r) 
                Jij[:,i*6 : i*6 + 6] = Fij                                #TODO:if error check here
                Jij[:,self.m*6 + j*3: self.m*6 + j*3 + 3] = Eij
                J.append(Jij)
        J = np.asarray(J)
        print("J.shape",J.shape)
        return J

    
    def get_deltaXcXp(self, Jacob, res):
        """
        1. get H=sum(Jij.t @ Jij)
        2. get g = -J.t @ res
        3. separate H  into B, E, C
        4. get delta_xc, delta_xp
        """
        H = Jacob[0].T @ Jacob[0]
        g = -Jacob[0].T * res[0]
        for i in range(1,self.m*self.n):
            H += Jacob[i].T @ Jacob[i]             #####
            g += -Jacob[i].T * res[i]

        print("H",H.shape)
        # g = -Jacob.T @ res

        # no Schur complement
        delta_x = np.linalg.inv(H) @ g
        delta_c = delta_x[0:self.m*6,:]
        delta_p = delta_x[self.m*6:,:]
        # print("delta_x",delta_x)

        # # Schur complement
        # B = H[0:6*self.m,0:6*self.m]
        # print("B",B.shape)
        # E = H[0:6*self.m,6*self.m:]
        # print("E",E.shape)
        # C = H[6*self.m:,6*self.m:]
        # print("C",C.shape)
        # inv_C = np.linalg.inv(C)
        # print("C",C)

        # v = g[0:self.m*6,:]
        # print("v",v.shape)
        # w = g[self.m*6:,:]
        # print("w",w.shape)
        # delta_c = np.linalg.inv(B - (E @ inv_C @ E.T)) @ (v - (E @ inv_C @ w))
        # delta_p = inv_C @ (w - (E.T @ delta_c))


        print("delta_c",delta_c.shape)
        print("delta_p",delta_p.shape)
        # print("delta_c",delta_c)
        # print("delta_p",delta_p)
        
        
        return delta_c, delta_p

    def change_form(self, xc,xp):

        c = []
        p = []
        for i in xc:
            c.append(np.sum(i))
        for j in xp:
            p.append(np.sum(j))
        c = np.asarray(c).reshape(-1,6)
        p = np.asarray(p).reshape(-1,3)

        # print(c)
        # print(p)
        return c,p

    def get_initXcXp(self):
        """
        xc is camera pose 6xm m = 9
        xp is point in space 3xn n = 25
        """
        xc,xp = [],[]
        point_world = self.points3D########
        for i in range(self.m):
            xc.append(np.hstack((self.r[i],self.t[i])))
        for j in range(self.n):
            xp.append(point_world[j])

        xc = np.asarray(xc)
        xp = np.asarray(xp)
        print("xc.shape",xc.shape)
        #print(xc)
        print("xp.shape",xp.shape)
        #print(xp)
        return xc,xp

    def updateXcXp(self, xc, xp, delta_xc, delta_xp):
        xp += delta_xp
        update_xc = []

        for i in range(self.m):

            delta_r = delta_xc[i][0:3]
            delta_t = delta_xc[i][3:]
            delta_R = Rotation.from_euler('xyz',delta_r)
            delta_R = delta_R.as_matrix()

            r = xc[i][0:3]
            t = xc[i][3:]
            R = Rotation.from_euler('xyz',r)
            R = R.as_matrix()

            update_R = delta_R @ R
            update_t = delta_R @ t.reshape(3,-1) + delta_t.reshape(3,-1)
            update_r = Rotation.from_matrix(update_R)
            update_r = update_r.as_euler('xyz')
            
            print(update_r)
            update_t = [update_t[0][0],update_t[1][0],update_t[2][0]]
            print(update_t)

            update_xc.append(np.hstack((update_r,update_t)))
        print(xp)

        return np.asarray(update_xc),xp

    def read_bal(self):
    
        """
        input:
        self
        output: 
        camera_index, point_index, cameras, points, observations
        """
        xc,xp = self.get_initXcXp()

        camera_index = [] # The camera index in each observation
        point_index = [] # The point index in each observation
        cameras = [] # Parameters of each camera (R(3x1 Rodrigues), t(3x1 Vector), f, k1, k2)
        #points = []  # 3D Coordinates of each point (x, y, z)
        observations = [] # 2D Coordinates of each pixel (x, y) 

        num_cams = self.m
        num_points = self.n
        num_observations = self.m * self.n
        focal = 500
        k1 = 0
        k2 = 0
        point_imgs = self.points2D
        #print(point_imgs)
        for i in range(num_cams):
            for j in range(num_points):
                camera_index.append(i)
                point_index.append(j)
                observations.append([point_imgs[i][j][0], point_imgs[i][j][1]]) ###############if error check here
        
        # print(camera_index)
        # print(point_index)
        # print(observations)
        
        for i in range(num_cams):
            cam_param = []
            for j in range(6):
                cam_param.append(xc[i][j])
            cam_param.append(focal)
            cam_param.append(k1)
            cam_param.append(k2)
            cameras.append(cam_param)
            
        # print(cameras)
 
        points = xp
        # print(points)
            
        return camera_index, point_index, cameras, points, observations
    
    def write_bal(self,camera_index, point_index, cameras, points, observations):
        Note=open('a.txt',mode='w')
        num_observations = 225
        for i in range(num_observations):
            Note.writelines([str(camera_index[i])," ",str(point_index[i]),"     ",str(observations[i][0])," ",str(observations[i][1])+" \n"])#####
        for i in range(9):
            for j in cameras[i]:
                Note.writelines([str(j)," \n"])
        for i in range(25):
            for j in points[i]:
                Note.writelines([str(j)," \n"])
        Note.close()
        pass
    
    def gauss_newton_BA_algebra(self):
        num_cameras = self.m
        num_points = self.n
        num_observations = self.m*self.n
        camera_index, point_index, cameras, points, observations = self.read_bal()
        self.write_bal(camera_index, point_index, cameras, points, observations)
        cameras = np.array(cameras, dtype=np.float64)
        points = np.array(points, dtype=np.float64)
        observations = np.array(observations, dtype=np.float64)
        
        epochs = 20
        for epoch in range(epochs):
            cost = 0
            # J = np.zeros((6*num_cameras+3*num_points, 2*num_observations), dtype=np.float64) # Cannot store dense jacobian like this -> 80GB memory needed
            J = csr_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),shape=(2*num_observations, 6*num_cameras+3*num_points))
            J = lil_matrix(J)
            b = np.zeros(6*num_cameras+3*num_points)
            for i in range(num_observations):
                observed = observations[i]
                print(observed)
                cam_idx = camera_index[i]
                pts_idx = point_index[i]
                camera = cameras[cam_idx]
                #print(camera)
                point = points[pts_idx]
                #print(point)
                projected = Transforms.project(camera, point)
                print(projected)
                
                # Calc Jacobian
                # Camera Jacobian
                fx = camera[6]
                fy = fx
                p3d = Transforms.transform(camera, point)
                X = p3d[0]
                Y = p3d[1]
                Z = p3d[2]
                Z2 = Z**2
                JT = -np.array([
                    [fx/Z, 0, -fx*X/Z2, -fx*X*Y/Z2, fx+fx*X**2/Z2, -fx*Y/Z],
                    [0, fy/Z, -fy*Y/Z2, -fy-fy*Y**2/Z2, fy*X*Y/Z2, fy*X/Z]
                    ])
                J[2*i : 2*i+2, cam_idx*6 : cam_idx*6+6] = JT
        
                # Point Jacobian
                JP = -np.array([
                    [fx/Z, 0, -fx*X/Z2],
                    [0, fy/Z, -fy*Y/Z2]
                    ])
                J[2*i : 2*i+2, num_cameras*6 + pts_idx*3 : num_cameras*6 + pts_idx*3 + 3] = JP
                residual = observed - projected
                b[cam_idx*6 : cam_idx*6 + 6] += -np.dot(JT.T, residual)
                b[num_cameras*6 + pts_idx*3 : num_cameras*6 + pts_idx*3 + 3] += -np.dot(JP.T, residual)
                cost += np.linalg.norm(residual)
            print(cost)
            break
            H = J.T @ J
            print("H",H.shape)
            print("b",b.shape)
            update = spsolve(H, b)
            print("update",update.shape)
            cam_update = np.reshape(update[0:6*num_cameras], (num_cameras, 6))
            print("cam_update",cam_update)
            pts_update = np.reshape(update[6*num_cameras:], (num_points, 3))
            cameras[:,0:6] += cam_update
            points += pts_update
            print(f"Epoch {epoch} Total cost:", cost)
              

    def gauss_newton_BA(self):
        """
        input: 

        """
        numCameras = 9
        numPoints = 25
        threhold = 0.5

        xc, xp = self.get_initXcXp()

        res = 100
        while(np.average(res) > threhold):

            Jacob = self.get_Jacobian(xc,xp)
            # print(xc[:,0:3])
            res = self.get_residual(xp, xc[:,0:3], xc[:,3:])
            delta_c, delta_p = self.get_deltaXcXp(Jacob,res)
            delta_xc, delta_xp = self.change_form(delta_c, delta_p)
            
            xc, xp = self.updateXcXp(xc, xp, delta_xc, delta_xp)
            #print(xc)
            
            #print(xp)
            pass

        pass

from scipy.optimize import least_squares

class SBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices):

        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

    def rotate(self, points, rot_vecs):
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = 500
        k1 = 0
        k2 = 0
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 6 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(6):
            A[2 * i, cameraIndices * 6 + s] = 1
            A[2 * i + 1, cameraIndices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        # print(x0)
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)
        print(f0)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

        camera_params, points_3d = self.optimizedParams(res.x, numCameras, numPoints)
        return camera_params, points_3d
    
def visualize_world(points_3d,point_world_noise):
    fig = plt.figure()
    
    ax = plt.axes()
    points_3d = points_3d.T
    z1 = points_3d[2]
    x1 = points_3d[0]
    y1 = points_3d[1]
    ax.scatter(x1, y1, c="b")

    point_world_noise = point_world_noise.T
    z2 = point_world_noise[2]
    x2 = point_world_noise[0]
    y2 = point_world_noise[1]
    ax.scatter(x2, y2, c="r")

    # ax.set_title('3d Scatter plot')
    # plt.show()
    plt.savefig("2.png")
    pass


