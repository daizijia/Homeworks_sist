import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# part B
class KFilter(object):
    """
    Formula:
    x(k+1) = F(k)x(k) + G(k)u(k) + v(k)
    y(k) = H(k)x(k) + w(k)
    x(k) is the n-dimensional state vector (unknown)
    u(k) is the m-dimensional input vector (known)
    y(k) is the p-dimensional output vector (known, measured)
    F(k), G(k), H(k) are appropriately dimensioned system matrices (known)
    P(k) is the covariance matrix (unknown)
    v(k), w(k) are zero-mean, white Gaussian noise with (known)
            covariance matrices Q(k), R(k)
    """
    def __init__(self, x0 = None):

        self.F = np.array([[1.0, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, 0, 1.0, 0],
                            [0, 0, 0, 1.0]])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],#####TODO:[0,0,1,0]
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
        self.G = 0.0

        self.Q = np.diag([1.0, 1.0, 1.0, 1.0]) ** 2
        self.R = np.array([[0.05,0,0,0],
                        [0,0.05,0,0],
                        [0,0,0.02,0],
                        [0,0,0,0.05]])
        self.n = 4
        self.P = np.eye(self.n) 
        self.x = np.zeros((self.n, 1)) if x0 is None else x0


    def kf_estimation(self, z, u = 0):
        
        # predict
        self.x = self.F @ self.x# + self.G @ u
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.R

        # update
        print(z)
        y = z - np.dot(self.H, self.x)
        S = self.Q + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.Q), K.T)
        
        return self.x, self.P


    

#part A
class EKFilter(object):
    """
    Motion model:
    x[t] = Ax[t-1] + Bu[t]
    x[t] = [x[t], y[t], theta[t], v[t]] is the state vector
    u[t] = [v[t], w[t]] is the control vector
    Observation model:
    z[t] = Cx[t]
    z[t] = [x[t], y[t]] is the observation model
    input:
    ud odometry, z landmark
    """
    def __init__(self, dt = None, xEst = None):
        
        self.xEst = np.zeros((4, 1)) if xEst is None else xEst
        self.PEst = np.eye(4)

        self.Q = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance
        # self.R = np.diag([
        #         0.1,  #  x-axis
        #         0.1,  #  y-axis
        #         np.deg2rad(1.0),  # yaw angle
        #         1.0  # velocity
        #     ]) ** 2  # predict state covariance
        self.R = np.array([[0.01,0,0,0],
                        [0,0.01,0,0],
                        [0,0,0.0002,0],
                        [0,0,0,5.0]]) 
        
        #  Simulation parameter
        self.dt = 0.1 if dt is None else dt # time tick [s]
        pass
    
    def motion_model(self, x, u):
        A = np.array([[1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, 0],
                    [0, 0, 0, 0]])

        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                    [self.dt * math.sin(x[2, 0]), 0],
                    [0.0, self.dt],
                    [1.0, 0.0]])
        
        x = A @ x + B @ u
        return x
    
    def observation_model(self,x):
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = C @ x
        return z
    
    def jacob_f(self, x, u):
        yaw = x[2, 0]
        v = u[0, 0]
        G = np.array([
            [1.0, 0.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
            [0.0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return G

    def jacob_h(self):
        # Jacobian of Observation Model
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return C
    
    def ekf_estimation(self,u,z):
        #  Predict
        xPred = self.motion_model(self.xEst, u)
        G = self.jacob_f(self.xEst, u)
        PPred = G @ self.PEst @ G.T + self.R

        #  Update
        C = self.jacob_h()
        zPred = self.observation_model(xPred)
        S = C @ PPred @ C.T + self.Q
        K = PPred @ C.T @ np.linalg.inv(S)
        self.xEst = xPred + K @ (z - zPred)
        self.PEst = (np.eye(len(self.xEst)) - K @ C) @ PPred

        return self.xEst, self.PEst
    
    pass