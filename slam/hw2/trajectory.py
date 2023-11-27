import numpy as np
from matplotlib import pyplot as plt
from math import cos,sin, atan2
from KalmanFilter import *


def read_txt(filepath):
    
    with open(filepath,'r',encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip('\n')
            data_line = line.split()
            data.append([float(i) for i in data_line])
    return np.asarray(data)


def visualize_trajectory(x, y, theta = None):

    length = 5
    lines=np.zeros(x.shape)+4
    for i in range(theta.shape[0]-1):

        dx = length*cos(theta[i]+np.pi/2)
        dy = length*sin(theta[i]+np.pi/2)

        plt.arrow(x[i], y[i], dx, dy, color = 'k', head_length = 0.05, head_width = 2)

    plt.scatter(x,y,s=5,c='b',alpha=0.5,linewidths=lines)
    plt.scatter(x[0],y[0],s=20,c='r',alpha=0.5,linewidths=lines,label='start')
    plt.scatter(x[-1],y[-1],s=20,c='g',alpha=0.5,linewidths=lines,label='end')
    plt.grid(True)
    plt.show()
    
    pass

def get_trajectory(rot1, trans, rot2, x_init, y_init, theta_init):
    """input odometry 
       output x,y,theta
    """
    x_t = np.zeros(rot1.shape[0])
    y_t = np.zeros(rot1.shape[0])
    theta_t = np.zeros(rot1.shape[0])
    for i in range(rot1.shape[0]):
        if i == 0:
            x_t[i] = x_init
            y_t[i] = y_init
            theta_t[i] = theta_init
        else:
            x_t[i] = x_t[i-1] + trans[i-1] * cos(theta_t[i-1] +rot1[i-1])
            y_t[i] = y_t[i-1] + trans[i-1] * sin(theta_t[i-1] +rot1[i-1])
            theta_t[i] = theta_t[i-1]+ rot1[i-1] +rot2[i-1]
    return x_t, y_t, theta_t

def get_velocity_yawrate1(x, y, theta, dt = 0.1):

    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    ulist = []
    for i in range(x.shape[0]-1):
        v = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) / dt
        yawrate = (theta[i+1]-theta[i]) / dt
        u = np.array([[theta[i]], [v]])
        #u = np.array([[v], [yawrate]])
        ulist.append(u)
    ulist.append(u)
    return np.asarray(ulist)

def get_velocity_yawrate2(rot1, trans, rot2, dt = 0.1):

    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    ulist = []
    for i in range(rot1.shape[0]):
        v = trans[i] / dt
        yawrate = (rot1[i] + rot2[i]) / dt ###### may be wrong
        u = np.array([[v], [yawrate]])
        ulist.append(u)
    return np.asarray(ulist)

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

def main_A(A_data, od_data):

    x_t, y_t, _ = get_trajectory(od_data[0],od_data[1],od_data[2],
                   A_data[0][0], A_data[1][0], A_data[2][0])
    zlist = A_data[0:2].T
    ulist = get_velocity_yawrate2(od_data[0],od_data[1],od_data[2], dt = 0.1)
    xpred = []
    ppred = []

    # history
    hxEst = np.asarray([[x_t[0]],[y_t[0]],[0],[0]])
    hxDR = np.asarray([[x_t[0]],[y_t[0]],[0],[0]])
    hz = np.asarray([[x_t[0]],[y_t[0]]])

    ekf = EKFilter(dt = 0.1, xEst = hxEst)

    for i in range(A_data.shape[1]-1): 
        # print(zlist[i].reshape(2,1))
        z = zlist[i].reshape(2,1)
        x, p = ekf.ekf_estimation(ulist[i], z)
        xpred.append(x)
        ppred.append(p)

        # store data history
        xDR = np.asarray([[x_t[i]],[y_t[i]],[0],[0]])
        hxEst = np.hstack((hxEst, x))
        hxDR = np.hstack((hxDR, xDR))
        hz = np.hstack((hz, z))

        # plot 
        length = 0.3
        dx = length*cos(x[2][0]-np.pi)
        dy = length*sin(x[2][0]-np.pi)
        
        #plt.cla()
        plt.plot(hz[0, :], hz[1, :], ".b")
        plt.plot(hxDR[0, :].flatten(),
                hxDR[1, :].flatten(), "-g")
        plt.plot(hxEst[0, :].flatten(),
                hxEst[1, :].flatten(), "-r")
        plt.arrow(x[0][0], x[1][0], dx, dy, color = 'k', head_length = 0.05, head_width = 0.02)
        # plot_covariance_ellipse(x, p)
        plt.grid(True)
    
    plt.show()  

    # return np.asarray(xpred), np.asarray(ppred)


def main_B(B_data):

    zlist = B_data[0:2].T
    vlist = get_velocity_yawrate1(B_data[0],B_data[1],B_data[2], dt = 0.1)
    
    xpred = []
    ppred = []

    x0 = np.asarray([[B_data[0][0]], [B_data[1][0]],[0.0],[0.0]])

    # history
    hxEst = x0
    hz = np.asarray([[B_data[0][0]], [B_data[1][0]],[0.0],[0.0]])

    kf1 = KFilter(x0 = x0)

    for i in range(B_data.shape[1]-1): 
        z = zlist[i].reshape(2,1)
        z = np.vstack((z, vlist[i]))

        x, p = kf1.kf_estimation(z)
        xpred.append(x)
        ppred.append(p)

        # store data history
        hxEst = np.hstack((hxEst, x))
        hz = np.hstack((hz, z))

        # plot 
        length = 5
        dx = length*cos(x[2][0]-np.pi)
        dy = length*sin(x[2][0]-np.pi)

        #plt.cla()
        plt.plot(hz[0, :], hz[1, :], ".b")
        plt.plot(hxEst[0, :].flatten(),
                hxEst[1, :].flatten(), "-r")
        plt.arrow(x[0][0], x[1][0], dx, dy, color = 'k', head_length = 0.05, head_width = 2)
        # plot_covariance_ellipse(x, p)
        plt.grid(True)
    plt.show()

    return np.asarray(xpred)
    
    pass

def plt_velocity(x,v,label='velocity'):
    # plt.switch_backend('Agg')
    plt.figure()

    plt.plot(x,v, ".b",linewidth = 1,linestyle = 'solid',label = label)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('speed')
    plt.ylim(-5,5)
    plt.show()
    pass

if __name__ == '__main__':
    A_path = 'datas/landmark_A.txt'
    B_path = 'datas/landmark_B.txt'
    odometry_path = 'datas/odometry.txt'

    A_data = read_txt(A_path).T
    B_data = read_txt(B_path).T
    od_data = read_txt(odometry_path).T

    # landmarks_A angular
    # visualize_trajectory(A_data[0], A_data[1], A_data[2])

    # # odometry_data trajectory
    # x_t, y_t, theta_t = get_trajectory(od_data[0],od_data[1],od_data[2],
    #                A_data[0][0], A_data[1][0], A_data[2][0])
    # visualize_trajectory(x_t, y_t, theta_t)
    
    # perform ekf
    # main_A(A_data, od_data)

    # landmarks_B
    # visualize_trajectory(B_data[0], B_data[1], B_data[2])
    
    xlist = main_B(B_data)
    v = []
    w = []
    rate = []
    for i in xlist:
        v.append(i[3][0])
        w.append(i[2][0])
    for i in range(len(w)-1):
        yawrate = (w[i+1]-w[i]) / 0.1 #dt=0.1
        rate.append(yawrate)
    plt_velocity(range(len(v)),v,label = 'velocity')
    plt_velocity(range(len(rate)),rate,label = 'angular rate')
    # plt_velocity(range(len(w)),w,label = 'angular rate')
    # print(vlist)



