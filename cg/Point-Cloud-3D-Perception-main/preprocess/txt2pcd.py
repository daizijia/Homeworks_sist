import math
import os
def txt2pcd(filename):
    xlist = []
    ylist = []
    zlist = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            x, y, z,_,_,_ = [math.fabs(float(i)) for i in lines.split(',')]
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
 
    savefilename = filename + ".pcd"
    #savefilename = "../data/pcd/" + savefilename
    if not os.path.exists(savefilename):
        f = open(savefilename, 'w')
        f.close()
    with open(savefilename, 'w') as file_to_write:
        file_to_write.writelines("# .PCD v0.7 - Point Cloud Data file format\n")
        file_to_write.writelines("VERSION 0.7\n")
        file_to_write.writelines("FIELDS x y z\n")
        file_to_write.writelines("SIZE 4 4 4\n")
        file_to_write.writelines("TYPE F F F\n")
        file_to_write.writelines("COUNT 1 1 1\n")
        file_to_write.writelines("WIDTH " + str(len(xlist)) + "\n")
        file_to_write.writelines("HEIGHT 1\n")
        file_to_write.writelines("VIEWPOINT 0 0 0 1 0 0 0\n")
        file_to_write.writelines("POINTS " + str(len(xlist)) + "\n")
        file_to_write.writelines("DATA ascii\n")
        for i in range(len(xlist)):
            file_to_write.writelines(str(xlist[i]) + " " + str(ylist[i]) + " " + str(zlist[i]) + "\n")
            
filename = 'visual/result/000144.txt'
txt2pcd(filename)