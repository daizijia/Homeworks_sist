import numpy as np

def ToVoxel(value, lowestVal, voxel_size = 0.5):
        return (int((1.0 / voxel_size) * value) - int(1.0 / voxel_size * lowestVal))

def random_down_sample(input, radio = 5):
    output = np.array([input[i*radio] for i in range(input.shape[0]//radio)])
    return output
    

def voxel_down_sample(input, voxel_size = 0.5):
    # voxel down sample

    output = []
    highestVal = [-100, -100, -100]
    lowestVal = [100, 100, 100]

    for row in input.tolist():
        for i in range(len(row)):
            if float(row[i]) > highestVal[i]:
                highestVal[i] = float(row[i])
            elif float(row[i]) < lowestVal[i]:
                lowestVal[i] = float(row[i])

    np_input = np.array(input)
    np_input = np_input.astype(float)
    np_input = np_input[np_input[:, 0].argsort()]
    print(len(np_input))

    # Initialize size of voxel cube and size of voxel grid
    voxel_x_range = ToVoxel(highestVal[0], lowestVal[0], voxel_size) + 1 
    voxel_y_range = ToVoxel(highestVal[1], lowestVal[1], voxel_size) + 1
    voxel_z_range = ToVoxel(highestVal[2], lowestVal[2], voxel_size) + 1 
    #print(voxel_x_range)

    #initialize voxel_grid and array 'has_point' to hold voxel cube points that have points from input in them
    voxel_grid = [[[VoxelPoint(0.0, 0.0, 0.0, 0) for z in range(voxel_z_range)] for y in range(voxel_y_range)] for x in range(voxel_x_range)]
    has_point = []
    #print(voxel_grid)

    # iterate through points in the input file and keep talley of their components in the voxel grid to later use to average the points
    for index in range(len(np_input)):
        Voxel_X = ToVoxel(np_input[index][0], lowestVal[0], voxel_size) 
        Voxel_Y = ToVoxel(np_input[index][1], lowestVal[1], voxel_size) 
        Voxel_Z = ToVoxel(np_input[index][2], lowestVal[2], voxel_size) 
        voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].x += np_input[index][0]
        voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].y += np_input[index][1]
        voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].z += np_input[index][2]
        voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].count += 1
        if (voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].count == 1):
            has_point.append([Voxel_X, Voxel_Y, Voxel_Z])

    # Iterate through each cube in the voxel grid and average their x, y, and z components to find the average point in each cube. Then append average point to output file 
    for index in range(1, len(has_point)):
        averaged_point = [0, 0, 0]
        Voxel_X = has_point[index][0]
        Voxel_Y = has_point[index][1]
        Voxel_Z = has_point[index][2]
        averaged_point[0] = round(voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].x / voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].count, 3)
        averaged_point[1] = round(voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].y / voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].count, 3)
        averaged_point[2] = round(voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].z / voxel_grid[Voxel_X][Voxel_Y][Voxel_Z].count, 3)
        output.append(averaged_point)
    return output

class VoxelPoint():
    
    def __init__(self, x, y, z, count):
        self.x = x
        self.y = y
        self.z = z
        self.count = count

