%%
- you can find the lines coordinates in './data' folder
  - in the .txt file each line is a image coordinate pair (x, y)
- randomly pick one or three lines to run the straight line method
- show the best undistorted images you can get in the submitted file

1. get_points(filename)
2. get_ABC(points)
3. get_x0y0(ABC_raw)
4. get_lamda(x0,y0,a,b,c)
5. undistort(img, x0, y0, lamda) 