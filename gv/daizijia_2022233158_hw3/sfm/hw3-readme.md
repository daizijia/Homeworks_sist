# DataSet
data is in the '[NotreDame](https://www.cs.cornell.edu/projects/bigsfm/)' folder

# Functions to be filled:
1. sfminit/src/trans_solver.cc: 
    solve_translations_problem
2. sfminit/src/mfass.cc: 
    1) flip_neg_edges
    2) mfas_ratio
    3) broken_weight
3. sfminit/onedsfm.py: 
    kdeSample

# Set Up
1. install docker engine
2. install the container: 
``````
./xproj build
``````
Note that the currect version of python and ceres are already installed in this container

3. get into the container 
``````
./xproj run
``````

4. after you filled the above functions, compile the C++ function to python
``````
python setup.py build_ext --inplace
``````

5. make an output folder, e.g. "output", and run the demo
``````
python scripts/eccv_demo.py NotreDame output
``````


