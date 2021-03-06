﻿# realsense_experiments

A series of experiment and tooling for working with the Intel Realsense L515 lidar camera

# High level description:
- The camera is pointed at an empty flat surface (floor, table, etc..)
- Least squares algorithm is used to fit a plane to the point cloud of the flat surface
- The calculated plane can be used for background subtraction to create a binary mask
- the bounding box, area, centroid, and trajectory of objects is tracked using opencv

# Runing this on your machine
- Get your hands on an Intel Realsense L515 camera
- Install the realsense SDK (https://www.intelrealsense.com/sdk-2/)
- Install anaconda and git on your machine if you dont have them already
- Run the following commands on your terminal

```
git clone https://github.com/juanborbon93/realsense_experiments.git
cd realsense_experiments
conda env create -f environment.yml
conda activate realsense
```

- Point the camera at a flat surface
- Start the program:
```
python object_tracking.py
```
- Wait for the program to calibrate the scene
- Press enter when prompted and introduce objects into the scene
- Video capture will be saved as test.avi

# TO DO:
- clean up environment.yml (there are some packages that are not really needed)
- fix recursion error in modules.cvtools.frame_objects.assign_contours (this sometimes crashes the program)
- add other cool stuff :)
