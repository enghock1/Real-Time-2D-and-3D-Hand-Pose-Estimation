This folder is used to compute the intrinsic camera parameter K of the camera (or webcam). 

Visit https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html 
for more information about camera calibration.

To obtain the camera parameter K, first you need to calibrate your camera.
To do that, you can use your cell phone with the checkerboard pattern image (located in the same folder) and take 
roughly 20-30 images using the camera by running take_picture.py.

Once the images are taken, run camera_calibration.py to obtain the parameter K. 

The parameter K is needed to run real-time hand pose estimation using camera or webcam.  