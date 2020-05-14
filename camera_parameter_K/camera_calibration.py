import os
import glob
import cv2
import numpy as np

def calibrate_camera(cal_img_path):
    """
    Calibrate the camera given a directory which contains calibration chessboards.
    """
    # Read in all calibration images in a given file directory
    images = glob.glob(os.path.join(cal_img_path, 'img*.png'))

    # Creat arrays to store object (3D pints in real-world space) points and image points (2D points on image)
    obj_points = [] # 3d point in real world space
    img_points = [] # 2d points in image plane.


    # Prepare object points
    #square_size = 0.01905
    obj_p = np.zeros((9*9, 3), np.float32)
    obj_p[:,:2] = np.mgrid[0:9, 0:9].T.reshape(-1,2)
    #obj_p = obj_p*square_size

    for img_fname in images:
        # Read in images
        chessboard = cv2.imread(img_fname)

        # Convert chessboard image to gray scale
        gray_chess = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_chess, (9, 9), None)

        # If corners are found, add image points and object points
        if ret==True:
            img_points.append(corners)
            obj_points.append(obj_p)

            # Draw corners on the image
            chessboard = cv2.drawChessboardCorners(chessboard, (7, 7), corners, ret)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_chess.shape[::-1], None, None)

    print(mtx)

    return ret, mtx, dist, rvecs, tvecs

if __name__ == "__main__":
    calibrate_camera("calibration_images")
