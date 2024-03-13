import cv2
import numpy as np
import glob

# Define the chess board rows and columns
rows = 6
cols = 8

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# List of calibration images
images = glob.glob('distortion2/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    
    if ret == True:
        objpoints.append(objp)

        # Refines the corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)
        print("success")
    else:
        cv2.destroyAllWindows()

# Calibrate fisheye camera
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    np.expand_dims(np.asarray(objpoints), -2),
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    criteria)
print(K)
print(D)