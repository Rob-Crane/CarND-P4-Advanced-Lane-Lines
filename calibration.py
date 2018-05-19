import glob
import numpy as np
import cv2

import config

def getCameraCalibration(draw_images=False):

    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:6, 0:9].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    images = glob.glob(config.IMG_DIR + '/*.jpg')

    get_name = lambda fname : fname.split('/')[-1].split('.')[0] 
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(config.OUT_DIR + '/' + get_name(fname) + '_gray.jpg', gray)

        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

        if draw_images:
            cv2.drawChessboardCorners(img, (6, 9), corners, ret)
            cv2.imwrite(config.OUT_DIR + '/' + get_name(fname) + '_corners.jpg', img)

    img_shape = cv2.imread(images[0]).shape[1::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    if draw_images:
        for fname in images:
            img = cv2.imread(fname)
            undst = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite(config.OUT_DIR + '/' + get_name(fname) + '_undistort.jpg', undst)
    return mtx, dist

if __name__ == '__main__':
    getCameraCalibration(True)
    print('done')
