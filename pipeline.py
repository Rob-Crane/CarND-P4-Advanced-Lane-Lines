import os
import csv
import glob
from collections import deque
import cv2
import numpy as np
from scipy.optimize import minimize
from moviepy.editor import VideoFileClip
import config
import calibration

np.set_printoptions(precision=config.PRINT_PREC)

out_dir = config.OUT_DIR + '/' + config.OUTPUT_NAME
os.mkdir(out_dir)

class Cache:

    def __init__(self):
        self.count = 0
        buff_shape = (config.N, config.POLY_DEG + 1)
        self.buffer = np.zeros(shape=buff_shape)

    def update_fit(self,l, p):

        row = self.count%config.N
        if self.count > 0:
            diff = np.abs((p - self.p) / self.p)
            if np.any(diff > config.MAX_DIFF):
                update = False
            else:
                update = True
        else:
            update = True

        if update:
            self.buffer[row][0] = l
            self.buffer[row][1:] = p
        else:
            self.buffer[row][0] = self.l
            self.buffer[row][1:] = self.p

        self.count = self.count + 1

        if self.count <= config.N:
            avg = self.buffer[0:self.count].mean(axis=0)
        else:
            avg = self.buffer.mean(axis=0)

        self.l = avg[0]
        self.p = avg[1:]
        return self.l, self.p

cache = Cache()
def pipeline(image, mtx, dist, name, video_mode=False, logger=None):

    undistort = cv2.undistort(image, mtx, dist, None, mtx)

    # convert to HLS color space
    hls = cv2.cvtColor(undistort, cv2.COLOR_BGR2HLS)

    def threshold(img, threshold):
        return (img > threshold[0]) & (img < threshold[1])

    sobel_kernel = config.SOBEL_KERNEL
    def magnitude_gradient(img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = (grad_x**2 + grad_y**2)**0.5
        return mag

    def direction_gradient(img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        direc = np.arctan2(np.absolute(grad_y), np.absolute(grad_x))
        return direc
    
    lighting_img = hls[:,:,1]
    saturation_img = hls[:,:,2]

    lighting_mag = threshold(magnitude_gradient(lighting_img),
                            config.LMAG_THRESHOLD)
    lighting_dir = threshold(direction_gradient(lighting_img),
                            config.LDIR_THRESHOLD)
    saturation_mag = threshold(magnitude_gradient(saturation_img),
                            config.SMAG_THRESHOLD)
    saturation_dir = threshold(direction_gradient(saturation_img),
                            config.SDIR_THRESHOLD)

    # turn boolean values from threhsold into b&w image
    threshold_img = np.uint8(
            (saturation_mag & saturation_dir) | \
            (lighting_mag & lighting_dir)) * 255

    # apply a mask to only examing road region   
    mask = np.zeros_like(threshold_img, dtype = np.uint8)   
    cv2.fillPoly(mask, [np.array(config.ROAD_REGION)], 255)
    masked_image = cv2.bitwise_and(threshold_img, mask)

    to_top = True
    to_road = False

    def shift(img, direction):
        im_h, im_w = img.shape[0], img.shape[1]
        road_region = np.array(config.ROAD_REGION, dtype=np.float32)
        reg_w, reg_h = config.OVERHEAD_RECT
        rect = np.array([[im_w/2-reg_w/2, im_h],
                        [im_w/2-reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h]],
                        dtype=np.float32)

        if direction:
            M = cv2.getPerspectiveTransform(road_region, rect)
        else:
            M = cv2.getPerspectiveTransform(rect, road_region)
        view = cv2.warpPerspective(img, M, (im_w, im_h))
        return view

    overhead_thresholds = shift(masked_image, to_top)

    # get pixel locations
    lane_px = np.argwhere(overhead_thresholds) #row, col

    # zero-center + normalize (convert to z-scores)
    means = np.mean(lane_px, axis=0) # avg(rows), avg(cols)
    devs = np.std(lane_px, axis=0) # std(rows), std(cols)
    norm_px = (lane_px - means) / devs # norm(rows), norm(cols)

    x,y = norm_px[:, 0], norm_px[:, 1] 

    # calculates estimated lane positions
    def get_y_primes(p, l, x):
        (left, right) = (np.zeros(len(x)), np.zeros(len(x)))
        for i in range(len(p)):
            left = left + p[i] * x**i
            width_norm = l / devs[1]
            right = left + width_norm

        return left, right

    # line_bin = overlay(overhead_thresholds, line_image) # polygon overlay on binary threshold image
    # cv2.imwrite(config.OUT_DIR + '/' + name + '_033.png', line_bin)                # 5: overhead bin overlay of fit

    # end draw debug

    def abs_loss(p, l, x, y, ret_grads=False):

        left, right = get_y_primes(p, l, x)
        deltas = np.array([(y-left), (y-right)]).T
        candidate_losses = np.abs(deltas)

        # avg = m*x + y_int
        # eligible_lane = np.array(y > avg)
        eligible_lane = np.array(y > overhead_thresholds.shape[1]/2/devs[1])
        minimum_lane = candidate_losses.argmin(axis=1)
        # following produces mask thats true for points in correct region and
        # that are closer to that lane's fit line
        region_mask = np.logical_not(np.logical_xor(eligible_lane, minimum_lane))
    
        M = deltas.shape[0]
        if ret_grads: # returning gradients
            grad_multipliers = np.where(deltas > 0, -1, 1) # when loss term is pos/neg -> linear
                                                          # inc/dec in loss
            closer_multiplier = np.choose(minimum_lane, grad_multipliers.T)
            active_multiplier = closer_multiplier[region_mask]

            partial_grads = np.zeros((len(active_multiplier), len(p)+1))
            partial_grads[:,0] = active_multiplier # (happens to be) gradient of l (learned lane width)
            for i in range(len(p)):
                partial_grads[:,i+1] = active_multiplier * x[region_mask]**i

            return np.sum(partial_grads, axis=0) / M

        else: # just return total loss
            min_losses = np.choose(minimum_lane, candidate_losses.T)
            eligible_losses = min_losses[region_mask]
            loss = np.sum(min_losses) / M / 2
            return loss

    def lines_image(p,l):
        draw_x = np.arange(0, overhead_thresholds.shape[0]) 
        draw_x_norm = (draw_x - means[0]) / devs[0]
        left_norm, right_norm = get_y_primes(p, l,
                draw_x_norm)
        left_y = left_norm * devs[1] + means[1]
        right_y = right_norm * devs[1] + means[1]

        left_y = np.nan_to_num(left_y)
        right_y = np.nan_to_num(right_y)

        # From Udacity notes, arrange points and create polygon
        pts_left = np.array([np.transpose(np.vstack([left_y, draw_x]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_y, draw_x])))])
        pts = np.hstack((pts_left, pts_right))

        blank_channel = np.zeros_like(overhead_thresholds)
        poly_image = np.dstack([blank_channel, blank_channel, blank_channel])

        # Draw the lane onto the warped blank image
        cv2.fillPoly(poly_image, np.int_([pts]), (0,255, 0))

        return poly_image

    def overlay(img, poly_img):

        if len(img.shape) == 2:
            img = np.dstack([img, img, img])

        result = cv2.addWeighted(img, 1, poly_img, 0.3, 0)
        return result
    

    # expected pixel width - this will be a trained parameter
    lane_width = config.LANE_WIDTH / config.REG_DIM[0] * config.OVERHEAD_RECT[0]

    if video_mode and cache.count >  0:
        p = cache.p
        l = cache.l
    else: # need to initialize p

        p = np.zeros(config.POLY_DEG)
        p[0] = -lane_width / devs[1]/2 # roughly place left lane line
        l = lane_width

    ipoly = lines_image(p, l )
    ipoly_bin = overlay(overhead_thresholds, ipoly) # polygon overlay on binary threshold image

    trainables = np.concatenate(([l], p))
    res = minimize(
            lambda trainables : abs_loss(trainables[1:],trainables[0],x,y) , 
        trainables,
        jac=lambda trainables : abs_loss(trainables[1:],trainables[0],x,y, True))
    lnew = res.x[0]
    pnew = res.x[1:]
    loss = res.fun

    if video_mode:
        cache.update_fit(lnew, pnew)
        if logger is not None:
            if cache.count == 1:
                logger.writerow(
                        ['count', 'loss', 
                        'lnew', 'cache.l',
                        'pnew[0]', 'cache.p[0]',
                        'pnew[1]', 'cache.p[1]',
                        'pnew[2]', 'cache.p[2]'])

            logger.writerow(
                [cache.count, loss, 
                lnew, cache.l,
                pnew[0], cache.p[0],
                pnew[1], cache.p[1],
                pnew[2], cache.p[2]])
    
    poly = lines_image(pnew, lnew) 
    poly_bin = overlay(overhead_thresholds, poly) # polygon overlay on binary threshold image
    overhead_original = shift(image, to_top) 
    poly_col = overlay(overhead_original, poly) # polygon overlay on color overhead view
    poly_shifted = shift(poly, to_road)
    poly_orig = overlay(image, poly_shifted) # polygon overlay of original image
    if video_mode:
        npoly = lines_image(cache.p, cache.l) # get a view of normalzied poly
        npoly_bin = overlay(overhead_thresholds, npoly) # overhead threshold view of fitted lines (normalized)
        npoly_col = overlay(overhead_original, npoly)
        npoly_shifted = shift(npoly, to_road)
        npoly_orig = overlay(image, npoly_shifted)

    bot = (overhead_thresholds.shape[0] - means[0]) / devs[0]
    yb_est = get_y_primes(p, l, np.array([bot]))
    lane_center = (np.mean(yb_est) * devs[1] + means[1])
    image_center = (threshold_img.shape[1] / 2)
    location = (image_center - lane_center) * 0.0054  # approx. scale from pixels to meters

    row = 500
    if len(p) == 3:
        x = (row - means[0])
        # coefficients calculated by optimization are produced from
        # optimization on normalized y values.  Need to adjust to
        # produce output in meters
        if video_mode:
            p, l = cache.p, cache.l
        else:
            p, l = pnew, lnew

        A = p[2] * (144/1280) * devs[1] / devs[0]**2 # scale coefficients to pixel scale
        B = p[1] * (144/1280) * devs[1] / devs[0]

        px_curv = (1 + (2 * A * x + B)**2)**(3/2) / np.abs(2*A) # curvature value is in pixels
        curvature = px_curv * 0.048 # scale to meters (width compressed in this scale)
        width = l * 0.0054
        loc_str = "{:.2f}L".format(-location) if location < 0 else "{:.2f}R".format(location)
        # img_text = '{} Curv: {:.2f}m   Pos: {}{:.2f}'.format(cache.count, curvature, l_or_r, np.abs(location))
        if video_mode:
            img_text = '{}: Pos:{} Width:{:.2f} Curv:{:.2f}'.format(cache.count, loc_str, width, curvature)
            img = npoly_orig
        else:
            img_text = 'Pos:{} Width:{:.2f} Curv:{:.2f}'.format(loc_str, width, curvature)
            img = poly_orig
        results_img = cv2.putText(img, img_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    scale = config.SCALE
    def scale_img(image):
        if scale !=1.0:
            return cv2.resize(image, (0,0), fx=scale, fy=scale)
        else:
            return image

    if video_mode:
        name = name + str(cache.count)
    cv2.imwrite(out_dir + '/' + name + '_00.png', scale_img(image))                 # 0: original image
    cv2.imwrite(out_dir + '/' + name + '_01.png', scale_img(undistort))             # 1: camera undistortion
    cv2.imwrite(out_dir + '/' + name + '_02.png', scale_img(masked_image))            # 2: binary threshold image
    cv2.imwrite(out_dir + '/' + name + '_03.png', scale_img(overhead_thresholds))     # 3: overhead shift of binary img
    cv2.imwrite(out_dir + '/' + name + '_04.png', scale_img(ipoly_bin))         # 4: overhead bin overlay of initialized p
    cv2.imwrite(out_dir + '/' + name + '_05.png', scale_img(poly_bin))                # 5: overhead bin overlay of fit
    cv2.imwrite(out_dir + '/' + name + '_07.png', scale_img(poly_col))                # 7: overhead col overlay of fit
    cv2.imwrite(out_dir + '/' + name + '_09.png', scale_img(poly_orig))               # 9: original image overlay of fit
    if video_mode:
        cv2.imwrite(out_dir + '/' + name + '_06.png', scale_img(npoly_bin))               # 6: overhead bin overlay of npoly
        cv2.imwrite(out_dir + '/' + name + '_08.png', scale_img(npoly_col))               # 8: overhead col overlay of npoly
        cv2.imwrite(out_dir + '/' + name + '_10.png', scale_img(npoly_orig))               # 10: originalimage overlay of npoly


    return results_img


mtx, dist = calibration.getCameraCalibration()
img_files = glob.glob('test_images/*.jpg')
for fname in img_files:
    name = fname.split('/')[-1].split('.')[0]
    image = cv2.imread(fname)
    pipeline(image, mtx, dist, name)

logfname = out_dir + '/' + config.OUTPUT_NAME + '.csv'
logfile = open(logfname, 'w', newline='')
logger = csv.writer(logfile)

clip = VideoFileClip('project_video.mp4')
out_clip = clip.fl_image(lambda frame : pipeline(frame, mtx, dist, 'vid', True, logger))
out_clip.write_videofile(out_dir + '/' + config.OUTPUT_NAME + '.mp4', config.FPS)

