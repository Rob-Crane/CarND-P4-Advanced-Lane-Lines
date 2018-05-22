import glob
import cv2
import numpy as np
from scipy.optimize import minimize
from moviepy.editor import VideoFileClip
import config
import calibration

np.set_printoptions(precision=2)

class Cache:
    def __init__(self, N):
        self.center = 0.0
        self.radius = 0.0
        self.n = 0
        self.N = N
        self.p = np.zeros(config.POLY_DEG)
        self.count = 0

    # wavg = lambda old, new : old = ( (n-1) / n * old ) + ( 1/n * new )

    # def update_p(self, p):
        # if n < N:
            # self.n = self.n + 1
        # Cache.wavg(self.p, p)
cache = Cache(config.N)
def pipeline(image, mtx, dist, name, video_mode):

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

    # get expected pixel width of lane
    lane_width = config.LANE_WIDTH / config.REG_DIM[0] * config.OVERHEAD_RECT[0]

    # calculates estimated lane positions
    def get_y_primes(p, x):
        (left, right) = (np.zeros(len(x)), np.zeros(len(x)))
        for i in range(len(p)):
            left = left + p[i] * x**i
            width_norm = lane_width / devs[1]
            right = left + width_norm

        return left, right

    def abs_loss(p, x, y, ret_grads=False):
        left, right = get_y_primes(p, x)
        deltas = np.array([(y-left), (y-right)]).T
        candidate_losses = np.abs(deltas)
        m = deltas.shape[0]

        if ret_grads: # returning gradients
            active_terms = candidate_losses.argmin(axis=1)
            are_pos = np.where(deltas > 0, -1, 1) # when loss term is pos/neg -> linear
                                                  # inc/dec in loss
            act_multipliers = np.choose(active_terms, are_pos.T)

            grads = np.zeros((len(x), len(p)))
            for i in range(len(p)):
                grads[:,i] = act_multipliers * x**i

            return np.sum(grads, axis=0) / m

        else: # just return total loss
            min_losses = candidate_losses.min(axis=1)
            loss = np.sum(min_losses) / m / 2
            return loss

    def lines_image(p):
        draw_x = np.arange(0, overhead_thresholds.shape[0]) 
        draw_x_norm = (draw_x - means[0]) / devs[0]
        left_norm, right_norm = get_y_primes(p, 
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


    if video_mode and cache.n >  0:
        p = cache.p
    else: # need to initialize p

        # random initialization of polynomial coefficients observed to
        # cause bad convergence at local minima, solved by initializing
        # x^0 and x^1 terms to an approximation of final values and 0
        # for higher order terms

        top_threshold = 0.25 # top 25% of image to average
        top_th_px = overhead_thresholds.shape[0]*top_threshold
        top_th_norm = (top_th_px - means[0]) / devs[0]
        top_y = y[x < top_th_norm]
        top_mean = np.mean(top_y)
        yt_est = top_mean - lane_width / devs[1] / 2 # est. of left lane at top

        bottom_threshold = 0.25 # bottom 25%
        bot_th_px = overhead_thresholds.shape[0]*(1-bottom_threshold)
        bot_th_norm = (bot_th_px - means[0]) / devs[0]
        bot_y = y[x > bot_th_norm]
        bot_mean = np.mean(bot_y)
        yb_est = bot_mean - lane_width / devs[1] / 2

        # put those estimates at the top and bottom of image
        xt = -devs[0] / means[0]
        xb = (overhead_thresholds.shape[0] - means[0]) / devs[0]

        # slope
        m = (yb_est - yt_est) / (xb - xt)
        # y-intercept
        y_int = m*(-xt) + yt_est

        p = np.zeros(config.POLY_DEG)
        p[0], p[1] = y_int, m

    res = minimize(
        lambda p : abs_loss(p,x,y) , 
        p,
        jac=lambda p : abs_loss(p,x,y,True))
    p = res.x
    loss = res.fun


    def wavg(old, new):
        old = ( (cache.n-1) / cache.n * old ) + ( 1/cache.n * new )

    if video_mode:
        cache.count = cache.count + 1
        if cache.n == 0:
            cache.p = p
            cache.n = cache.n + 1
        else:
            if cache.n < cache.N:
                cache.n = cache.n + 1
            wavg(cache.p, p)
    
    poly = lines_image(p) 
    poly_bin = overlay(overhead_thresholds, poly) # polygon overlay on binary threshold image
    overhead_original = shift(image, to_top) 
    poly_col = overlay(overhead_original, poly) # polygon overlay on color overhead view
    poly_shifted = shift(poly, to_road)
    poly_orig = overlay(image, poly_shifted) # polygon overlay of original image
    if video_mode:
        npoly = lines_image(cache.p) # get a view of normalzied poly
        npoly_bin = overlay(overhead_thresholds, npoly) # overhead threshold view of fitted lines (normalized)
        npoly_col = overlay(overhead_original, npoly)
        npoly_shifted = shift(npoly, to_road)
        npoly_orig = overlay(image, npoly_shifted)

    bot = (overhead_thresholds.shape[0] - means[0]) / devs[0]
    yb_est = get_y_primes(p, np.array([bot]))
    lane_center = (np.mean(yb_est) * devs[1] + means[1]) * 0.048 # get in meters referenced from left
    image_center = (threshold_img.shape[1] / 2) * 0.048
    location = image_center - lane_center

    row = 500
    if len(p) == 3:
        x = (row - means[0]) / devs[0]
        # coefficients calculated by optimization are produced from
        # optimization on normalized y values.  Need to adjust to
        # produce output in meters
        A = p[2] * devs[1] / devs[0]**2 / (config.LANE_WIDTH / lane_width)
        B = p[1] * devs[1] / devs[0]
        l_or_r = 'L' if location < 0 else 'R'
        curvature = (1 + (2 * A * x + B)**2)**(3/2) / np.abs(2*A)
        # img_text = '{} Curv: {:.2f}m   Pos: {}{:.2f}'.format(cache.count, curvature, l_or_r, np.abs(location))
        if video_mode:
            img_text = '{}: P:{} J:{:.2f}'.format(cache.count, p, loss)
        else:
            img_text = 'P:{} J:{:.2f}'.format(p, res.fun)
        results_img = cv2.putText(poly_orig, img_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


    scale = config.SCALE
    def scale_img(image):
        if scale !=1.0:
            return cv2.resize(image, (0,0), fx=scale, fy=scale)
        else:
            return image

    # cv2.imwrite(config.OUT_DIR + '/' + name + '_00.png', scale_img(image))                 # 0: original image
    # cv2.imwrite(config.OUT_DIR + '/' + name + '_01.png', scale_img(undistort))             # 1: camera undistortion
    # cv2.imwrite(config.OUT_DIR + '/' + name + '_02.png', scale_img(masked_image))            # 2: binary threshold image
    # cv2.imwrite(config.OUT_DIR + '/' + name + '_03.png', scale_img(overhead_thresholds))     # 3: overhead shift of binary img
    # cv2.imwrite(config.OUT_DIR + '/' + name + '_04.png', scale_img(initial_lines))         # 4: overhead bin overlay of initialized p
    cv2.imwrite(config.OUT_DIR + '/' + name + '_05.png', scale_img(poly_bin))                # 5: overhead bin overlay of fit
    cv2.imwrite(config.OUT_DIR + '/' + name + '_07.png', scale_img(poly_col))                # 7: overhead col overlay of fit
    cv2.imwrite(config.OUT_DIR + '/' + name + '_09.png', scale_img(poly_orig))               # 9: original image overlay of fit
    if video_mode:
        name = name + str(cache.count)
        cv2.imwrite(config.OUT_DIR + '/' + name + '_06.png', scale_img(npoly_bin))               # 6: overhead bin overlay of npoly
        cv2.imwrite(config.OUT_DIR + '/' + name + '_08.png', scale_img(npoly_col))               # 8: overhead col overlay of npoly
        cv2.imwrite(config.OUT_DIR + '/' + name + '_10.png', scale_img(npoly_orig))               # 10: originalimage overlay of npoly


    return results_img


mtx, dist = calibration.getCameraCalibration()
img_files = glob.glob('test_images/*.jpg')
for fname in img_files:
    name = fname.split('/')[-1].split('.')[0]
    image = cv2.imread(fname)
    pipeline(image, mtx, dist, name, False)

clip = VideoFileClip('project_video.mp4')
out_clip = clip.fl_image(lambda frame : pipeline(frame, mtx, dist, 'vid', True))
out_clip.write_videofile(config.OUT_DIR + '/' + config.VID_NAME, config.FPS)

