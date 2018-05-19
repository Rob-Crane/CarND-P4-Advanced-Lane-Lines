import glob
import cv2
import numpy as np
from scipy.optimize import minimize
from moviepy.editor import VideoFileClip
import config

def pipeline(image, name=None):

    if name is None:
        write_images = False
    else:
        write_images = True

    # convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

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
        reg_w, reg_h = config.UNDISTORTED_RECT
        rect = np.array([[im_w/2-reg_w/2, im_h],
                        [im_w/2-reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h]],
                        dtype=np.float32)
        road_region = np.array(config.ROAD_REGION, dtype=np.float32)

        if direction:
            M = cv2.getPerspectiveTransform(road_region, rect)
        else:
            M = cv2.getPerspectiveTransform(rect, road_region)
        view = cv2.warpPerspective(img, M, (im_w, im_h))
        return view

    overhead_thresholds = shift(masked_image, to_top)
    if write_images:
        cv2.imwrite(config.OUT_DIR + '/' + name + '_1.png', overhead_thresholds)

    # get pixel locations
    lane_px = np.argwhere(overhead_thresholds) #row, col

    # zero-center + normalize (convert to z-scores)
    means = np.mean(lane_px, axis=0) # avg(rows), avg(cols)
    devs = np.std(lane_px, axis=0) # std(rows), std(cols)
    norm_px = (lane_px - means) / devs # norm(rows), norm(cols)

    x,y = norm_px[:, 0], norm_px[:, 1] 

    # calculates estimated lane positions
    def get_y_primes(p, x):
        (left, right) = (np.zeros(len(x)), np.zeros(len(x)))
        for i in range(len(p)):
            left = left + p[i] * x**i
            width_norm = config.LANE_WIDTH / devs[1]
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
        pts = np.int32([list(zip(left_y, draw_x))])
        pts2 = np.int32([list(zip(right_y, draw_x))])

        blank_img = np.zeros_like(overhead_thresholds)
        lines = cv2.polylines(blank_img, pts=pts, isClosed=False, color=1, thickness=2)
        lines = cv2.polylines(lines, pts=pts2, isClosed=False, color=1, thickness=2)
        return lines

    def overlay(img, line_img):
        img = np.copy(img)
        bool_px = line_img != 0
        if len(img.shape) > 2:
            img[bool_px]=[0,0,255]
        else:
            img[bool_px] = 255

        return img

    # random initialization of polynomial coefficients observed to
    # cause bad convergence at local minima, solved by initializing
    # x^0 and x^1 terms to an approximation of final values and 0
    # for higher order terms

    top_threshold = 0.25 # top 25% of image to average
    top_th_px = overhead_thresholds.shape[0]*top_threshold
    top_th_norm = (top_th_px - means[0]) / devs[0]
    top_y = y[x < top_th_norm]
    top_mean = np.mean(top_y)
    yt_est = top_mean - config.LANE_WIDTH / devs[1] / 2 # est. of left lane at top

    bottom_threshold = 0.25 # bottom 25%
    bot_th_px = overhead_thresholds.shape[0]*(1-bottom_threshold)
    bot_th_norm = (bot_th_px - means[0]) / devs[0]
    bot_y = y[x > bot_th_norm]
    bot_mean = np.mean(bot_y)
    yb_est = bot_mean - config.LANE_WIDTH / devs[1] / 2

    # put those estimates at the top and bottom of image
    xt = -devs[0] / means[0]
    xb = (overhead_thresholds.shape[0] - means[0]) / devs[0]

    # slope
    m = (yb_est - yt_est) / (xb - xt)
    # y-intercept
    y_int = m*(-xt) + yt_est


    p = np.zeros(config.POLY_DEG)
    p[0], p[1] = y_int, m

    initial_lines = overlay(overhead_thresholds, lines_image(p))
    if write_images:
        cv2.imwrite(config.OUT_DIR + '/' + name + '_2.png', initial_lines)
    p = minimize(
        lambda p : abs_loss(p,x,y) , 
        p,
        jac=lambda p : abs_loss(p,x,y,True)).x
    
    fitted_lines = lines_image(p)
    threshold_overlay = overlay(overhead_thresholds, fitted_lines)
    overhead_original = shift(image, to_top)
    overhead_overlay = overlay(overhead_original, fitted_lines)
    lines_shifted = shift(fitted_lines, to_road)
    lines_overlay = overlay(image, lines_shifted)
    if write_images:
        cv2.imwrite(config.OUT_DIR + '/' + name + '_3.png', threshold_overlay)
        cv2.imwrite(config.OUT_DIR + '/' + name + '_4.png', overhead_overlay)
        cv2.imwrite(config.OUT_DIR + '/' + name + '_5.png', lines_overlay)


    yb_est = get_y_primes(p, np.array([xb]))
    lane_center = (np.mean(yb_est) * devs[1] + means[1]) * 0.048 # get in meters referenced from left
    image_center = (threshold_img.shape[1] / 2) * 0.048

    location = image_center - lane_center


    row = 450
    if len(p) == 3:
        # import pdb; pdb.set_trace()
        x = (row - means[0]) / devs[0]
        # coefficients calculated by optimization are produced from
        # optimization on normalized y values.  Need to adjust to
        # produce output in meters
        A = p[2] * devs[1] / devs[0]**2 / 0.048
        B = p[1] * devs[1] / devs[0]
        l_or_r = 'L' if location < 0 else 'R'
        curvature = (1 + (2 * A * x + B)**2)**(3/2) / np.abs(2*A)
        img_text = 'Curv: {:.2f}m   Pos: {}{:.2f}'.format(curvature, l_or_r, np.abs(location))
        text_overlay = cv2.putText(lines_overlay, img_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if write_images:
            cv2.imwrite(config.OUT_DIR + '/' + name + '_6.png', text_overlay)
        return text_overlay
    else:
        print('Warning: only 3rd degree polynomial supported for curvature calculation')


i = 0
def process_video(image):
    global i
    out = pipeline(image, 'vid_' + str(i))
    i = i + 1
    return out

    


img_files = glob.glob('test_images/*.jpg')
for fname in img_files:
    name = fname.split('/')[-1].split('.')[0]
    image = cv2.imread(fname)
    pipeline(image, name)
clip = VideoFileClip('harder_challenge_video.mp4')
out_clip = clip.fl_image(process_video)
out_clip.write_videofile('outvideo_challenge.mp4', fps=10)

