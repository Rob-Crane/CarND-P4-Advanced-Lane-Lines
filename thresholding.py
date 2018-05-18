import glob
import cv2
import numpy as np
from scipy.optimize import minimize
import config

class LaneImage:
    def __init__(self, fname):
        self.img = cv2.imread(fname)
        hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        self.l = hls[:,:,1]
        self.s = hls[:,:,2]
        self.name = fname.split('/')[-1].split('.')[0]

    # The next 4 methods allow compute and threshold using saturation and lighting gradient thresholds
    __threshold = lambda img, threshold : (img > threshold[0]) & (img < threshold[1])

    def lght_direc(self, threshold, save_img=False):
        bool_img = LaneImage.__threshold(LaneImage.__gradient_dir(self.l), threshold)
        if save_img:
            LaneImage.__bool_imwrite('dir/lighting/' + self.name + '_' + str(threshold) + 'ldir.png',
                bool_img)
        return bool_img

    def sat_direc(self, threshold, save_img=False):
        bool_img = LaneImage.__threshold(LaneImage.__gradient_dir(self.s), threshold)
        if save_img:
            LaneImage.__bool_imwrite('dir/saturation/' + self.name + '_' + str(threshold) + 'sdir.png',
                bool_img)
        return bool_img

    def lght_mag(self, threshold, save_img=False):
        bool_img = LaneImage.__threshold(LaneImage.__gradient_mag(self.l), threshold)
        if save_img:
            LaneImage.__bool_imwrite('mag/lighting/' + self.name + '_' + str(threshold) + 'lmag.png',
                bool_img)
        return bool_img

    def sat_mag(self, threshold, save_img=False):
        bool_img = LaneImage.__threshold(LaneImage.__gradient_mag(self.s), threshold)
        if save_img:
            LaneImage.__bool_imwrite('mag/saturation/' + self.name + '_' + str(threshold) + 'smag.png',
                bool_img)
        return bool_img

    def __warp_view(img, road_region, to_overhead):
        im_h, im_w = img.shape
        reg_w, reg_h = config.UNDISTORTED_RECT
        rect = np.array([[im_w/2-reg_w/2, im_h],
                        [im_w/2-reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h]],
                        dtype=np.float32)

        if to_overhead:
            M = cv2.getPerspectiveTransform(road_region, rect)
        else:
            M = cv2.getPerspectiveTransform(rect, road_region)
        view = cv2.warpPerspective(img, M, (im_w, im_h))

        return view

    # def top_view(self):
        # bool_img = (self.sat_mag(config.SMAG_THRESHOLD) & \
                # self.sat_direc(config.SDIR_THRESHOLD)) | \
                # (self.lght_mag(config.LMAG_THRESHOLD) & \
                # self.lght_direc(config.LDIR_THRESHOLD))

        # road_region = np.array(config.ROAD_REGION, dtype=np.float32)

        # masked_img = LaneImage._region_of_interest(
                # np.uint8(bool_img),
                # np.int32([road_region]))
        # top_view = LaneImage.__warp_view(masked_img, road_region, True)
        # return top_view
    
    def threshold_img(self):
        return np.uint8((self.sat_mag(config.SMAG_THRESHOLD) & \
                self.sat_direc(config.SDIR_THRESHOLD)) | \
                (self.lght_mag(config.LMAG_THRESHOLD) & \
                self.lght_direc(config.LDIR_THRESHOLD)))

    to_top = True
    to_road = False
    def __shift(img, direction):
        im_h, im_w = img.shape
        reg_w, reg_h = config.UNDISTORTED_RECT
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
        

    def lane_poly(self, n, lane_width):

        # apply gradient thresholds
        thresholded = self.threshold_img()

        # apply a mask to only examing road region   
        mask = np.zeros_like(thresholded, dtype = np.uint8)   
        road_region = np.array(config.ROAD_REGION)
        cv2.fillPoly(mask, [road_region], 255)
        masked_image = cv2.bitwise_and(thresholded, mask)

        # shift to a top-down perspective
        top_view = LaneImage.__shift(masked_image, LaneImage.to_top)

        # Pull out x,y variables from pixels
        # top_view = self.top_view()

        # get pixel locations
        lane_px = np.argwhere(top_view) #row, col

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

        # random initialization of polynomial coefficients observed to
        # cause bad convergence at local minima, solved by initializing
        # x^0 and x^1 terms to an approximation of final values and 0
        # for higher order terms

        top_threshold = 0.25 # top 25% of image to average
        top_th_px = self.img.shape[0]*top_threshold
        top_th_norm = (top_th_px - means[0]) / devs[0]
        top_y = y[x < top_th_norm]
        top_mean = np.mean(top_y)
        yt_est = top_mean - lane_width / devs[1] / 2 # est. of left lane at top

        bottom_threshold = 0.25 # bottom 25%
        bot_th_px = self.img.shape[0]*(1-bottom_threshold)
        bot_th_norm = (bot_th_px - means[0]) / devs[0]
        bot_y = y[x > bot_th_norm]
        bot_mean = np.mean(bot_y)
        yb_est = bot_mean - lane_width / devs[1] / 2

        # put those estimates at the top and bottom of image
        xt = -devs[0] / means[0]
        xb = (self.img.shape[0] - means[0]) / devs[0]

        # slope
        m = (yb_est - yt_est) / (xb - xt)
        # y-intercept
        y_int = m*(-xt) + yt_est

        p = np.zeros(n)
        p[0], p[1] = y_int, m

        res = minimize(
            lambda p : abs_loss(p,x,y) , 
            p,
            jac=lambda p : abs_loss(p,x,y,True))

        draw_x = np.arange(0, top_view.shape[0]) 
        draw_x_norm = (draw_x - means[0]) / devs[0]
        left_norm, right_norm = get_y_primes(p, 
                draw_x_norm)
        left_y = left_norm * devs[1] + means[1]
        right_y = right_norm * devs[1] + means[1]

        pts = np.int32([list(zip(left_y, draw_x))])
        pts2 = np.int32([list(zip(right_y, draw_x))])

        blank_img = np.zeros_like(top_view)
        lines = cv2.polylines(blank_img, pts=pts, isClosed=False, color=1)
        lines = cv2.polylines(lines, pts=pts2, isClosed=False, color=1)
        return lines
    
    def __overlay(img, line_img):
        import pdb; pdb.set_trace()
        img = np.copy(img)
        bool_px = line_limg != 0
        if len(img.shape) > 2:
            img[bool_px, 3]=255
        else:
            img[bool_px] = 255

        return img


    # # copied from Udacity CarND Project1
    # def _region_of_interest(img, vertices):
        # """
        # Applies an image mask.
        
        # Only keeps the region of the image defined by the polygon
        # formed from `vertices`. The rest of the image is set to black.
        # """
        # #defining a blank mask to start with
        # mask = np.zeros_like(img)   
        
        # #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        # if len(img.shape) > 2:
            # channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            # ignore_mask_color = (255,) * channel_count
        # else:
            # ignore_mask_color = 255
            
        # #filling pixels inside the polygon defined by "vertices" with the fill color    
        # cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        # #returning the image only where mask pixels are nonzero
        # masked_image = cv2.bitwise_and(img, mask)
        # return masked_image

        

    def __bool_imwrite(fname, img):
        cv2.imwrite(config.OUT_DIR + '/' + fname, np.uint8(255*img))

    
    def __gradient_mag(
            img, 
            sobel_kernel=config.SOBEL_KERNEL):

        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = (grad_x**2 + grad_y**2)**0.5
        return mag

    def __gradient_dir(
            img, 
            sobel_kernel=config.SOBEL_KERNEL):

        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        direc = np.arctan2(np.absolute(grad_y), np.absolute(grad_x))
        return direc

    def test_pipeline(self):
        lines_img = self.lane_poly(3, 75)
        overhead = LaneImage.__shift(self.img, WarpDirection.to_top)
        overlay = LaneImage.__overlay(overhead, lines_img)
        cv2.imwrite(config.OUT_DIR + '/' + self.name + '_ovh.png', overlay)

def thresholds(begin, end, delta):
    while True:
        yield (begin, begin+delta)
        begin = begin + delta
        if begin >= end:
            break

if __name__ == '__main__':

    img_files = glob.glob('test_images/*.jpg')
    for fname in img_files:
        img = LaneImage(fname)
        # img.lane_poly(3, 75)
        img.test_pipeline()


