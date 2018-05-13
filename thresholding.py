import glob
import cv2
import numpy as np
import config

class LaneImage:
    def __init__(self, fname):
        self.img = cv2.imread(fname)
        hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        self.l = hls[:,:,1]
        self.s = hls[:,:,2]
        self.name = fname.split('/')[-1].split('.')[0]

    # The next 4 methods allow compute and threshold using saturation and lighting gradient thresholds

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

    def get_lanes(self, save_img = False):
        bool_img = (self.sat_mag(config.SMAG_THRESHOLD) & \
                self.sat_direc(config.SDIR_THRESHOLD)) | \
                (self.lght_mag(config.LMAG_THRESHOLD) & \
                self.lght_direc(config.LDIR_THRESHOLD))

        # mask = np.zeros_like(bool_img, dtype=np.int32)
        road_region = np.array(config.ROAD_REGION, dtype=np.float32)

        # import pdb; pdb.set_trace()

        masked_img = LaneImage._region_of_interest(
                np.uint8(bool_img),
                np.int32([road_region]))
        # cv2.fillPoly(mask, np.int32([road_region]), 255)
        # masked_img = cv2.bitwise_and(np.uint8(bool_img), np.uint8(mask))

        im_h, im_w = masked_img.shape
        reg_w, reg_h = config.UNDISTORTED_RECT
        rect = np.array([[im_w/2-reg_w/2, im_h],
                        [im_w/2-reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h-reg_h],
                        [im_w/2+reg_w/2, im_h]],
                        dtype=np.float32)

        
        # import pdb; pdb.set_trace()
        M = cv2.getPerspectiveTransform(road_region, rect)
        top_view = cv2.warpPerspective(masked_img, M, (im_w, im_h))

        if save_img:
            LaneImage.__bool_imwrite('pipeline/' + self.name + '_msk.png', masked_img)
            LaneImage.__bool_imwrite('pipeline/' + self.name + '_top.png', top_view)
        return top_view

    # copied from Udacity CarND Project1
    def _region_of_interest(img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

        

    def __bool_imwrite(fname, img):
        cv2.imwrite(config.OUT_DIR + '/' + fname, np.uint8(255*img))

    def __threshold(img, threshold):
        return (img > threshold[0]) & (img < threshold[1])
    
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
        img.get_lanes(True)

        # lmag_thresholds = thresholds(0, 140000, 7000)
        # for threshold in lmag_thresholds:
            # img.lght_mag(threshold, True) 

        # smag_thresholds = thresholds(0, 170000, 8500)
        # for threshold in smag_thresholds:
            # img.sat_mag(threshold, True) 

        # sdir_thresholds = thresholds(0, 1.6, 0.08)
        # for threshold in sdir_thresholds:
            # img.sat_direc(threshold, True) 

        # ldir_thresholds = thresholds(0, 1.6, 0.08)
        # for threshold in ldir_thresholds:
            # img.lght_direc(threshold, True) 
