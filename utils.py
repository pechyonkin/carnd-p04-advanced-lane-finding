import numpy as np
import glob
import cv2
from scipy.misc import imread
from scipy.signal import find_peaks_cwt
from os.path import join
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_video(v_path, video_size=(640, 480)):
    '''
    this routine displays video inline inside of the notebook
    '''
    return HTML("""
    <video width="{}" height="{}" controls>
      <source src="{}">
    </video>
    """.format(*video_size, v_path))

class Pipeline:
    '''
    this class provides all functionality necessary 
    for this project
    '''
    def __init__(self, debug_mode=False, show_plots=False):
        # frame counter used for tracking internal state
        self.frame_counter = 0

        # image size to be used by routines 
        self.image_size = None

        # fitted lines will be kept here
        self.left_fit = None
        self.right_fit = None

        # masks
        self.mask_left = None
        self.mask_right = None

        # for debug purposes
        self.show_plots = show_plots
        self.debug_mode = debug_mode

        self.line_left_start = None
        self.line_right_start = None

        # lane width in meters
        self.lane_width = 3.7

        # curvature and distance from center of lane
        self.curv = None
        self.dist = None

    def apply_pipeline(self, in_path, out_path, pipeline, mtx, dist, **kwargs):
        '''
        this routine accepts path to an image or a video,
        applies `pipeline` and saves result at `out_path`
        '''
        
        # retreive input and output extensions
        in_extension = in_path.split('.')[-1]
        out_extension = out_path.split('.')[-1]
        
        # simple test
        assert in_extension == out_extension, 'Extensions of input and output aren\'t same'
        
        if in_extension == 'mp4':
            print("Applying video pipeline.")
            in_clip = VideoFileClip(in_path)
            out_clip = in_clip.fl_image(pipeline)
            out_clip.write_videofile(out_path, audio=False)
            print("Apply video pipeline: SUCCESS!")
        elif in_extension == 'jpg':
            # print("Applying image pipeline.")
            # reads in image as RGB
            in_image = imread(in_path)
            out_image = pipeline(in_image, mtx, dist)
            return out_image
        
    def test_pipeline(self, path_to_img):
        '''
        this routine applies the pipeline and displays result
        for development and testing purposes
        '''
        debug_image = apply_pipeline(path_to_img, path_to_img, pipeline)
        plt.figure(figsize=(15,10))
        plt.imshow(debug_image)
        
    def camera_calibration(self, calibration_path, debug=False):
        '''
        this routine performs camera calibration
        it returns `mtx` and `dist` needed to
        undistort images taken from this camera
        '''
        # list all calibration images paths
        cal_images_names = glob.glob(join(calibration_path, '*.jpg'))
        
        # chessboard-specific parameters
        nx = 9
        ny = 6

        # code below is based on classroom example
        objpoints = [] # 3D points
        imgpoints = [] # 2D points

        # (x,y,z): (0,0,0), (1,0,0), etc
        objp = np.zeros((nx * ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates, z stays 0

        for fname in cal_images_names:
            # read in image
            img = cv2.imread(fname)

            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # in case chessboard was found successfully
            # it skips 3 images that do not show full chessboard (1, 4 and 5)
            if ret == True:
                # image points will be different for each calibration image
                imgpoints.append(corners)
                # object points are the same for all calibration images
                objpoints.append(objp)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                if debug:
                    plt.figure(figsize=(15,10))
                    plt.imshow(img)

        # calibration parameters calculation
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                           imgpoints, 
                                                           gray.shape[::-1], 
                                                           None, None)
        
        # will only use `mtx` and `dist` in this project, hence return
        return mtx, dist

    def undistort_image(self, image, mtx, dist, debug=False):
        '''
        returns an undistorted image (after camera calibration)
        '''
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        if debug:
            plt.figure(figsize=(15,10))
            plt.imshow(dst)
        return dst

    def gaussian_blur(self, image, kernel=5):
        '''
        this routine applies blur to reduce noise in images
        '''
        blurred = cv2.GaussianBlur(image, (kernel,kernel), 0)
        return blurred

    def apply_sobel(self, img, direction='x', sobel_kernel=3, thresh=(50, 255), one_channel=True):
        '''
        this routine calculates directional gradient and returns binary image
        '''
        # min and max limits
        thresh_min, thresh_max = thresh[0], thresh[1]
        # convert to grayscale
        if one_channel:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # x or y sobel
        if direction == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if direction == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # rescale to 8-bit integers
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # apply threshold
        result = np.zeros_like(scaled_sobel)
        result[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1

        # result
        return result


    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(50, 200), one_channel=True):
        '''
        this routine calculates sobel magnitude binary image
        '''
        # Convert to grayscale
        if one_channel:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # apply sobel to both directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # rescale to 8-bit integers
        scale_factor = np.max(magnitude)/255 
        magnitude = (magnitude/scale_factor).astype(np.uint8) 
        # apply threshold
        result = np.zeros_like(magnitude)
        result[(magnitude > mag_thresh[0]) & (magnitude < mag_thresh[1])] = 1

        # return binary image
        return result


    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2), one_channel=True):
        '''
        this function calculates direction of gradient and returns masked binary image
        '''
        # Convert to grayscale
        if one_channel:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # x, y gradients calculations
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # apply threshold
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        result =  np.zeros_like(absgraddir).astype(np.uint8)
        result[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # return binary image
        return result

    def apply_color_mask(self, hsv, img, low, high):
        '''
        this routine applies a thresholding color mask to an HSV image
        this returns pixels from original image that correspond to the mask
        all other pixels are blacked out
        '''
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    def color_mask(self, hsv, low, high):
        '''
        this routine creates a thresholding color mask to an HSV image
        '''
        mask = cv2.inRange(hsv, low, high)
        return mask

    def white_yellow_mask(self, img, ythresh=([0,100,100],[80,255,255]), wthresh=([10,0,240],[255,80,255])):
        '''
        this routine combined white and yellow threshold masks to detect both yellow and white lanes
        '''
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_mask = color_mask(hsv_img, np.array(ythresh[0]), np.array(ythresh[1]))
        white_mask = color_mask(hsv_img, np.array(wthresh[0]), np.array(wthresh[1]))
        white_yellow_mask = cv2.bitwise_or(yellow_mask, white_mask)
        return white_yellow_mask

    def in_range(self, img, low, high):
        result = np.copy(img)
        result[(result < low) | (result > high)] = 0
        return result

    # ------- #

    def combine_binary(self, img, ksize=5, 
                       xthresh=(30,150), ythresh=(50,255), 
                       mthresh=(30,100), dthresh=(0.7,1.3), 
                       with_lanes=True):
        '''
        this routine applies color space conversion and thresholding
        to be further used in fitting lanes
        '''

        # convert to HLS and retreive s channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # thresholded sobel x
        sobel_x = self.apply_sobel(gray, direction='x', sobel_kernel=5, thresh=(30, 150), one_channel=True)

        # threshold on s color channel
        s_thresh_min = 175
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_thresh = cv2.inRange(s_channel.astype('uint8'), 175, 250)
        s_binary[(s_thresh == 255)] = 1

        # combine sobel x and thresholded s channel binaries
        combined_binary = np.zeros_like(sobel_x)
        combined_binary[(s_binary == 1) | (sobel_x == 1)] = 1

        return combined_binary

    # ------- #

    def get_roi_vertices(self, image, vert_coef, hor_up, hor_down):
        '''
        this routine returns 4 vertices of the area in front of the car
        for top-down view perspective warp
        '''
        # finding the best region of interest
        y_size, x_size = image.shape[:2]

        # proportion of vertical size before upper line of ROI starts (starting from top)
        v_coef = vert_coef
        v_down = 0.95
        # proportion of horizontal size before upper left point of ROI (starting from left)
        up_left_coef = hor_up
        # proportion of horizontal size before upper right point of ROI (starting from left)
        up_right_coef = 1 - up_left_coef

        low_left_coef = hor_down
        low_right_coef = 1 - low_left_coef

        # defining tuple of 4 points of ROI
        low_left_point = [x_size * low_left_coef, y_size * v_down]
        low_right_point = [x_size * low_right_coef, y_size * v_down]
        up_left_point = [x_size * up_left_coef, y_size * v_coef]
        up_right_point = [x_size * up_right_coef, y_size * v_coef]

        # defining a poly (first and last points are the same to close the poly)
        verts = np.array( [up_left_point, up_right_point, low_right_point, low_left_point], dtype=np.float32)
        verts = np.reshape(verts, (4,1,2))
        return verts

    def corners_unwarp(self, img, vert_coef=0.63, hor_up=0.436, hor_down=0):
    # def corners_unwarp(img, vert_coef=0.63, hor_coef=0.444):
        '''
        this routine unwarps area in front of car into top-down view
        it returns warped image as well as warm and unwarp matrices
        '''
        # `o` is offset (was used in classroom example)
        off_l = 0.0
        off_r = 1 - off_l
        o = 0
        # grab the image shape
        width = img.shape[0]
        height = img.shape[1]
        src = self.get_roi_vertices(img, vert_coef=vert_coef, hor_up=hor_up, hor_down=hor_down)
        # destination points are corners of output image
        dst = np.float32([[o,width * off_l], [height-o,width * off_l], [height-o,width * off_r], [o,width * off_r]])
        dst = np.reshape(dst, (4,1,2))
        # given src and dst points, calculate the perspective transform matrix and inverse
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, (height, width))

        # Return the resulting image and matrix
        return warped, M, Minv

    def hsv_rgb(self, hsv):
        ''' prints out rgb value given an hsv value'''
        hsv = np.uint8([[hsv]])
        print(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    def rgb_hsv(self, rgb):
        ''' prints out hsv value given an rgb value'''
        rgb = np.uint8([[rgb]])
        print(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))

    def moving_average(self, array, window=5):
        '''
        this routine finds moving average using numpy
        vectorized operations
        '''
        result = np.cumsum(array, dtype=float)
        result[window:] = result[window:] - result[:len(result)-window]
        return result[window - 1:] / window

    def get_a_peak(self, hist):
        '''
        this routine finds a peak given a histogram
        '''
        peaks = np.argsort(hist)[::-1]

        if len(peaks) > 0:
            return peaks[0]
        else:
            return None



    def find_peaks(self, hist):
        '''
        this routine returns left and right lane indices as a list
        it takes 1D density distribution of a horizontal slice of the window 
        '''

        # we will look for peaks separately in the left and right halves of unwarped
        # binary image and then return them at the end
        midpoint = len(hist) // 2

        left_peak = self.get_a_peak(hist[:midpoint])
        right_peak = self.get_a_peak(hist[midpoint:])

        if (left_peak is None) and (right_peak is None):
            return []
        elif left_peak is None:
            return [right_peak + midpoint]
        elif right_peak is None:
            return [left_peak]
        else:
            return [left_peak, right_peak + midpoint]

        print("left peak:", left_peak)
        print("right peak:", right_peak)

    #-----------------------------------------------------------------------------#

    def get_lanes_mask(self, binary_mask):
        '''
        this routine takes `binary_mask` with applied color masks, Sobel,
        gradient etc. and then applies sliding windows approach to produce a mask
        with two detected lines
        ---------
        First, we get initial positions of lanes by using half of the image.
        Then, we split input image into horizontal strips (8 in this case)
        then, for each strip, we try to detect two peaks, this is where
        centers of lanes are. Then we also create two empy (zero-valued) masks
        for left and right lane. For each peak we will take a 50 pixel window
        to each side of each peak and make this window one-valued in the mask.
        After we did this, we will have two masks and we can fit two polynomials, 
        which will be our left and right lanes.
        '''

        # size of input
        height, width = binary_mask.shape

        # size to go left and right from peak
        fallback_window_size = 70
        window_size = 70
        window_size_increment = 30

        # how many vertical strips to have
        n_splits = 8

        # lanes can't be closer than this (in pixels)
        sanity_threshold = 150

        # for smoothing densities
        rolling_window_size = int(width / 20)

        # default lane positions: just in case none found in the beginning
        x = 0.2
        fallback_peaks = (int(x * width), int((1 - x) * width))

        # use bottom 50% of image to detect initial lanes positions
        initial_lanes = np.mean(binary_mask[height * 5 // 10 : , : ], axis=0)
        initial_lanes = self.moving_average(initial_lanes, rolling_window_size)


        # what if there are no peaks found initially, or just one?
        # this piece of code handles that
        # at the beginnin of video, find initial line locations
        if self.frame_counter == 0:
            peaks_init = self.find_peaks(initial_lanes)
            if len(peaks_init) == 2:
                # both found: noice!
                left_line_index, right_line_index = peaks_init
            elif len(peaks_init) == 1:
                # only one found: the closest to default get assigned to detected position
                # the other stays at default value
                if np.abs(fallback_peaks[0] - peaks_init[0]) > np.abs(fallback_peaks[1] - peaks_init[0]):
                    left_line_index = peaks_init[0]
                    right_line_index = fallback_peaks[1]
                else:
                    left_line_index = fallback_peaks[0]
                    right_line_index = peaks_init[0]
            else:
                # none found: both stay at default values
                left_line_index, right_line_index = fallback_peaks
        else:

            # use mask from previous fits
            left_line_index = self.line_left_start
            right_line_index = self.line_right_start
            # apply detected masks to the initial image values and return result
            left_masked = np.zeros_like(binary_mask)
            right_masked = np.zeros_like(binary_mask)

            left_masked[(binary_mask > 0.1) & (self.mask_left) > 0.1] = 1
            right_masked[(binary_mask > 0.1) & (self.mask_right) > 0.1] = 1

            # return masks for left and right line regions individually, used later for fitting
            return left_masked, right_masked, self.mask_left, self.mask_right


        # get indices for left window left and right borders
        left_window_min = left_line_index - window_size
        left_window_max = left_line_index + window_size

        # get indices for right window left and right borders
        right_window_min = right_line_index - window_size
        right_window_max = right_line_index + window_size

        # empty (for now) masks used later to fit polynomials
        left_line_mask = np.zeros_like(binary_mask)
        right_line_mask = np.zeros_like(binary_mask)

        # when iterating, we need to keep track of last step's indices
        # in case we will not find any peaks during some step
        left_line_index_last = left_line_index
        right_line_index_last = right_line_index

        # iterate through all `n_splits` stripes of input image
        for i in range(n_splits):
            # define low and high vertical coordinate of stripe
            # convert to int because these are indices
            low = int(height * (1.0 - i / n_splits))
            high = int(height * (1 - (i + 1) / n_splits))
            
            # define vertical pixel density for each strip
            # index from high to low because (0,0) point is upper left
            strip_hist = np.mean(binary_mask[high:low, :], axis=0)
            # moving average in order to smooth out peaks
            strip_hist = self.moving_average(strip_hist, rolling_window_size)
            # retreive the peaks
            peaks = self.find_peaks(strip_hist)
            
            # there are 3 cases for peaks detected at each iteration: 
            #    a) 2 peaks (the best case) – noice!
            #    b) 1 peak
            #    c) no peaks
            if len(peaks) == 2:
                # update left and right indices with new values
                left_line_index, right_line_index = peaks
                window_size = fallback_window_size
            elif len(peaks) == 1:
                # update the closest of old peaks to the new peak
                # the other one update with old value
                if np.abs(left_line_index_last - peaks[0]) > np.abs(right_line_index_last - peaks[0]):
                    left_line_index = peaks[0]
                    right_line_index = right_line_index_last
                else:
                    left_line_index = left_line_index_last
                    right_line_index = peaks[0]

                window_size += window_size_increment
            else:
                # no peaks found – update both using old values
                left_line_index = left_line_index_last
                right_line_index = right_line_index_last
                window_size += window_size_increment
                
            # sanity check / outlier removal
            # if new values are more than `sanity_threshold` away
            # from old values, then disregard new values
            if np.abs(left_line_index - left_line_index_last) > sanity_threshold:
                left_line_index = left_line_index_last
                window_size += window_size_increment
            if np.abs(right_line_index - right_line_index_last) > sanity_threshold:
                right_line_index = right_line_index_last
                window_size += window_size_increment
            
            
            # get indices for left window left and right borders
            left_window_min = left_line_index - window_size
            left_window_max = left_line_index + window_size

            # get indices for right window left and right borders
            right_window_min = right_line_index - window_size
            right_window_max = right_line_index + window_size
            
            # apply ones in the detected lines region
            left_line_mask[high:low, left_window_min:left_window_max] = 1
            right_line_mask[high:low, right_window_min:right_window_max] = 1
            
            # update before next step
            left_line_index_last = left_line_index
            right_line_index_last = right_line_index


        # apply detected masks to the initial image values and return result
        left_masked = np.zeros_like(binary_mask)
        right_masked = np.zeros_like(binary_mask)

        left_masked[(binary_mask > 0.1) & (left_line_mask) > 0.1] = 1
        right_masked[(binary_mask > 0.1) & (right_line_mask) > 0.1] = 1

        # return masks for left and right line regions individually, used later for fitting
        return left_masked, right_masked, left_line_mask, right_line_mask
        # return left_line_mask, right_line_mask

    def fit_line(self, lane_mask, side, smoothing=0.25):
        '''
        fits a quadratic polynomial based on a lane mask;
        it uses exponential smoothing as weighted average
        of fit from last frame and new fit from current frame
        '''
        indices = lane_mask.nonzero()
        # for the purpose of fitting, x is vertical coordinates
        x_vals = indices[0]
        # y is horizontal coordinate
        y_vals = indices[1]
        # returns tuple of 3 coefficients
        if len(x_vals) == 0:
            # return line from last frame
            if side == 'left':
                return self.left_fit
            elif side == 'right':
                return self.right_fit
        new_fit = np.polyfit(x_vals, y_vals, 2)

        if side == 'left':
            old_fit = self.left_fit
        elif side == 'right':
            old_fit = self.right_fit

        # apply smoothing if already fit exists
        if (self.left_fit is not None) and (self.right_fit is not None):
            result = smoothing * new_fit + (1 - smoothing) * old_fit
        else:
            result = new_fit

        # update state with latest result
        if side == 'left':
            self.left_fit = result
            self.line_left_start = int(result[0] * self.height**2 + result[1] * self.height + result[2])
            self.mask_left = self.get_fitted_mask(result)
        elif side == 'right':
            self.right_fit = result
            self.line_right_start = int(result[0] * self.height**2 + result[1] * self.height + result[2])
            self.mask_right = self.get_fitted_mask(result)
        return result

    def get_fitted_mask(self, fitted, n_splits=8, window_size=70):
        mask = np.zeros((self.height, self.width)).astype(np.uint8)
        '''
        this routine updates the mask for the next frame based on fit in
        the current frame, so that we do not need to search for the mask
        from scratch
        (intuition: in the next fram line cannot be further away than some
        `window_size` in pixels)
        '''

        # iterate through all `n_splits` stripes of input image
        for i in range(n_splits):
            # define low and high vertical coordinate of stripe
            # convert to int because these are indices
            low = int(self.height * (1.0 - i / n_splits))
            high = int(self.height * (1 - (i + 1) / n_splits))
            
            line_index = int(fitted[0]*low*low + fitted[1]*low + fitted[2])

            # get indices for left window left and right borders
            window_min = line_index - window_size
            window_max = line_index + window_size
            
            # apply ones in the detected lines region
            mask[high:low, window_min:window_max] = 1

        return mask

    def get_line_points(self, fitted_line, lane_mask):
        '''
        this routine takes the fitted line and a lane mask
        returns two lists of coordinates: vertical and horizontal
        '''
        height = lane_mask.shape[0]
        # y is vertical coordinates
        vert_coords = np.array(range(height), dtype=np.uint32)
        # x is horizontal coordinates
        hor_coords = np.uint32(fitted_line[0] * vert_coords**2 + fitted_line[1] * vert_coords + fitted_line[2])
        return (vert_coords, hor_coords)

    def prepare_poly_points(self, left_points, right_points):
        '''
        this routine takes two points tuples for left and right lanes
        and prepares an ordered list of points that can be used
        by cv2.fillPoly to overlay detected lane on top of original image;
        it also prepares points to be used for drawing fitted lines
        '''
        vert_left = left_points[0]
        hor_left = left_points[1]
        vert_right = right_points[0]
        hor_right = right_points[1]
        
        # points for left line go from top to bottom
        left_line_points = np.array([np.transpose(np.vstack([hor_left, vert_left]))]).astype(np.int32)
        # points for right line go from bottom to top, hence `np.flipud`
        right_line_points = np.array([np.flipud(np.transpose(np.vstack([hor_right, vert_right])))]).astype(np.int32)
        # stack them horizontally (along axis 1)
        all_points = np.hstack((left_line_points, right_line_points))
        
        return all_points, left_line_points, right_line_points

    def prepare_lane_overlay(self, all_points, left_points, right_points, image, unwarp_mtx, fill_color=(0,200,0)):
        '''
        this routine prepares an outlay to be put on top of original image for 
        final video processing step
        '''
        height, width = image.shape[:2]
        overlay = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(overlay, all_points, fill_color)
        cv2.polylines(overlay, left_points, isClosed=False, color=(0,0,255), thickness=15)
        cv2.polylines(overlay, right_points, isClosed=False, color=(0,0,255), thickness=15)
        unwarped_overlay = cv2.warpPerspective(overlay, unwarp_mtx, (width, height))
        return unwarped_overlay, overlay

    def distance_from_center(self):
        '''
        off-center distance calculation, based on 2 assumptions:
            1) camera mounted in the center of vehicle
            2) road lane is 3.7 meters wide
        '''
        normalization = self.lane_width / (self.line_right_start - self.line_left_start)
        lane_center = (self.line_left_start + self.line_right_start) / 2
        car_center = self.width / 2 # assume camera mounted in the center
        distance = (lane_center - car_center) * normalization
        return distance

    def get_curvature(self):
        '''
        this calculates curvature according to resource provided in class
        assumption is made about length of dashed road lines
        '''
        # normalization coefficient converts pixels into meters based on assumtion of 10 feet line length
        normalization = 3.048 / 8.68
        # caluclate "average" from both fitted lanes
        fit = 0.5 * self.left_fit + 0.5 * self.right_fit
        # fit = self.left_fit
        # at which vertical position to calculate?
        position = self.height ## at the position of car (bottom of image)
        # based on averaged quadratic fir calculate radius of curvature
        radius = (1 + (2 * fit[0] * 720 + fit[1])**2)**1.5 / np.abs(2 * fit[0])
        # radius is in pixels of wrapped perspective image, need to convert to meters
        curvature = radius * normalization
        return curvature