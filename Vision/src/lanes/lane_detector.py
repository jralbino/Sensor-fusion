import cv2
import numpy as np

class LaneDetector:
    def __init__(self, debug=False):
        """
        Classic lane detector using Sliding Windows and Polynomials.
        """
        self.debug = debug
        
        # Previous state to smooth detections between frames (stable video)
        self.left_fit_prev = None
        self.right_fit_prev = None
        self.sanity_check_passes = 0

    def preprocess(self, img):
        """Applies color and gradient thresholds to isolate lines."""
        # 1. Convert to HLS (Hue, Lightness, Saturation)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # 2. Filter whites (High luminosity)
        white_mask = cv2.inRange(l_channel, 200, 255)
        
        # 3. Filter yellows (Specific Hue + High saturation)
        # Yellow Hue in OpenCV is approx 20-30
        yellow_lower = np.array([15, 30, 115])
        yellow_upper = np.array([35, 204, 255])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        
        # 4. Combine
        combined_binary = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Morphological cleaning (remove noise)
        kernel = np.ones((5,5), np.uint8)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)
        
        return combined_binary

    def perspective_transform(self, img_binary):
        """Transforms front view to 'Bird's Eye View'."""
        h, w = img_binary.shape
        
        # Hand-defined points for BDD100K (1280x720)
        # Adjusted to focus on the lane in front of the car
        src = np.float32([
            [200, 720],  # Bottom Left
            [1100, 720], # Bottom Right
            [595, 450],  # Top Left (Horizon)
            [685, 450]   # Top Right (Horizon)
        ])
        
        # Destination: Flat rectangle
        dst = np.float32([
            [300, h],       # Bottom Left
            [980, h],       # Bottom Right
            [300, 0],       # Top Left
            [980, 0]        # Top Right
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src) # To go back
        
        warped = cv2.warpPerspective(img_binary, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, Minv

    def find_lines_sliding_window(self, binary_warped):
        """Finds line pixels using histograms."""
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0]//2)
        
        # Histogram peaks = line base
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Hyperparameters
        nwindows = 9
        margin = 100
        minpix = 50
        window_height = int(binary_warped.shape[0]//nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = []
        right_lane_inds = []
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            # X limits
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify pixels within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Re-center if enough pixels are found
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract coordinates
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # 2nd degree Polynomial Fit (Parabola): x = Ay^2 + By + C
        # If nothing is detected, use the previous one (simple smoothing)
        if len(leftx) == 0 or len(rightx) == 0:
            return self.left_fit_prev, self.right_fit_prev

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Save for the next frame
        self.left_fit_prev = left_fit
        self.right_fit_prev = right_fit
        
        return left_fit, right_fit

    def draw_lane(self, original_img, binary_warped, left_fit, right_fit, Minv):
        """Paints the lane area on the original image."""
        h, w = binary_warped.shape
        ploty = np.linspace(0, h-1, h)
        
        # Generate X points from the polynomial
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create empty image for painting
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Points for polygon
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw green lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Undo perspective (return to front view)
        newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
        
        # Combine
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return result

    def detect(self, img):
        """
        Main pipeline.
        Returns: Image with painted lane.
        """
        # Copy to not alter original
        out_img = np.copy(img)
        
        # 1. Preprocessing (Color -> Binary)
        binary = self.preprocess(out_img)
        
        # 2. Perspective (Frontal -> Aerial)
        warped, Minv = self.perspective_transform(binary)
        
        # 3. Find Lines (Math)
        left_fit, right_fit = self.find_lines_sliding_window(warped)
        
        # 4. Draw and return to original perspective
        if left_fit is not None and right_fit is not None:
            result = self.draw_lane(out_img, warped, left_fit, right_fit, Minv)
            return result
        else:
            return out_img # If it fails, return original