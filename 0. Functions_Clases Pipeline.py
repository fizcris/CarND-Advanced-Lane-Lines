#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pprint import pprint
import pickle


# ### [function] Resize Images

# In[2]:


def resizeImage(image,shape=(1280,720)):
    """
    image should be a np.array image, note that it will be modified
    """
    ysize = image.shape[0]
    xsize = image.shape[1]

    # Resize image if necesaary
    if xsize != shape[0] and ysize != shape[1]:
        image = cv2.resize(image, (shape[0], shape[1]),)

    return image


# ### [function] Display a list of images

# In[3]:


def displayListImages(images,titles=[],cols=1,cmap=None, figSize = [12,12], overplot = None):
    """
    Function to display and resize a list of images
    images is a list of matplotlib image (use imread)
    titles is a list of strings
    cols is an integer with the number of columns to display
    rows is an integer with the number of rows to display
    """
    rows = len(images)//cols
    if len(images)%cols > 0:
        rows+=1

    # Helper to adapt images to full width
    plt.rcParams['figure.figsize'] = [figSize[0], figSize[1]*cols]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

    for i in range(len(images)):
        
        plt.subplot(rows, cols, i+1)
        
        image = resizeImage(images[i])
        if titles:
            plt.title(titles[i]) 
        plt.imshow(image, cmap=cmap, aspect='auto')
        
    if overplot:
        overplot
    return plt

        
#Test function
#displayListImages(cal_images,cols=2)


# ### [function] Undistort images

# In[4]:


def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# ### [function] Color enhancement

# In[5]:


def colorEnhancement(img):
    """Converts the image to HSL
    Creates two masks to filter white and
    lleyow lines
    Applies the mask
    Be carefull to input an 8bit image!
    cv2.IMREAD_COLOR is your friend when using imgread
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    ##find lleyows
    color1_hls = (70, 120, 0)
    color2_hls = (100, 255, 255)
    mask1 = cv2.inRange(hls, color1_hls,color2_hls)
    
    #find whites
    color1_hls_w = (0, 220, 0)
    color2_hls_w = (180, 255, 255)
    mask2 = cv2.inRange(hls, color1_hls_w,color2_hls_w)
    
    #    Add masks together
    mask = cv2.bitwise_or(mask1, mask2)
    
    res = cv2.bitwise_and(img,img, mask= mask)
    return res


# ### [function] Grayscale

# In[6]:


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
    Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ### [function] gaussian_blur

# In[7]:


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# ### [function] sobel_thresh

# In[8]:


def sobel_thresh(img, sobel_kernel=3, x_thresh=[1,255], y_thresh=[1,255], mag_thresh=[1,255], dir_thresh=[-np.pi/2, np.pi/2]):

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    scaled_sobelx = np.uint8(255*sobelx/np.max(sobelx))
    
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    scaled_sobely = np.uint8(255*sobely/np.max(sobely))
    
    # 3) Calculate the magnitude 
    square_sobelx = np.square(sobelx)
    square_sobely = np.square(sobely)
    abs_sobelxy = np.sqrt(np.add(square_sobelx,square_sobely))   
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
    # 5) Create a binary mask where mag thresholds are met
    mag_binary  = np.zeros_like(scaled_sobelxy)
    mag_binary [(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    
    # 6) Diretion threshold [0 to π/2.] 0 is vertical an π/2 horizontal
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    #Converted to 8 bits 0 is vertical an π/2 horizontal
    scaled_sobelxydir = np.uint8(255*absgraddir/np.max(absgraddir))
    
    dir_binary  =  np.zeros_like(scaled_sobelxydir)
    dir_binary [(scaled_sobelxydir >= dir_thresh[0]) & (scaled_sobelxydir <= dir_thresh[1])] = 1
    
    # 7) Sobel x
    gradx = np.zeros_like(scaled_sobelx)
    gradx[(scaled_sobelx >= x_thresh[0]) & (scaled_sobelx <= x_thresh[1])] = 1
    
    # 8) Sobel y
    grady = np.zeros_like(scaled_sobely)
    grady[(scaled_sobely >= y_thresh[0]) & (scaled_sobely <= y_thresh[1])] = 1
    
    # Create a copy and apply the threshold    
    combined = np.zeros_like(scaled_sobelx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Return the result
    return combined


# ### [function] region_of_interest

# In[9]:


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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


# ### [function] draw_lines

# In[10]:


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """ 
    for line in lines:
        #print(line)
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# ### [function] hough_lines

# In[11]:


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be a np.array image
    Returns an image with hough lines drawn and the hough lines points.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    
    try:
        draw_lines(line_img, lines, color=[255, 0, 0], thickness=4)
    except:
        lines = []
        print("No line found")
        pass
    return line_img,lines


# ### [function] Warp images

# In[12]:


def warp_image(img,conversion = 'warp' ,hwidth = 250 ,offset = -0, height = -450, overplotLines= True ):
    
    #Source
    # Place source points for uimage wrapping
    dotS_UL=[592,450]; dotS_UR= [692,450]
    dotS_LL=[195,720] ; dotS_LR= [1120,720]

    src= np.float32([dotS_UL,dotS_LL,dotS_LR,dotS_UR])

    # Create and plot source plane
    xs = [x[0] for x in src]
    ys = [x[1] for x in src]

    if overplotLines:
        plt.plot(xs,ys, 'ro-')

    #Destination
    dotD_UL=[offset+(1280//2)-hwidth,height]; dotD_UR= [offset+(1280//2)+hwidth,height]
    dotD_LL=[offset+(1280//2)-hwidth,720] ; dotD_LR= [offset+(1280//2)+hwidth,720]

    dst= np.float32([dotD_UL,dotD_LL,dotD_LR,dotD_UR])

    xd = [x[0] for x in dst]
    yd = [x[1] for x in dst]

    if overplotLines:
        plt.plot(xd,yd, 'bo-')

    # Create and plot source plane
    xd = [x[0] for x in dst]
    yd = [x[1] for x in dst]
    
    #Computye perspective transform
    M = cv2.getPerspectiveTransform(dst,src)
    Minv = cv2.getPerspectiveTransform(src, dst)
    if conversion == 'unwarp':
        warped = cv2.warpPerspective(img, M, (1280,720), flags=cv2.INTER_LINEAR)
    else:
        warped = cv2.warpPerspective(img, Minv, (1280,720), flags=cv2.INTER_LINEAR)
        
    
    return warped, M, Minv


# ### [Class] Line

# In[13]:


class Line():
# Class to receive the characteristics of each line detection
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # missdetections (resets every good detection)
        self.missdetections = 0  
        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the curernt iteration
        self.currentx = 0  
        #average x values of the fitted line over the last n iterations
        self.bestx = 0  
        
        #polynomial coefficients averaged over the last n iterations
        self.poly_best_fit = np.array([False])    
        #polynomial coefficients for the most recent fit
        self.poly_current_fit = np.array([False])  
        #difference in fit coefficients between last and new fits
        self.poly_diffs = np.array([False]) 
        
        #Evaluated polynomial over line X points
        self.poly_plotx = np.array([False])  
        #Evaluated polynomial over line Y points
        self.poly_ploty = np.array([False])  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        
        
        
    def isGoodPolyFit(self):
        if self.best_fit[0]: #Check if we are not in the first iteration
            if  np.isclose(self.current_fit,self.best_fit,[1e2,1e2,1e2]):
                self.best_fit = current_fit
    
    def updateXbase(self, currentx, limit=50):
        """Updates the bestx with the given currentx
        if the difference is below a threshold gets appeded
        othewise discarded
        """
        self.currentx = currentx 
        if self.bestx > 0:
            if (abs(self.currentx - self.bestx) < limit):
                self.bestx = self.currentx
        else:
            self.bestx = self.currentx
        
        self.recent_xfitted.append(self.bestx)
    
    def updatePixels(self,allx,ally):
        """Updates the line pixels positions
        for the current frame
        """
        self.allx = allx
        self.ally = ally
               
    def updateCoeffsLine(self,detected, current_fit, left_fitx, ploty ):
        """Updates the line ploynom equation coeficients
        for the current frame
        """
        if detected:
            self.detected = True
            self.missdetections = 0  
            # Calculate diffs
            if any(self.poly_current_fit): 
                self.poly_diffs = np.subtract(current_fit,self.poly_current_fit)
            
            self.poly_current_fit = current_fit
            self.poly_plotx = left_fitx
            self.poly_ploty = ploty
        else: 
            self.detected = False
            self.missdetections += 1  
            
            
    def measure_real_curvature(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/450 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        y_eval = np.max(ploty)

        ##### Implement the calculation of R_curve (radius of curvature) #####
        self.radius_of_curvature = ((1 + (2*self.best_fit[0]*y_eval*ym_per_pix + self.best_fit[1])**2)**1.5) / np.absolute(2*self.best_fit[0])


# ### [function] find_lane_x_points

# In[14]:


def find_lane_x_points(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base,rightx_base


# ### [function] find_lane_pixels

# In[15]:


def find_lane_pixels(binary_warped, leftx_base, rightx_base):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 40
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    
    nonzero = binary_warped.nonzero() # Returns indeices of noncero elements
    
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    fillx_R = np.array(False)
    filly_R = np.array(False)
    fillx_L = np.array(False)
    filly_L = np.array(False)

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        #The four below boundaries of the window 
        win_xleft_low =  leftx_current - margin//2
        win_xleft_high = leftx_current  + margin//2
        
        win_xright_low = rightx_current  - margin//2
        win_xright_high = rightx_current  + margin//2
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        

        ###Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        if window == 0:
            for  y in range (win_y_low,win_y_high):
                thisx_R = np.linspace(win_xright_low,win_xright_high, 100, dtype=np.int32)               
                fillx_R = np.append(fillx_R,thisx_R )
                filly_R = np.append(filly_R,np.ones(thisx_R.shape, dtype=np.int32)*y)
                
                thisx_L = np.linspace(win_xleft_low,win_xleft_high, 100, dtype=np.int32)
                fillx_L = np.append(fillx_L,thisx_L )
                filly_L = np.append(filly_L,np.ones(thisx_L.shape, dtype=np.int32)*y)
               
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #If you found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Seaparate left and right o avoid error propagation
    try:
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds] 
    lefty = nonzeroy[left_lane_inds] 
    
    leftx = np.concatenate((leftx,fillx_L))
    lefty = np.concatenate((lefty,filly_L))

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    rightx = np.concatenate((rightx,fillx_R))
    righty = np.concatenate((righty,filly_R))
    
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [255, 255, 0]

    return leftx, lefty, rightx, righty, out_img


# ### [function] fit_polynomial

# In[16]:


def fit_polynomial(binary_warped,xPixels,yPixels):
    
    # Fit a second order polynomial to each using `np.polyfit`
    coeffs_fit = np.polyfit(yPixels, xPixels, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        line_fitx = np.polyval(coeffs_fit, ploty)  # evaluate the polynomial
        lineDetected = True
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        coeffs_fit = [np.array([False])]  
        line_fitx = [np.array([False])]  
        lineDetected = False
        

    # Plots the left and right polynomials on the lane lines
    verts = np.array(list(zip(line_fitx.astype(np.int32),ploty.astype(np.int32))))
    line_img = cv2.polylines(binary_warped,[verts],False,(0,0,255),thickness=4)
    
    out_img = cv2.addWeighted(line_img, 1, binary_warped, 1, 0) 


    return  coeffs_fit, lineDetected, line_fitx, ploty, out_img


# ### [function] similarCurvature

# In[17]:


def similarCurvature(lineLeft,lineRight):
    """This function evaluates if both lanes 
    are concave/convex
    """
    return True


# ### [function] rightSeparation

# In[18]:


def rightSeparation(left_fitx, right_fitx, limitDist = 50):
    """This function evaluates if both lanes 
    are separated rougly the same ammount
    it takes into consideration only the maximum deviation
    """
    dist = right_fitx-left_fitx
    maxDist = dist.max() - dist.min()
    if  maxDist < limitDist:
        return True
    else:
        return False


# ### [function] areParallel

# In[19]:


def areParallel(lineLeft,lineRight, margin=100):
    """This function evaluates both line 
    polynoms at different y heights and
    detects if the separation is within 
    a specific range thorugghout the frame  
    """
    
    return True


# # Convert notebook into .py script

# In[20]:


get_ipython().system('jupyter nbconvert --to script "0. Functions_Clases Pipeline.ipynb"')

