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
import os


# ### [function] moving_average

# In[2]:


def moving_average(x, w):
    vectLen = len(x)
    if vectLen > w:
        sol = np.convolve(x, np.ones(w), 'valid') / w
    else:
        sol = sol = np.convolve(x, np.ones(vectLen), 'valid') / vectLen
    return sol


# ### [function] Resize Images

# In[3]:


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

# In[4]:


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

# In[5]:


def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# ### [function] Color enhancement

# In[6]:


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
    color1_hls = (90, 120, 80)
    color2_hls = (110, 255, 255)
    mask1 = cv2.inRange(hls, color1_hls,color2_hls)
    
    #find whites
    color1_hls_w = (0, 199, 0)
    color2_hls_w = (180, 255, 255)
    mask2 = cv2.inRange(hls, color1_hls_w,color2_hls_w)
    
    #    Add masks together
    mask = cv2.bitwise_or(mask1, mask2)
    
    res = cv2.bitwise_and(img,img, mask= mask)
    return res


# ### [function] Grayscale

# In[7]:


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

# In[8]:


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# ### [function] sobel_thresh

# In[9]:


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
    
    # 6) Diretion threshold [0 to ??/2.] 0 is vertical an ??/2 horizontal
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    #Converted to 8 bits 0 is vertical an ??/2 horizontal
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

# In[10]:


def region_of_interest(img, vertices, overplot=False):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    
    Be aware that if overplot=True image will have 3 color dimensions
    instead of one!
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
    
    if overplot:    
        rgbROI = np.dstack((img*255, img*255, img*255))
        maskContour = cv2.polylines(rgbROI,[vertices],True,(0,0,255),thickness=5)
        masked_image = cv2.addWeighted(rgbROI, 1, maskContour, 0.8, 0)

    
    return masked_image


# ### [function] draw_lines

# In[11]:


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

# In[12]:


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

# In[13]:


def warp_image(img,conversion = 'warp' ,hwidth = 250 ,offset = -0, height = -450, overplotLinesSrc= False, overplotLinesDst= False ):
    
    #Source
    # Place source points for uimage wrapping
    dotS_UL=[592,450]; dotS_UR= [692,450]
    dotS_LL=[195,720] ; dotS_LR= [1120,720]

    src= np.array([dotS_UL,dotS_LL,dotS_LR,dotS_UR], dtype=np.float32)



    #Destination
    dotD_UL=[offset+(1280//2)-hwidth,height]; dotD_UR= [offset+(1280//2)+hwidth,height]
    dotD_LL=[offset+(1280//2)-hwidth,720] ; dotD_LR= [offset+(1280//2)+hwidth,720]

    dst= np.array([dotD_UL,dotD_LL,dotD_LR,dotD_UR], dtype=np.float32)
        

    #Computye perspective transform
    M = cv2.getPerspectiveTransform(dst,src)
    Minv = cv2.getPerspectiveTransform(src, dst)
    if conversion == 'unwarp':
        warped = cv2.warpPerspective(img, M, (1280,720), flags=cv2.INTER_LINEAR)
    else:
        warped = cv2.warpPerspective(img, Minv, (1280,720), flags=cv2.INTER_LINEAR)
        
        
    # Create and plot source plane   
    if overplotLinesSrc or overplotLinesDst:
        if len(warped.shape) > 2:
            pass
        else:
            warped = np.dstack((warped*255, warped*255, warped*255))  
        # Plot lines
        if overplotLinesSrc:
            cv2.polylines(warped,np.array([src], dtype=np.int32),False,(255,0,0),thickness=4)    
        if overplotLinesDst:
            cv2.polylines(warped,np.array([dst], dtype=np.int32),False,(0,0,255),thickness=4)
        
    
    return warped, M, Minv


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


def find_lane_pixels(binary_warped, leftx_base, rightx_base, showRectangles = True):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 8
    # Set the width of the windows +/- margin
    margin = 120
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
        if showRectangles:
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
    out_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, out_img


# ### [function] fit_polynomial

# In[16]:


def fit_polynomial(binary_warped,xPixels,yPixels, drawPoly = True):
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        # Fit a second order polynomial to each using `np.polyfit`
        coeffs_fit = np.polyfit(yPixels, xPixels, 2)
        line_fitx = np.polyval(coeffs_fit, ploty)  # evaluate the polynomial
        lineDetected = True
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        coeffs_fit = [np.array([False])]  
        line_fitx = [np.array([False])]  
        lineDetected = False
        return coeffs_fit, lineDetected, line_fitx, ploty, out_img
        

    # Plots the left and right polynomials on the lane lines
    if drawPoly:
        verts = np.array(list(zip(line_fitx.astype(np.int32),ploty.astype(np.int32))))
        line_img = cv2.polylines(binary_warped,[verts],False,(0,255,0),thickness=4)
    
        out_img = cv2.addWeighted(line_img, 1, binary_warped, 1, 0) 
    else:
        out_img = binary_warped


    return  coeffs_fit, lineDetected, line_fitx, ploty, out_img


# ### [function]search_around_poly

# In[17]:


def search_around_poly(binary_warped, lineLane):
    # Create an output image to draw on and visualize the result
    if len(binary_warped.shape) < 3:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    else:
        out_img = binary_warped
    
    # Width of the margin around the previous polynomial to search
    margin = 100
        
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    lane_inds = ((nonzerox > (lineLane.poly_best_fit[0]*(nonzeroy**2) + lineLane.poly_best_fit[1]*nonzeroy + 
                lineLane.poly_best_fit[2] - margin)) & (nonzerox < (lineLane.poly_best_fit[0]*(nonzeroy**2) + 
                lineLane.poly_best_fit[1]*nonzeroy + lineLane.poly_best_fit[2] + margin)))
    
    leftx = nonzerox[lane_inds]
    lefty = nonzeroy[lane_inds] 
    
    if len(leftx) < 20 or len(leftx) < 20: 
        lineDetected = False
        print("NO pixels with search around poly")
        coeffs_fit = [0,0,0]
        line_fitx = []
        return leftx,lefty, coeffs_fit, lineDetected, line_fitx, out_img
    else:
        lineDetected = True
        
    coeffs_fit = np.polyfit(lefty, leftx, 2)
    line_fitx = np.polyval(coeffs_fit, lineLane.poly_ploty)  # evaluate the polynomial   
    
    
        
    out_img[lefty, leftx] = [255, 0, 0]
    
    return leftx,lefty, coeffs_fit, lineDetected, line_fitx, out_img


# ### [function] similarCurvature

# In[18]:


def similarCurvature(lineLeft,lineRight):
    """This function evaluates if both lanes 
    are concave/convex
    """
    # TODO
    return True


# ### [function] rightSeparation

# In[19]:


def rightSeparation(left_fitx, right_fitx, limitDist = 50):
    """This function evaluates if both lanes 
    are separated rougly the same ammount
    it takes into consideration only the maximum deviation
    """       
    dist = right_fitx-left_fitx
    maxDist = dist.max() - dist.min()
    #print(maxDist)
    if  maxDist < limitDist:
        return True
    else:
        return False


# ### [function] areParallel

# In[20]:


def areParallel(lineLeft,lineRight, margin=100):
    """This function evaluates both line 
    polynoms at different y heights and
    detects if the separation is within 
    a specific range thorugghout the frame  
    """
    # TODO
    return True


# ### [function] calculateDeviation

# In[21]:


def calculateDeviation(img, lineLeft,lineRight, ):
    """This function calculates 
    the deviation of the vehicle from the center of the 
    image
    """
    frameCenter = np.mean([lineLeft.bestx,lineRight.bestx] , dtype=np.int32)
    imgCenter = img.shape[1]//2
    
    dev = frameCenter - imgCenter
    
    xm_per_pix = 3.7/450 # meters per pixel in x dimension
    
    result = dev*xm_per_pix
    
    # Moving average deviation (Not needed as applied to bestx)
    #x = np.append(lineLeft.center_deviation, [dev])
    #result = moving_average(x, movingAvg)[-1]
    #lineLeft.center_deviation = np.append(lineLeft.center_deviation, result)  
    
    if dev > 0.01:
        text = "Vehicle is {:.2f} m -->".format(abs(result))
    elif dev < -0.01:
        text = "Vehicle is {:.2f} m <--".format(abs(result))
    else:
        text = "Vehicle is spot on center!"
    
    
    return result , text


# ### [function]checkRadius

# In[22]:


def checkRadius(lineLeft, lineRight, limitStraight=10000,movingAvg = 1 ):
    
    if (lineLeft.radius_of_curvature and lineRight.radius_of_curvature ):
    
        diff = abs(lineLeft.radius_of_curvature-lineRight.radius_of_curvature)

        mean = np.mean([lineLeft.radius_of_curvature ,lineRight.radius_of_curvature] )

        x = np.append(lineLeft.list_radius, [mean])

        result = moving_average(x, movingAvg)[-1]

        lineLeft.list_radius = np.append(lineLeft.list_radius, result)   


        text = "Radius of curvature {:.0f} m".format(abs(result)) 
        if result > limitStraight:
            text = "Straight Line".format(abs(result))

    else:
        diff = 0
        result = 0
        text = "No line found"

    return diff, result, text


# # Convert notebook into .py script

# In[23]:


import os
# Remove existing .py script if exists
try:
    os.remove("0. Functions_Clases Pipeline.py")
except:
    pass


get_ipython().system('jupyter nbconvert --to script "0. Functions_Clases Pipeline.ipynb"')


# In[ ]:




