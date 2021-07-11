class Line():
# Class to receive the characteristics of each line detection
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # missdetections (resets every good detection)
        self.missdetections = 0  
        
        # x values of the last n fits of the line
        self.recent_xfitted = np.array([0])
        #average x values of the curernt iteration
        self.currentx = 0  
        #average x values of the fitted line over the last n iterations
        self.bestx = 0  
        
        #polynomial coefficients averaged over the last n iterations
        self.poly_best_fit = np.array([]) 
        # Staged coefs to handle missdetections    
        self.poly_best_fit_staged = np.array([]) 
        #polynomial coefficients for the most recent fit
        self.poly_current_fit = np.array([]) 
        #difference in fit coefficients between last and new fits
        self.poly_diffs = np.array([])
        #All difference in fit coefficients between last and new fits
        self.all_poly_diffs = np.array([[0,0,0]]) 
        #Last fitted coefs
        self.recent_poly_fits = np.array([[0,0,0]]) 
        
        #Evaluated polynomial over line X points
        self.poly_plotx = np.array([])  
        self.poly_plotx_staged = np.array([])  
        #Evaluated polynomial over line Y points
        self.poly_ploty = np.linspace(0, 720-1, 720 )

        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        self.list_radius = np.array([]) 
        
        # center deviation
        self.center_deviation = np.array([]) 
    
    def updateXbase(self, currentx, limit=100, movingAvg = 10 ):
        """Updates the bestx with the given currentx
        if the difference is below a threshold gets appeded
        othewise discarded
        """
         # Not First iteration
        if np.any((self.recent_xfitted != 0)):
            # Check for outliers
            if (abs(currentx - self.bestx) < limit):
                self.currentx = currentx
            else:
                print("Discarded X point")
                self.currentx = self.bestx
            
            # Apply moving average
            x = np.append(self.recent_xfitted, [self.currentx])
            self.bestx = moving_average(x, movingAvg)[-1]
        
        
        else:
            # First iteration
            self.currentx = currentx
            self.bestx = currentx
            self.recent_xfitted = currentx
            
        self.recent_xfitted = np.append(self.recent_xfitted, self.bestx)  
    
    def updatePixels(self,allx,ally):
        """Updates the line pixels positions
        for the current frame
        """
        self.allx = allx
        self.ally = ally
        
    def measure_real_curvature(self, amplif = 800):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        my = 30/(720 + amplif) # meters per pixel in y dimension
        mx = 3.7/700 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.poly_ploty)

        a =self.poly_best_fit[0] * (my**2/ mx)
        b =self.poly_best_fit[1] *(my/mx)

        ##### Implement the calculation of R_curve (radius of curvature) #####
        res = ((1 + (2*a*y_eval + b)**2)**1.5) / np.absolute(2*a)
        self.radius_of_curvature = res *  (my**2/ mx)

    def updateCoeffsLine(self,detected, current_fit, left_fitx, ploty, coefLimits=[1,1,10], movingAvg=5 ):
        """Updates the line ploynom equation coeficients
        for the current removing outliers and applying moving average filters
        to coeffs
        """
        # Not First iteration
        if  np.any((self.recent_poly_fits != 0)):
            if detected:
                self.detected = True
                if any(current_fit): 
                    self.poly_diffs = np.subtract(self.poly_best_fit,current_fit)
                    self.all_poly_diffs = np.vstack((self.all_poly_diffs,self.poly_diffs)) 
                # If outlier
                if (abs(self.poly_diffs[0]) > coefLimits[0] or abs(self.poly_diffs[1]) > coefLimits[1] or abs(self.poly_diffs[2]) > coefLimits[2] ):
                    print("missdetection")
                    print(self.poly_diffs)
                    self.all_poly_diffs = self.all_poly_diffs[:-1,:]
                    self.detected = False
                    self.missdetections += 1 
                
                else:# If not outlier (Good detection)
                    self.detected = True
                    self.missdetections = 0 
                    # Mean average filter coefs                    
                    x = np.vstack((self.recent_poly_fits,current_fit)) 
                    c0 = moving_average(x[:,0], movingAvg)[-1]
                    c1 = moving_average(x[:,1], movingAvg)[-1]
                    c2 = moving_average(x[:,2], movingAvg)[-1]
                    self.poly_best_fit = np.array([c0,c1,c2]) 
                    self.recent_poly_fits = np.vstack((self.recent_poly_fits,self.poly_best_fit)) 
                    #print(self.recent_poly_fits)               
                    self.poly_plotx = np.polyval(self.poly_best_fit, self.poly_ploty)
                    
            else: #Not detected
                self.detected = False
                self.missdetections += 1 
        
        # First iteration
        else:
            self.poly_best_fit = current_fit
            self.recent_poly_fits = np.array([current_fit])

            self.poly_plotx = left_fitx
            self.poly_ploty = ploty
        
        self.measure_real_curvature()
    
    def sanityCheck(self,limit):
        '''
        Resets the line if it has multiple missdetections.
        '''
        if self.missdetections > limit:
            self.recent_poly_fits = self.recent_poly_fits[:-(limit-1),:]
            self.recent_xfitted = self.recent_xfitted[-(limit-1)]
            self.missdetections = 0
            print("Reset by SanityCheck")

    
    def show(self):
        '''
        Prints it current properties
        '''
        pprint(vars(self))
        
        