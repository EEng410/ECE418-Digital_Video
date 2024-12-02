import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

class SteerablePyramid():
    def __init__(self, k, level, input_dim):
        # k specifies a number of bands to extract per level 
        self.k = k
        # level specifies the level of the transform
        self.level = level

        # input dim is height x width, a tuple. 
        self.input_dim = input_dim

        # design for the steerable pyramid. 
        self.filters = []
        
        self.resolution_levels = []
        
        for i in range(0, self.level):
            _tmp = 2.** i
            self.resolution_levels.append( (int(self.input_dim[0]/_tmp), int(self.input_dim[1]/_tmp)) )
                            
        self.wx = []
        self.wy = []
        self.grid = []
        self.r = []
        self.theta = []

        # normalization factor for bandpass filters
        self.alpha_k = 2.**(self.k-1) * math.factorial(self.k-1)/np.sqrt(self.k * float(math.factorial(2*(self.k-1))))

        # set up a list of frequency domain grids [-pi, pi] x [-pi x pi] depending on resolution level
        for i in range(0, self.level):
            wx_tmp = np.linspace(-np.pi, np.pi, self.resolution_levels[i][0])
            wy_tmp = np.linspace(-np.pi, np.pi, self.resolution_levels[i][1])
            self.wx.append(wx_tmp)
            self.wy.append(wy_tmp)
            self.grid.append(np.zeros((self.resolution_levels[i][0], self.resolution_levels[i][1])))
            r_tmp, theta_tmp = self.cartesian_to_polar(wx_tmp, wy_tmp)
            self.r.append(r_tmp)
            self.theta.append(theta_tmp)
        
        # create a filter bank
        self.h0_filter = self.gen_h0_filter()
        self.l0_filter = self.gen_l0_filter()

        self.h_filters = self.gen_h_filter()
        self.l_filters = self.gen_l_filter()

        self.b_filters = self.gen_b_filters()

        

    def cartesian_to_polar(self, wx, wy):
        # meshgrid on wx, wy
        x, y = np.meshgrid(wx, wy)

        # calculates angle and radius on a grid
        angle = np.arctan2(y, x)
        radius = np.sqrt(x**2 + y**2)

        return radius, angle
    
    def gen_h0_filter(self):
        grid = self.grid[0].copy()
        r_vals = self.r[0]
        # grab the outer regions beyond the radial scope of the filter and set them to 1 as expected
        grid[np.where(r_vals >= np.pi)] = 1

        # set the inner region of the filter to 0
        grid[np.where(r_vals <= np.pi/2)] = 0

        # interpolate
        interp_inds = np.where((r_vals < np.pi) & (r_vals > np.pi/2))
        grid[interp_inds] = np.cos(np.pi/2. * np.log2(r_vals[interp_inds]/np.pi))

        return grid        

    def gen_l0_filter(self):
        grid = self.grid[0].copy()
        r_vals = self.r[0]
        # grab the outer regions beyond the radial scope of the filter and set them to 1 as expected
        grid[np.where(r_vals >= np.pi)] = 0

        # set the inner region of the filter to 0
        grid[np.where(r_vals <= np.pi/2)] = 1

        # interpolate
        interp_inds = np.where((r_vals < np.pi) & (r_vals > np.pi/2))
        grid[interp_inds] = np.cos(np.pi/2. * np.log2(2. * r_vals[interp_inds]/np.pi))

        return grid      

    def gen_h_filter(self):
        filters = []
        for i in range(0, self.level):
            grid = self.grid[i].copy()
            r_vals = self.r[i]
            # grab the outer regions beyond the radial scope of the filter and set them to 1 as expected
            grid[np.where(r_vals >= np.pi/4)] = 1

            # set the inner region of the filter to 0
            grid[np.where(r_vals <= np.pi/2)] = 0

            # interpolate
            interp_inds = np.where((r_vals < np.pi/2) & (r_vals > np.pi/4))
            grid[interp_inds] = np.cos(np.pi/2. * np.log2(2. * r_vals[interp_inds]/(np.pi)))

            filters.append(grid)

        return filters

    def gen_l_filter(self):
        filters = []
        for i in range(0, self.level):
            grid = self.grid[i].copy()
            r_vals = self.r[i]
            # grab the outer regions beyond the radial scope of the filter and set them to 1 as expected
            grid[np.where(r_vals >= np.pi/4)] = 0

            # set the inner region of the filter to 1
            grid[np.where(r_vals <= np.pi/2)] = 1

            # interpolate
            interp_inds = np.where((r_vals < np.pi/2) & (r_vals > np.pi/4))
            grid[interp_inds] = np.cos(np.pi/2. * np.log2(4. * r_vals[interp_inds]/(np.pi)))

            filters.append(grid)

        return filters

    def gen_b_filters(self):
        filters = []
        for i in range(0, self.level):
            bp_filters = []
            for k in range(0, self.k):
                fil_= np.zeros_like(self.grid[i])
                th1= self.theta[i].copy()
                th2= self.theta[i].copy()

                th1[np.where(self.theta[i] - k*np.pi/self.k < -np.pi)] += 2.*np.pi
                th1[np.where(self.theta[i] - k*np.pi/self.k > np.pi)] -= 2.*np.pi
                ind_ = np.where(np.absolute(th1 - k*np.pi/self.k) <= np.pi/2.)
                fil_[ind_] = self.alpha_k * (np.cos(th1[ind_] - k*np.pi/self.k))**(self.k-1)
                th2[np.where(self.theta[i] + (self.k-k)*np.pi/self.k < -np.pi)] += 2.*np.pi
                th2[np.where(self.theta[i] + (self.k-k)*np.pi/self.k > np.pi)] -= 2.*np.pi
                ind_ = np.where(np.absolute(th2 + (self.k-k) * np.pi/self.k) <= np.pi/2.)
                fil_[ind_] = self.alpha_k * (np.cos(th2[ind_]+ (self.k-k) * np.pi/self.k))**(self.k-1)

                fil_= self.h_filters[i] * fil_
                bp_filters.append(fil_.copy())
            filters.append(bp_filters)

        return filters

    def create_pyramids(self, image):
        # If I cared enough I would implement some checks for the image dimensions but whatever

        out = []
        lvl0 = []
        # DFT on the image
        im_ft = np.fft.fft2(image)
        _im_ft = np.fft.fftshift(im_ft)

        # apply h0 filter (all filters are defined in the frequency domain)
        h0 = _im_ft * self.h0_filter 
        h0_img = np.fft.ifft2(np.fft.ifftshift(h0))

        lvl0.append({'f': _im_ft, 's': h0_img})
        
        # apply l0 filter 
        l0 = _im_ft * self.l0_filter
        l0_img = np.fft.ifft2(np.fft.ifftshift(l0))

        lvl0.append({'f': _im_ft, 's': l0_img})

        out.append(lvl0)

        # apply bandpass filters and iteratively downsample
        _last = l0
        for n in range(0, self.level):
            # Store outputs per level
            _t = []
            for k in range(len(self.b_filters[n])):
                # temporary dictionary for storing frequency and spatial representations of the image
                _tmp = {'f':None, 's':None}
                lb = _last * self.b_filters[n][k]
                img_back = np.fft.ifft2(np.fft.ifftshift(lb))
                # frequency
                _tmp['f'] = lb
                # space
                _tmp['s'] = img_back
                _t.append(_tmp)
            
            ln = _last * self.l_filters[n]

            # Downsample original low-passed frqeuency domain image by a factor of 2
            down_img = _last[0::2, 0::2]    
            
            # ifft the image
            img_back = np.fft.ifft2(np.fft.ifftshift(down_img))
            _t.append({'f': down_img, 's': img_back})
            out.append(_t)
            _last = down_img

        return out

