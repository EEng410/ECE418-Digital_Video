import numpy as np
import matplotlib.pyplot as plt
import scipy

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
            self.resolution_levels.append( (int(self.XRES/_tmp), int(self.YRES/_tmp)) )
                            
        self.wx = []
        self.wy = []
        self.grid = []
        self.r = []
        self.theta = []

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
        self.h0_filter = np.array([])
        self.l0_filter = np.array([])

        self.h_filters = []
        self.l_filters = []

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
        grid[interp_inds] = np.cos(np.pi/2. * np.log2(2. * r_vals[interp_inds]/np.pi))

        # Design a filter 


        

    def transform(input):
        

