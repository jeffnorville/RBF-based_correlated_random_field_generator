#############################################################################
#
# FieldGen.py - a spatially-correlated random 2-D field generator
#
# by Walt McNab
#
#############################################################################


from numpy import *
import pandas as pd
from scipy.spatial import distance 
import scipy.stats as stats
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

### classes ###


class FracSet:

    def __init__(self, fracFile, params):
        lineInput = []        
        inputFile = open(fracFile,'r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        self.orient = float(lineInput[0][1]) * pi/180.                  # direction (counterclockwise from east, radians)
        self.b = float(lineInput[1][1])                                 # average fracture spacing
        self.db = float(lineInput[2][1])                                # standard deviation in fracture spacing
        self.a = float(lineInput[3][1])                                 # fracture zone width
        self.Kmean = float(lineInput[4][1])                             # mean hydraulic conductivity
        self.dK = float(lineInput[5][1])                                # potential variation in K per unit distance
        self.Kmin = float(lineInput[6][1])                              # minimum hydraulic conductivity
        self.Kmax = float(lineInput[7][1])                              # maximum hydraulic conductivity 
        self.scale = sqrt((params.gridend[0]-params.grid0[0])**2 +
            (params.gridend[1]-params.grid0[1])**2)                     # length of domain diagonal
        self.psi = arctan2(params.gridend[1]-params.grid0[1],
            params.gridend[0]-params.grid0[0])                          # angle between diagonal and x-axis (positive)


    def FracSeedsLine(self, params):            # return array of points along appropriate domain diagonal to seed discrete fractures
        
        # distribute seed points
        if self.orient > 0.:
            alpha = self.psi - self.orient
        else:
            alpha = pi/2. - (abs(self.orient) + self.psi)
        hyp = self.b/cos(alpha)
        nSeeds = int(self.scale/hyp) + 1
        fracSeeds = linspace(0., self.scale, nSeeds)
        noise = random.normal(0., hyp*self.db/self.b, len(fracSeeds))
        fracSeeds = fracSeeds + noise                           # (slightly) randomize fracture positions
        
        # add in adjacent seeds (+x direction) to address fracture width
        span = hyp * self.a/self.b
        dspan = span/min(params.dcell)
        numSpan = int(round(dspan))
        fracSpan = []
        for i in range(numSpan):
            fracSpan.append(fracSeeds + (i+1)*dspan)
        for f in fracSpan:
            fracSeeds = concatenate((fracSeeds, array(f)))
        
        # fracture end points
        if self.orient > 0.0:
            xs = params.grid0[0] + fracSeeds*cos(self.psi)    
            ys = params.gridend[1] - fracSeeds*sin(self.psi)
        else:
            xs = params.grid0[0] + fracSeeds*cos(self.psi)    
            ys = params.grid0[1] + fracSeeds*sin(self.psi)
        return xs, ys


    def PropLine(self, xs, ys, params):
    
        # propagate line of posited property values spanning the domain and crossing seed point with proper orientation
        x0 = xs - cos(self.orient)*self.scale
        y0 = ys - sin(self.orient)*self.scale        
        xf = xs + cos(self.orient)*self.scale
        yf = ys + sin(self.orient)*self.scale
        
        # call params' intercept method to return list of grid cell indices crossed by fracture
        cellIndex = params.Intercept(x0, y0, xf, yf)

        # assign hydraulic conductivity along fracture by random walk
        path = BoundedWalk(self.Kmin, self.Kmax, len(cellIndex), self.dK)
        return cellIndex, path


    def SpawnFracs(self, params):
        
        # generate seed points along diagonal and propagate fractures in both directions
        xs, ys = self.FracSeedsLine(params)       # extract seed points along diagonal        

        # step through seed points and generate fractures; collect results as dataframe
        indx = []
        K = []
        for i in range(len(xs)):
            cellIndex, path = self.PropLine(xs[i], ys[i], params)
            indx.extend(cellIndex)
            K.extend(path)
        K = array(K)
        logK = log10(K)
        return indx, logK


class InvDist:
    
    def __init__(self, xp, yp, vp):
        self.pts = transpose([xp, yp])
        self.vp = vp
        
    def Interpolate(self, location, params):   
        d0 = 1e-6
        d = distance.cdist(location, self.pts)[0] + d0
        pointSet = pd.DataFrame(data={'distance':d, 'value':self.vp})
        nearSet = pointSet[pointSet['distance']<=params.searchInvDst]
        dLocal = sqrt(array(nearSet['distance']**2) + params.smooth**2)
        vLocal = array(nearSet['value'])
        h = sum(vLocal/dLocal**2) / sum(1./dLocal**2)
        return h


class Params:
    
    def __init__(self, paramsFile):
        # miscellaneous setup parameters
        lineInput = []        
        inputFile = open(paramsFile,'r')
        for line in inputFile: lineInput.append(line.split())
        inputFile.close()
        self.grid0 = array([float(lineInput[1][1]), float(lineInput[1][2])])
        self.gridend = array([float(lineInput[2][1]), float(lineInput[2][2])])        
        self.n = array([int(lineInput[3][1]), int(lineInput[3][2])])
        self.a = array([float(lineInput[4][1]), float(lineInput[4][2])])
        self.depth = float(lineInput[5][1])
        self.mu = float(lineInput[6][1])
        self.stdev0 = float(lineInput[7][1])
        self.lower = float(lineInput[8][1])
        self.upper = float(lineInput[9][1])
        self.rsearch0 = float(lineInput[10][1])
        self.expn = float(lineInput[11][1])        
        self.dmin = float(lineInput[12][1])
        self.f = lineInput[13][1]
        self.searchInvDst = float(lineInput[14][1])
        self.smooth = float(lineInput[15][1])
        self.rescale = bool(lineInput[16][1])
        self.rescaleMu = float(lineInput[17][1])
        self.rescaleSigma = float(lineInput[18][1])   
        lengthScale = array([self.gridend[0]-self.grid0[0], self.gridend[1]-self.grid0[1]])
        self.dcell = lengthScale/self.n
        print('Read random field setup parameters.')

    def CheckBounds(self, x, y):
        # check if point (x, y) is within domain
        check = False
        if ((x >= self.grid0[0]) & (x <= self.gridend[0]) & (y >= self.grid0[1]) & (y <= self.gridend[1])):
            check = True
        return check

    def GetIndex(self, x, y):
        # index number of grid cell corresponding to (x, y, z)
        col = ((x-self.grid0[0])/self.dcell[0]).astype(int)
        row = ((y-self.grid0[1])/self.dcell[1]).astype(int)
        indx = col + row*self.n[0]
        return indx

    def Intercept(self, x0, y0, xf, yf):   # find grid cell set correpsonding to line segment
        
        # find which dimension will require more grid cells (=nCells)
        nx = int(2.*(xf-x0)/self.dcell[0]) + 1
        ny = int(2.*(yf-y0)/self.dcell[1]) + 1
        numPts = max(nx, ny)
        
        # delineate line segment into vertices
        x = linspace(x0, xf, numPts)
        y = linspace(y0, yf, numPts)        
        
        # find corresponding cell indices for points inside model domain
        insidePt = zeros(len(x), bool)
        cellIndex = []
        for i in range(len(x)): insidePt[i] = self.CheckBounds(x[i], y[i])
        x = x[insidePt==True]
        y = y[insidePt==True]
        
        # reduce to set and sort by index
        if len(x) > 0: 
            cellIndex = self.GetIndex(x, y)
            cellIndex = list(set(cellIndex))
            cellIndex.sort()
        else: cellIndex = []
        return cellIndex


class Grid:
    
    def __init__(self, params, seedsFile):

        # seed points
        points = pd.read_csv(seedsFile)
        x = array(points['x'])
        y = array(points['y'])
        v = array(points['v'])        
        
        # grid setup
        self.xgrid = linspace(params.grid0[0]+0.5*params.dcell[0], params.gridend[0]-0.5*params.dcell[0], params.n[0])
        self.ygrid = linspace(params.grid0[1]+0.5*params.dcell[1], params.gridend[1]-0.5*params.dcell[1], params.n[1])
        X, Y = meshgrid(self.xgrid, self.ygrid)
        xg = X.flatten()
        yg = Y.flatten()
        indx = arange(0, len(xg), 1)
        marked = zeros(len(xg), bool) * False
        val = zeros(len(xg), float) - 9999.
        mat = ['matrix'] * len(xg)
        self.cells = pd.DataFrame(data={'index':indx, 'x':xg, 'y':yg, 'v':val, 'material':mat, 'marked':marked})
        print('Read seed points and set up random field grid data frame.')

        # populate grid cells that contain the initial seed points
        print('Marking grid cells containing seed points ...')
        indx = params.GetIndex(x, y)
        self.cells['marked'].iloc[indx] = True
        self.cells['v'].iloc[indx] = v



    def Posit(self, params, d):
        # return a posited new data point (represented by grid cell)
        subset = self.cells[self.cells['marked']==False]
        indices = list(subset.index.values)
        r = random.choice(indices)          # select a random grid cell to assign value
        rcell = subset.loc[r]
        xp = rcell['x'] / params.a[0]
        yp = rcell['y'] / params.a[1]
        marked = self.cells[self.cells['marked']==True]
        x = array(marked['x']) / params.a[0]
        y = array(marked['y']) / params.a[1] 
        v0 = array(marked['v'])
        x, y, v = DistFilter(x, y, v0, xp, yp, params, d)  # find nearby points
        if len(x) > 1:
            rbfi = Rbf(x, y, v, function=params.f)   # radial basis function interpolator
            mu = rbfi(xp, yp)
        else:
            mu = params.mu
        sigma = params.stdev0 * d
        X = stats.truncnorm((params.lower - mu) / sigma, (params.upper - mu) / sigma, loc=mu, scale=sigma)
        v = X.rvs(1)[0]
        self.cells['marked'].iloc[r] = True
        self.cells['v'].iloc[r] = v     
        
    def Fill(self, params):
        # fill in un-marked grid cells by interpolation
        print('Filling remaining random field cells by interpolation ...')
        marked = self.cells[self.cells['marked']==True].copy()
        marked.to_csv('points.csv', index=False)    # write points to file
        v = array(marked['v'])
        x = array(marked['x']) / params.a[0]
        y = array(marked['y']) / params.a[1]        
        fN = InvDist(x, y, v)
        xp = array(self.cells['x']) / params.a[0]
        yp = array(self.cells['y']) / params.a[1]
        vp = []
        for i in range(len(xp)): vp.append(fN.Interpolate( [[xp[i], yp[i]]], params))
        self.cells['v'] = vp

    def WriteOutput(self):
        # output to file
        self.cells = self.cells[['x', 'y', 'v', 'material', 'matrix_v']]
        self.cells.to_csv('filled_cells.csv', index=False)

    def PlotField(self, params):
        # display field as a color mesh
        z = array(self.cells['v']).reshape(params.n[0], params.n[1])
        plt.pcolormesh(self.xgrid, self.ygrid, z, cmap=cm.RdBu)      
        plt.colorbar()
        plt.title('Log K')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()



### utility functions ###


def DistFilter(x, y, v, xp, yp, params, d):
    # filter data set by distance
    rsearch = params.rsearch0 * d     # reduce size of subset to reflect updated (effective) search radius
    tpts = transpose([x, y])
    dist = distance.cdist([[xp, yp]], tpts)
    x = x[dist[0]<=rsearch]
    y = y[dist[0]<=rsearch]
    v = v[dist[0]<=rsearch]
    return x, y, v


def BoundedWalk(lowerBound, upperBound, numSteps, stepSize):
    # random walk between two boundaries
    wander = []
    point = random.uniform(lowerBound, upperBound)
    wander.append(point)
    for i in range(numSteps):
        shift = random.uniform(-stepSize, stepSize)
        if (point + shift) < lowerBound:
            point = 2*lowerBound - (point+shift)
        elif (point + shift) > upperBound:
            point = 2*upperBound - (point+shift)
        else:
            point += shift
        wander.append(point)
    return wander


### main script ###


def FieldGen(rMode, fMode, paramsFile, seedsFile, fracFiles):

    # read model parameters
    params = Params(paramsFile)
    
    # read seed points and construct grid
    grid = Grid(params, seedsFile)    
 
    if rMode:       # create correlated random field for log K
 
        # step through cells
        print('Positing pilot points ...')
        nfilled = int(params.depth*params.n[0]*params.n[1])
        for i in range(nfilled):
            d = 1.0 - (1.0-params.dmin)*(float(i)/float(nfilled))**params.expn
            grid.Posit(params, d)
        
        # fill in remaining cells with straight interpolations of existing marked cell set
        grid.Fill(params)   # primary variable
        
        # re-scale by stretching histogram (assumes a normal distribution)
        if params.rescale:
            print('Re-scaling random field ...')
            u0 = grid.cells['v'].mean()
            stdev0 = grid.cells['v'].std()
            vcdf = stats.norm.cdf(grid.cells['v'], loc=u0, scale=stdev0)
            vScaled = stats.norm.ppf(vcdf, loc=params.rescaleMu, scale=params.rescaleSigma)
            shift = vScaled - grid.cells['v']
            grid.cells['v'] = vScaled

    else: grid.cells['v'] = params.mu           # assign a single value for log K

    # copy of matrix-derived log K for use in mineralogy assignments (prior to fracture)
    grid.cells['matrix_v'] = grid.cells['v']

    # add fracture zones
    if fMode:
        for frac in fracFiles:
            fracSet = FracSet(frac, params)
            indx, logK = fracSet.SpawnFracs(params)
            for i, idx in enumerate(indx):
                grid.cells['v'].iloc[idx] = max(grid.cells['v'].iloc[idx], logK[i])
                grid.cells['material'].iloc[idx] = 'fracture'
    
    # write completed point set and plot random field
    grid.WriteOutput()
    grid.PlotField(params)
    
    print('Done with random field.')
    
    return grid.cells


### development run ###
rMode = True            # create continuous correlated random field
fMode = False           # generate fracture networks
paramsFile = 'field_params.txt'
seedsFile = 'seeds.csv'
fracFiles = ['fracSet1.txt']
#fracFiles = ['fracSet1.txt', 'fracSet2.txt']
field = FieldGen(rMode, fMode, paramsFile, seedsFile, fracFiles)