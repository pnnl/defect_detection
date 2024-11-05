import numpy as np
import math
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances



#Code  initialed with from https://docs.astropy.org/en/stable/stats/ripley.html
#edge case code from code adopted from https://github.com/spatstat/spatstat.core/blob/master/R/edgeRipley.R
class RipleysKEstimator:
    """
    Estimators for Ripley's K function for two-dimensional spatial data.
    See [1]_, [2]_, [3]_, [4]_, [5]_ for detailed mathematical and
    practical aspects of those estimators.

    Parameters
    ----------
    area : float
        Area of study from which the points where observed.
    x_max, y_max : float, float, optional
        Maximum rectangular coordinates of the area of study.
        Required if ``mode == 'translation'`` or ``mode == ohser``.
    x_min, y_min : float, float, optional
        Minimum rectangular coordinates of the area of study.
        Required if ``mode == 'variable-width'`` or ``mode == ohser``.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt # doctest: +SKIP
    >>> from astropy.stats import RipleysKEstimator
    >>> z = np.random.uniform(low=5, high=10, size=(100, 2))
    >>> Kest = RipleysKEstimator(area=25, x_max=10, y_max=10,
    ... x_min=5, y_min=5)
    >>> r = np.linspace(0, 2.5, 100)
    >>> plt.plot(r, Kest.poisson(r)) # doctest: +SKIP
    >>> plt.plot(r, Kest(data=z, radii=r, mode='none')) # doctest: +SKIP
    >>> plt.plot(r, Kest(data=z, radii=r, mode='translation')) # doctest: +SKIP
    >>> plt.plot(r, Kest(data=z, radii=r, mode='ohser')) # doctest: +SKIP
    >>> plt.plot(r, Kest(data=z, radii=r, mode='var-width')) # doctest: +SKIP
    >>> plt.plot(r, Kest(data=z, radii=r, mode='ripley')) # doctest: +SKIP

    References
    ----------
    .. [1] Peebles, P.J.E. *The large scale structure of the universe*.
       <https://ui.adsabs.harvard.edu/abs/1980lssu.book.....P>
    .. [2] Spatial descriptive statistics.
       <https://en.wikipedia.org/wiki/Spatial_descriptive_statistics>
    .. [3] Package spatstat.
       <https://cran.r-project.org/web/packages/spatstat/spatstat.pdf>
    .. [4] Cressie, N.A.C. (1991). Statistics for Spatial Data,
       Wiley, New York.
    .. [5] Stoyan, D., Stoyan, H. (1992). Fractals, Random Shapes and
       Point Fields, Akademie Verlag GmbH, Chichester.
    """

    def __init__(self, area, x_max=None, y_max=None, x_min=None, y_min=None):
        self.area = area
        self.x_max = x_max
        self.y_max = y_max
        self.x_min = x_min
        self.y_min = y_min

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        if isinstance(value, (float, int)) and value > 0:
            self._area = value
        else:
            raise ValueError(f'area is expected to be a positive number. Got {value}.')

    @property
    def y_max(self):
        return self._y_max

    @y_max.setter
    def y_max(self, value):
        if value is None or isinstance(value, (float, int)):
            self._y_max = value
        else:
            raise ValueError('y_max is expected to be a real number '
                             'or None. Got {}.'.format(value))

    @property
    def x_max(self):
        return self._x_max

    @x_max.setter
    def x_max(self, value):
        if value is None or isinstance(value, (float, int)):
            self._x_max = value
        else:
            raise ValueError('x_max is expected to be a real number '
                             'or None. Got {}.'.format(value))

    @property
    def y_min(self):
        return self._y_min

    @y_min.setter
    def y_min(self, value):
        if value is None or isinstance(value, (float, int)):
            self._y_min = value
        else:
            raise ValueError(f'y_min is expected to be a real number. Got {value}.')

    @property
    def x_min(self):
        return self._x_min

    @x_min.setter
    def x_min(self, value):
        if value is None or isinstance(value, (float, int)):
            self._x_min = value
        else:
            raise ValueError(f'x_min is expected to be a real number. Got {value}.')

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    #for each line, get the difference of that line, and the others
    def _pairwise_diffs(self, data1, data2):
        npts_data1 = len(data1)
        npts_data2 = len(data2)
        diff = np.zeros(shape=(npts_data1 * npts_data2, 2), dtype=np.double)
        k = 0
        for i in range(npts_data1):
            size = npts_data2
            diff[k:k + size] = abs(data1[i] - data2)
            k += size

        return diff

    def poisson(self, radii):
        """
        Evaluates the Ripley K function for the homogeneous Poisson process,
        also known as Complete State of Randomness (CSR).

        Parameters
        ----------
        radii : 1D array
            Set of distances in which Ripley's K function will be evaluated.

        Returns
        -------
        output : 1D array
            Ripley's K function evaluated at ``radii``.
        """

        return np.pi * radii * radii


    def Lfunction(self, data1, data2, radii, mode='none'):
        """
        Evaluates the L function at ``radii``. For parameter description
        see ``evaluate`` method.
        """
        return np.sqrt(self.evaluate(data1, data2, radii, mode=mode) / np.pi)


    def Hfunction(self, data1, data2, radii, mode='none'):
        """
        Evaluates the H function at ``radii``. For parameter description
        see ``evaluate`` method.
        """
        return self.Lfunction(data1, data2, radii, mode=mode) - radii

    def evaluate(self, data1, data2, radii, mode='ripley'):
        """
        Evaluates the Ripley K estimator for a given set of values ``radii``.

        Parameters
        ----------
        data1 : 2D array
            Set of observed points in as a n by 2 array which will be used to
            estimate Ripley's K function.
        data2 : 2D array
            Set of observed points in as a n by 2 array which will be used to
            estimate Ripley's K function as clustering around data1
        radii : 1D array
            Set of distances in which Ripley's K estimator will be evaluated.
            Usually, it's common to consider max(radii) < (area/2)**0.5.
        mode : str
            Keyword which indicates the method for edge effects correction.
            Available methods are 'none', 'translation', 'ohser', 'var-width',
            and 'ripley'.

            * 'none'
                this method does not take into account any edge effects
                whatsoever.
            * 'ripley'
                this method is known as Ripley's edge-corrected estimator.
                The weight for edge-correction is a function of the
                proportions of circumferences centered at each data point
                which crosses another data point of interest. See [3] for
                a detailed description of this method.

        Returns
        -------
        ripley : 1D array
            Ripley's K function estimator evaluated at ``radii``.
        """

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        if not data1.shape[1] == 2:
            raise ValueError('data1 must be an n by 2 array, where n is the '
                             'number of observed points.')

        if not data2.shape[1] == 2:
            raise ValueError('data2 must be an n by 2 array, where n is the '
                             'number of observed points.')

        npts_data1 = len(data1)
        npts_data2 = len(data2)
        ripley = np.zeros(len(radii))


        # code adopted from https://github.com/spatstat/spatstat.core/blob/master/R/edgeRipley.R
        #
        #get min distance to edge in the same place in list as the distances (so by 4, 3, 2, 1)
        #start and end is equal to the  size = npts - i - 1
        if mode == 'ripley':
            dL_dist = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            dR_dist = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            dD_dist = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            dU_dist = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            x_coord = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            y_coord = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            corner_list = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)
            start = 0
            for k in range(npts_data1):
                #get min distance from data1 point to edge of graph
                dL = data1[k][0] - self.x_min
                dR = self.x_max - data1[k][0]
                dD = data1[k][1] - self.y_min
                dU = self.y_max - data1[k][1]
                # detect whether any points are corners of the rectangle
                corner = sum(int(abs(i) < np.finfo(np.float32).eps) for i in [dL, dR, dD, dU]) >= 2

                #keep track of it in a list with one entity per data1 connection to data2
                size = npts_data2
                dL_dist[start: start + size] = dL * np.ones(npts_data2)
                dR_dist[start: start + size] = dR * np.ones(npts_data2)
                dD_dist[start: start + size] = dD * np.ones(npts_data2)
                dU_dist[start: start + size] = dU * np.ones(npts_data2)
                x_coord[start: start + size] = data1[k][0] * np.ones(npts_data2)
                y_coord[start: start + size] = data1[k][1] * np.ones(npts_data2)
                corner_list[start: start + size] = corner * np.ones(npts_data2)
                start = start + size

            #pairwise collects distance between data1 and each point in data2
            diff = self._pairwise_diffs(data1, data2)
            #actual distance between data1 and data2 points
            dist = np.hypot(diff[:, 0], diff[:, 1])

            # angle between (a) perpendicular to edge of rectangle
            # and (b) line from point to corner of rectangle
            bLU = np.arctan2(dU_dist, dL_dist)
            bLD = np.arctan2(dD_dist, dL_dist)
            bRU = np.arctan2(dU_dist, dR_dist)
            bRD = np.arctan2(dD_dist, dR_dist)
            bUL = np.arctan2(dL_dist, dU_dist)
            bUR = np.arctan2(dR_dist, dU_dist)
            bDL = np.arctan2(dL_dist, dD_dist)
            bDR = np.arctan2(dR_dist, dD_dist)

            aL = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)

            aR = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)

            aD = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)

            aU = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)

            def hang(dL_dist, dist):
                if dL_dist < dist:
                    degree = np.arccos(dL_dist/dist)
                else:
                    degree = 0
                return degree

            # half the angle subtended by the intersection between
             # the circle of radius r[i,j] centred on point i
             # and each edge of the rectangle (prolonged to an infinite line)
            for i in range(npts_data1 * npts_data2):
                aL[i] = hang(dL_dist[i], dist[i])
                aR[i] = hang(dR_dist[i], dist[i])
                aD[i] = hang(dD_dist[i], dist[i])
                aU[i] = hang(dU_dist[i], dist[i])

            cL = np.minimum(aL, bLU) + np.minimum(aL, bLD)
            cR = np.minimum(aR, bRU) + np.minimum(aR, bRD)
            cU = np.minimum(aU, bUL) + np.minimum(aU, bUR)
            cD = np.minimum(aD, bDL) + np.minimum(aD, bDR)

            # total exterior angle
            ext = np.vstack((cL, cR, cU, cD)).sum(axis = 0)
            # add pi/2 for corners
            ext = np.where(corner_list, ext + np.pi/2, ext)
            weight = np.zeros(shape=npts_data1 * npts_data2,
                                dtype=np.double)

            for i in range(len(ext)):
                if dist[i] <= radii[-1]:
                    weight[i] = 1 / (1 - ext[i]/(2 * np.pi))
                else:
                    weight[i] = 1

            for r in range(len(radii)):
                #going to end up with accumalative 1/w as radius increases.
                #same result as accumlative whl  (binning) funciton in spatstate
                ripley[r] = ((dist < radii[r]) * weight).sum()

            #since there isn't always one less with multi, don't subtract 1 in (n*(n-1))
            ripley = self.area  * ripley / (npts_data1 * npts_data2)

        else:
            raise ValueError(f'mode {mode} is not implemented.')
        
        return ripley


def MCconfidence(r, mins, maxs, n1, n2, alpha=0.05, N=99, mode='ripley'):
    if len(mins) == len(maxs):
        D = len(mins)
    else:
        raise ValueError('Number of min and max items are not equal.')


    # Note: code is only for 2-D
    Kest = RipleysKEstimator(area=np.prod(maxs-mins),
                             x_max=maxs[0], y_max=maxs[1], x_min=mins[0], y_min=mins[1])
    HMC = np.zeros((len(r),N))
    for i in range(N):
        z = np.random.uniform(low=0, high=1, size=(n1, D))
        X = (z * (maxs - mins) + mins).astype(float)

        z = np.random.uniform(low=0, high=1, size=(n2, D))
        Y = (z * (maxs - mins) + mins).astype(float)

        HMC[:,i] = Kest.Hfunction(data1=X, data2 = Y, radii=r, mode=mode)

    LCI = np.quantile(HMC, alpha/2, axis=1)
    UCI = np.quantile(HMC, 1-alpha/2, axis=1)
    CR = np.vstack((LCI,UCI)).transpose()

    return CR
