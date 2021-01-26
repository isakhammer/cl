import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

class BSpline:

    """
    Wrapper of scipy spline libary
    Input:
        x: (Nx1) array
        t_length: Chosen max progression along spline
    """
    def __init__(self, x, t_length):
        t = np.linspace(0, t_length, num=x.size, endpoint=True)
        that, c, k = interpolate.splrep(t, x, s=0, k=4)
        self.spline = interpolate.BSpline(that, c, k, extrapolate=True)
        self.tmin = t.min()
        self.tmax= t.max()

    """
    Return value of spline at progression t
    """
    def value(self, t):
        if isinstance(t, (list, tuple, np.ndarray)):
            return self.value_array(t)

        return self.spline(t)

    def value_array(self, t):
        spline_values = np.zeros(t.shape)
        for i in range(t.size):
            spline_values[i] = self.value(t[i])
        return spline_values

class BSpline2D:
    """
    2D wrapper of Bspline.

    Input:
        X: (Nx2) array
        t_length: Chosen max progression along spline
    """
    def __init__(self, X, t_length):
        self.x_spline = BSpline(X[:,0], t_length )
        self.y_spline = BSpline(X[:,1], t_length )
        return

    """
    2D wrapper of Bspline.

    Input:
        t: progression along spline
    Output:
        (1x2) matrix values at t
    """
    def value(self, t):
        x = self.x_spline.value(t)
        y = self.y_spline.value(t)
        p = np.array([x,y]).T
        return p


class Centerline:
    """
    Computing and constructing splines of all features along the
    progression of the given input.

    Input:
        reftrack    (Nx4) array of x, y, w_r, w_l

    """

    def __init__(self, reftrack):

        # Parameterize everything on t
        s = self.compute_length(reftrack[:,:2], exact=True)
        self.s_length       = s[-1]
        self.s_spline       = BSpline(s, self.s_length)
        self.p_spline       = BSpline2D(reftrack[:,:2], self.s_length)
        self.w_spline       = BSpline2D(reftrack[:, 2:], self.s_length)

        def compute_kappa(p):
            def det(v,q):
                # z axis of cross product
                return v[0]*q[1] - q[0]*v[1]
            def curvature(p1,p2,p3):
                # menger curvature
                return 2*det(p2 - p1, p3 - p1)/(np.linalg.norm(p2-p3)*np.linalg.norm(p2-p3)*np.linalg.norm(p1-p3))
            # Evalulating kappa, inclusive start and end point
            kappa = np.zeros(p.shape[0])
            kappa[0] = curvature(p[-1], p[0], p[1])
            kappa[-1] = curvature(p[-2], p[-1], p[0])

            for i in range(1, p.shape[0] -1):
                kappa[i] = curvature(p[i-1], p[i], p[i+1])

            return kappa

        # discretize centerline
        n_points = 1000
        t = np.linspace(0, self.s_length, n_points, endpoint=False)
        p = self.p_spline.value(t)

        # Make spline of curvature
        kappa = compute_kappa(p)
        self.kappa_spline = BSpline( kappa, self.s_length)

        # Make spline of normal vector spline
        R = np.array([[0, -1],
                      [1, 0]])
        T = p[1:] - p[:-1]
        N = T@R.T
        N /= np.linalg.norm(N, ord=1, axis=1, keepdims=True)
        self.N_spline       = BSpline2D(N, self.s_length)

        # cumulativive summation of heading angle with no normalization.
        psi = np.arctan2(T[:,1], T[:,0])
        dpsi = psi[1:] - psi[:-1]
        for i in range( dpsi.shape[0] ):
            while dpsi[i] > np.pi:
                dpsi[i] -= 2*np.pi
            while dpsi[i] < -np.pi:
                dpsi[i] += 2*np.pi
        psi = np.cumsum(dpsi)

        self.psi_spline       = BSpline(psi, self.s_length)

    def compute_length(self, wpts, exact=False, N_exact=1000):

        if exact:
            # inexact differ approx 0.16% from exact length
            x_fine_spline = BSpline(wpts[:,0], t_length=1)
            y_fine_spline = BSpline(wpts[:,1], t_length=1)
            t = np.linspace(0,1, N_exact)
            x = x_fine_spline.value(t)
            y = y_fine_spline.value(t)
            wpts = np.array([x,y]).T

        dp = wpts[1:] - wpts[:-1]
        l = np.sqrt(dp[:,0]**2 + dp[:,1]**2)
        s = np.cumsum(l)
        s = np.append([0],s)
        return s

    def proj(self, p, psi, t0 = None, method="bruteforce"):
        # Quite shitty method
        tt = None
        N, dt = 1000, 10
        if method=="bruteforce":
            if t0 is not None:
                tt = np.linspace(t0 - dt, t0 + dt, N)
            else:
                tt = np.linspace(0, self.t_length, N)

            e = np.inf
            tproj = 0
            for z in tt:
                pbar = self.p_spline.value(z)
                d = np.linalg.norm(p - pbar)
                if d < e:
                    e = d
                    tproj = z
        return tproj


    """
    Discretizing the splines along a interval.

    Input:
        t0: Start distance.
        t1: End distance
        N: Number of grids
    Output:
        s:          (Nx1) array of distance progression
        reftrack    (Nx4) array of x, y, w_r, w_l
        kappa:      (Nx1) Curvature along distance progression
        normvec:    (Nx2) Normal vector at each point
        psi:        (Nx1) Heading angle along distance progression

    """
    def discretize(self, t0, t1, N):
        t = np.linspace(t0, t1, N)
        reftrack        = np.zeros((N,4))
        reftrack[:, :2] = self.p_spline.value(t)
        reftrack[:, 2:] = self.w_spline.value(t)
        kappa           = self.kappa_spline.value(t)
        normvec         = self.N_spline.value(t)
        s               = self.s_spline.value(t)
        psi             = self.psi_spline.value(t)
        return s, reftrack, kappa, normvec, psi


    def end(self):
        return self.s_length


