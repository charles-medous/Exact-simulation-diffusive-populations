"""Define the functions that generates a trajectory with the Exact Algorithm.

Usage:
======
    Use 'Trajectory' function to plot a realization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats


def Phi(z, a):
    """Compute the intensity function phi for a given value of the proposed 
    Brownian motion conditioned on its ending point.

    Parameters
    ----------
    numpy array
        A value of the N_0-dimmensional Brownian bridge 
    float
        The growth rate 
        
    Returns
    -------
    float
        The intensity function evaluated in z 
    """
    n = np.size(z)
    norm = np.sum(z ** 2)
    num = n + (a / 2 + n - 2) * norm
    denom = 2 * (1 + norm) ** 2
    condition = n > 1 or a > 2
    if condition:
        output = a * num / denom
    else:
        output = a * num / denom + a * (a - 2) ** 2 / (64 - 16 * a)
    return(output / 2)

def M(n, a):
    """Compute the upper bound for the intensity function phi for a given 
    growth rate.

    Parameters
    ----------
    int
        The population size
    float
        The growth rate 
        
    Returns
    -------
    float
        The maximum of the function phi. 
    """
    if n >= a / 2 - 2:
        if n < 2 - a / 2:
            m = a * n / 2 + a * (a - 2) ** 2 / (64 - 16 * a)
        else:
            m = a * n / 2
    else:
        m = (a * (n + a / 2 - 2) ** 2) / (4 * (a - 4))
    return(m / 2)

def PPP(m, t_0, t_1):
    """Generate a unit-rate Poisson point process on [0,m]*[t_0,t_1].

    Parameters
    ----------
    float
        The upper bound of the intensity function phi
    float
        The initial time of the current window
    float
        The ending time of the current window
        
    Returns
    -------
    numpy array
        The time instances at which a point has been drawn
    numpy array
        The value of the Poisson process at the time instances.
    """
    lambda0 = 1 # Change to higher values for more points- longer simul. time 
    npts = np.random.poisson(lambda0 * m * (t_1 - t_0))
    if npts > 0:
        times = (t_1 - t_0) * np.random.uniform(0,1,npts) + t_0
        phi = m * np.random.uniform(0,1,npts) 
    else:
        times = []
        phi = []
    return(np.sort(times),phi)


def Fx(z, x, t_0, t_1, a):
    """Unormalized density distribution of the ending point.
    """
    n = np.size(x)
    temp_1 = (1 + np.sum(z ** 2)) ** (a / 4) 
    temp_3 = np.sum((z - x) ** 2) / (2 * (t_1 - t_0))
    return(temp_1 * np.exp(-temp_3) * (4 * np.pi * (t_1 - t_0)) ** (- n / 2) )

def GenerateEndPoint(x, t_0, t_1, a):
    """Generate a realization of the ending point according to f_x.

    Parameters
    ----------
    numpy array
        The initial position of the Brownian bridge
    float
        The initial time of the current window
    float
        The ending time of the current window
    float
        The growth rate
        
    Returns
    -------
    numpy array
        The N_0-dimensional ending point. 
    """
    thresh = 0
    z = 0
    c = 1.5 * (1 + np.sum(x ** 2)) ** (a / 4)
    count = 0
    while thresh > Fx(z, x, t_0, t_1, a) or count == 0:
        u = np.random.uniform()
        z = np.random.normal(x, np.sqrt(2 * (t_1 - t_0)))
        pi = np.prod(stats.norm.pdf(z, x, np.sqrt(2 * (t_1 - t_0))))
        thresh = u * c * pi
        count += 1
    return(z)
    

def CountPts(w, psi, a):
    """Count the number of point such that phi(w) > phi.

    Parameters
    ----------
    numpy array
        The state ofthe Brownian bridge at times instances
    numpy array
        The trait realizations of the Poisson point process at times instances
    float
        The growth rate
        
    Returns
    -------
    int
        The number of points of the PPP under the curve. 
    """
    n = np.size(psi)
    count = 0
    for i in range(n):
        if Phi(w[i],a) >= psi[i]:
            count += 1
    return(count)

def GenerateW(x, z, t_0, t_1, times):
    """Generate a proposal skeleton of the N_0-dimensional Brownian motion 
    starting at x in t_0 and conditioned on hitting z at time t_1, at the time
    instances times.

    Parameters
    ----------
    numpy array
        The initial position of the Brownian bridge
    numpy array
        The ending position of the Brownian bridge
    float
        The initial time of the current window
    float
        The ending time of the current window
    numpy array
        The time instances
        
    Returns
    -------
    list of numpy array
        The N_0-dimensional Brownian bridge skeleton. 
    """
    D = t_1 - t_0
    n = np.size(x)
    n_t = np.size(times)
    w = [np.ones(n) for i in range(n_t)]
    for i in range(n_t):
        w[i] = np.random.normal(x + (times[i] - t_0) * (z - x) / D, 
                            np.sqrt((t_1 - times[i]) * (times[i] - t_0) / D))
    return(w)

def RetrospectiveSampling(x, t_0, t_1, a):
    """Generate until time t_1, a skeleton of the N_0-dimensional trait process
    starting in x at time t_0.

    Parameters
    ----------
    numpy array
        The initial position of the traits
    float
        The initial time of the current window
    float
        The ending time of the current window
    float
        The growth rate
        
    Returns
    -------
    numpy array
        The time instances at which a point has been simulated
    list of numpy array
        The skeleton of the N_0-dimensional trait process
    numpy array
        The N_0-dimensional ending point. 
    """
    n_sample = 1 
    n = np.size(x)
    m = M(n, a)
    while n_sample >= 1:
        times, psi = PPP(m, t_0, t_1)
        z = GenerateEndPoint(x, t_0, t_1, a)
        w = GenerateW(x, z, t_0, t_1, times)
        n_sample = CountPts(w, psi, a)
    return(times, w, z)
    
def Trajectory(a, r, x_0, n_init, T):
    """Plot until time T, a realizaion of the trait process starting from 
    n_init individual(s) until time T, for a given set of parameters.

    Parameters
    ----------
    float
        The growth rate
    float
        The branching rate
    float
        The mean trait value at birth
    int
        The initial population size
    float
        The simulation time
        
    Returns
    -------
    none
        Plot the trajectories of the individuals traits in the population.
    """    
    npts = np.random.poisson(r * T)
    tableau = np.zeros(npts + n_init -1)
    clr = cm.get_cmap('rainbow',len(tableau))
    clr = clr(np.linspace(0,1,len(tableau))) 
    X = x_0 * np.random.uniform(0,1,n_init)
    t = np.zeros(npts + 2)
    t[-1] = T
    if npts > 0:
        t[1:-1] = np.sort(T * np.random.uniform(0,1,npts) )
    plt.figure(figsize=(7.24,4))
    for i in range(npts - 1):
        times, w, z = RetrospectiveSampling(X, t[i], t[i+1], a)
        traj_temp = np.ones((np.size(X), np.size(times) + 2))
        tps = np.ones(np.size(times) + 2)
        tps[0] = t[i]
        tps[-1] = t[i+1]- 0.0001
        for j in range(np.size(X)):
            traj_temp[j][0] = X[j]
            traj_temp[j][-1] = z[j]
            X[j] = z[j]
            for k in range(np.size(times)):
                traj_temp[j][k+1] = w[k][j]
                tps[k+1] = times[k]
            lab = 'Ind. %.d'%(j)
            if i == npts-2:
                plt.plot(tps, traj_temp[j][:] ** 2,'o-',markersize=2, 
                        color = clr[j], label = lab, linewidth = 1)
            else: 
                plt.plot(tps, traj_temp[j][:] ** 2, 'o-', 
                        color = clr[j], markersize = 2, linewidth = 1)
        X = np.append(X, x_0 * np.random.uniform())
    plt.xlabel('Time')
    plt.ylabel('Trait value')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
          ncol=npts + n_init -1,markerscale=1, mode="expand", borderaxespad=0.)
    plt.show()
    return()
