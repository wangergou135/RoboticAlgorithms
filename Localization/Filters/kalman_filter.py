"""
    Prediction

    𝐱¯=𝐅𝐱+𝐁𝐮
    𝐅 is called the state transition function or the state transition matrix.
    and this is the equation that is computed when you call KalmanFilter.predict().

    Δ𝐱=𝐁𝐮

    Here 𝐮 is the control input, and 𝐁 is the control input model or control function

    Your job as a designer is to specify the matrices for
    𝐱, 𝐏 : the state and covariance
    𝐅, 𝐐 : the process model and noise covariance
    𝐁,𝐮 : Optionally, the control input and function
"""



"""
The Kalman filter equation that performs this step is:

𝐲=𝐳−𝐇𝐱¯

where 𝐲
is the residual, 𝐱¯ is the prior, 𝐳 is the measurement, and 𝐇 is the measurement function
 that converts a state into a measurement.
"""

#   08-Designing-Kalman-Filters.ipynb
from numpy.random import randn

class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


import matplotlib.pyplot as plt
import numpy as np
from kf_book.book_plots import plot_measurements

pos, vel = (4, 3), (2, 1)
sensor = PosSensor(pos, vel, noise_std=1)
ps = np.array([sensor.read() for _ in range(50)])
plot_measurements(ps[:, 0], ps[:, 1])

plt.show()

from filterpy.kalman import KalmanFilter

tracker = KalmanFilter(dim_x=4, dim_z=2) # [x dx y dy]
dt = 1.   # time step 1 second

tracker.F = np.array([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]])

from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

# noise for the Newton's motion model
q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
tracker.Q = block_diag(q, q)
print(tracker.Q)

from filterpy.stats import plot_covariance_ellipse
from kf_book.book_plots import plot_filter

R_std = 0.35
Q_std = 0.04

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    tracker.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]])
    tracker.u = 0.
    # measurement is in feet, state is in meters
    # as 𝖿𝖾𝖾𝗍=𝗆𝖾𝗍𝖾𝗋𝗌/0.3048
    tracker.H = np.array([[1/0.3048, 0, 0, 0],
                          [0, 0, 1/0.3048, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4) * 500.
    return tracker

# simulate robot movement
N = 30
sensor = PosSensor((0, 0), (2, .2), noise_std=R_std)

zs = np.array([sensor.read() for _ in range(N)])

# run filter
robot_tracker = tracker1()
mu, cov, _, _ = robot_tracker.batch_filter(zs)

for x, P in zip(mu, cov):
    # covariance of x and y
    cov = np.array([[P[0, 0], P[2, 0]], 
                    [P[0, 2], P[2, 2]]])
    mean = (x[0, 0], x[2, 0])
    plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)
plt.show()
    
#plot results
zs *= .3048 # convert to meters
plot_filter(mu[:, 0], mu[:, 2])
plot_measurements(zs[:, 0], zs[:, 1])
plt.legend(loc=2)
plt.xlim(0, 20)
plt.show()
print(np.diag(robot_tracker.P))


