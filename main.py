import matplotlib.pyplot as plt
import cl
import numpy as np


track=np.loadtxt("fsg18dv_trackdrive.csv",delimiter=";")


cl = cl.Centerline(track)
s, reftrack, kappa, normvec, psi = cl.discretize(0,cl.end(), N=1000)

w_right = reftrack[:,2]
w_left = reftrack[:,3]
bound_r = reftrack[:, :2] - np.expand_dims(w_right, 1) * normvec[:]
bound_l = reftrack[:, :2] + np.expand_dims(w_left, 1) * normvec[:]

plt.title("Track layout")
plt.subplot(221)
plt.plot(reftrack[:,0], reftrack[:,1], label="centerline")
plt.plot(bound_l[:,0], bound_l[:,1], label="left boundary")
plt.plot(bound_r[:,0], bound_r[:,1], label="right boundary")
plt.gca().set_aspect('equal', adjustable='box') # equal length for axis
plt.legend()

plt.subplot(223)
plt.plot(s, reftrack[:,0], label="x")
plt.plot(s, reftrack[:,1], label="y")
plt.xlabel("Distance")
plt.legend()

plt.subplot(222)
plt.plot(s, kappa, label="kappa")
plt.xlabel("Distance")
plt.legend()

plt.subplot(224)
plt.plot(s, psi, label="psi")
plt.xlabel("Distance")
plt.legend()

plt.show()
