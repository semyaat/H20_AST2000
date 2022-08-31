import numpy as np
import scipy as sc
from scipy import interpolate
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.relativity import RelativityExperiments
import matplotlib.pyplot as plt
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)

# r0 = system.radii[1]*1000
# print(r0)
# r1 = np.linalg.norm(np.array([-14548.798*1000, -7991.458*1000]))
# r2 = np.linalg.norm(np.array([-8603.899*1000, -14195.204*1000]))
# t1 = 513.4350253*const.c_km_pr_s
# t2 = 513.4258409*const.c_km_pr_s
#
# print((r0**2 + r1**2 - t1**2)/(2*r0*r1),(r0**2 + r2**2 - t2**2)/(2*r0*r2))
# alpha1 = np.arccos((r0**2 + r1**2 - t1**2)/(2*r0*r1))
# alpha2 = np.arccos((r0**2 + r2**2 - t2**2)/(2*r0*r2))
#
# print(alpha1, alpha2)

# M = 0.5
# r = np.linspace(0,50,1000)
#
# def V(r):
#     return np.sqrt((1-2*M/r)/(r**2))
#
# plt.plot(r, V(r))
# plt.xlabel("Distance")
# plt.ylabel("Potential")
# plt.title("Potential near a black hole")
# plt.show()

M = 0.5
m = 0.001
theta = np.deg2rad(167)
v_sh = 0.993
R = 20*M
gamma_sh = 1/np.sqrt(1 - v_sh**2)
L_m = R*gamma_sh*v_sh*np.sin(theta)

r_extremum1 = L_m**2/2*M*(1 + np.sqrt(1 - 12*M**2/(L_m**2)))
r_extremum2 = L_m**2/2*M*(1 - np.sqrt(1 - 12*M**2/(L_m**2)))

r = np.linspace(0,20,1000)

def V_eff(r):
    return np.sqrt((1-2*M/r)*(1+(L_m**2)/(r**2)))

plt.plot(r*M, V_eff(r)/m)
# plt.plot(r_extremum1, V_eff(r_extremum1), marker='o', markersize=3, color="red")
# plt.plot(r_extremum2, V_eff(r_extremum2), marker='o', markersize=5, color="k")
plt.title("Potential of a rocket near a black hole")
plt.xlabel("Distance")
plt.ylabel("Effective potential")
plt.show()

print(V_eff(r_extremum2))









#HEI
