import numpy as np
import scipy as sc
from scipy import interpolate
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)
mission = SpaceMission.load('part5.bin')

M_m = const.G*const.m_sun/(const.c**2)
print("The mass of the Sun measured in meters is", M_m)

sun_ratio = M_m/const.R_sun
print("The M/r-ratio of the Sun is", sun_ratio)

doppler_shift = 1/np.sqrt(1-2*sun_ratio) - 1
print("The Doppler shift of the Sun is", doppler_shift)

earth_mass_kg = 5.972e+24
M_earth = const.G*earth_mass_kg/(const.c**2)
print("The mass of the Earth measured in meters is", M_earth)

R_earth = 6.371e+6
earth_ratio = M_earth/R_earth
print("The M/r-ratio of the Earth is", earth_ratio)

doppler_shift_earth = 1/np.sqrt(1-2*earth_ratio) - 1
print("The Doppler shift of the Earth is", doppler_shift_earth)

'''
delta_lambda = 2150 - 600
lambda_sh = 600
M_blackhole =
R_blackhole = 2*M_blackhole/(1 - 1/((delta_lambda/lambda_sh + 1)**2))
'''


















#kjh
