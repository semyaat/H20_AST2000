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

relativity = RelativityExperiments(seed)
#relativity.spaceship_duel(1)
#relativity.cosmic_pingpong(1)
#relativity.spaceship_race(1)
#relativity.laser_chase(1)
#relativity.neutron_decay(1)
#relativity.antimatter_spaceship(1)
relativity.black_hole_descent(1, consider_light_travel=True)
# relativity.crash_landing(1)
# relativity.twin_paradox(1)
#relativity.gps(1)

#print(system.masses[1]*const.m_sun)
