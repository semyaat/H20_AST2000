import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sc
from scipy import interpolate
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem

random.seed(None)
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)
mission = SpaceMission(seed)

class Destination:
    def __init__(self, G, star_mass, number_of_planets, initial_positions, initial_velocities, planet_masses, rotational_periods, semi_major_axis, Ts, Rs, r_sp):
        self.G = G
        self.star_mass = star_mass
        self.num_plan = number_of_planets
        self.init_pos = initial_positions
        self.init_vel = initial_velocities
        self.plan_mass = planet_masses
        self.rot_per = rotational_periods   #Periodene til alle planetene rundt sin egen akse
        self.semi_axis = semi_major_axis    #Store halvakse til alle planetene
        self.Ts = Ts
        self.Rs = Rs
        self.r_sp = r_sp

   #Egen kode
    def find_analytical_orbit(self,N_points,a,e):
        angles = np.linspace(0,2*np.pi,N_points)
        r_pos = a*(1-e**2)/(1+e*np.cos(angles))

        self.x_analytic = r_pos*np.cos(angles)
        self.y_analytic = r_pos*np.sin(angles)

    def add_habitable_zone(self):
        Rs = system.star_radius/const.AU*1000
        Tp = np.linspace(245,405)

        r_sp = (Rs*np.sqrt(0.25*(self.Ts/Tp)**4))

        angles = np.linspace(0,2*np.pi,10000)
        self.x_hab_start = r_sp[0]*np.cos(angles)
        self.y_hab_start = r_sp[0]*np.sin(angles)
        self.x_hab_end = r_sp[-1]*np.cos(angles)
        self.y_hab_end = r_sp[-1]*np.sin(angles)

        #plt.plot(self.x_hab_start, self.y_hab_start)
        #plt.plot(self.x_hab_end, self.y_hab_end)

        plt.fill_between(self.x_hab_start, self.y_hab_start, alpha=1., color='paleturquoise')
        plt.fill_between(self.x_hab_end, self.y_hab_end, alpha=1.0, color='white')

    #Egen kode
    def plot_analytical_orbit(self, hab_zone):
        #plt.figure(figsize=(10,10))
        fig = plt.figure()
        plt.title("Noon Universe")
        plt.xlabel("Distance in AU")
        plt.ylabel("Distance in AU")
        if hab_zone:
            self.add_habitable_zone()
        for i in range(system.number_of_planets):
            calc_orb.find_analytical_orbit(1000,system.semi_major_axes[i], system.eccentricities[i])
            plt.plot(self.x_analytic, self.y_analytic, label=f'{system.types[i]}',linestyle="dashed")
        plt.gca().set_aspect('equal',adjustable='box')
        plt.legend()
        #plt.savefig('filename.png', dpi=300)
        plt.show()

    def planet_temp(self):
        for i in range(8):
            r_sp = system.semi_major_axes[i]*const.AU/1000
            Tp = self.Ts*(((self.Rs/r_sp)**2)/4)**(1/4)
            Tp_C = Tp - 274.15                                     #in Celcius
            print(f"{i}: {Tp:.1f} K", f"{Tp_C:.1f} C")
            self.Tp = Tp


if __name__ == '__main__':
    calc_orb = Destination(const.G_sol, system.star_mass, system.number_of_planets, system.initial_positions, system.initial_velocities, system.masses, system.rotational_periods, system.semi_major_axes, system.star_temperature, system.star_radius, system.semi_major_axes)
    calc_orb.plot_analytical_orbit(True)
    calc_orb.planet_temp()

    def earth_temp():
        # Checks the formula for Earth's temperature

        r_sp = 149600000 #km const.AU/1000
        Ts = 5778
        Rs = 696340

        Tp = Ts*(((Rs/r_sp)**2)/4)**(1/4)
        Tp_C = Tp - 274.15                                     #in Celcius
        print(f"Earth: {Tp:.1f} K", f"{Tp_C:.1f} C")
    earth_temp()
