import numpy as np
import scipy as sc
from scipy import interpolate
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)
#mission = SpaceMission(seed)

#Snarvei fra del 1:
codes = [87556]
mission = SpaceMission.load('part1.bin')
shortcuts = SpaceMissionShortcuts(mission, codes)

from part1_classes_snarvei import launch
#Må kjøre denne biten for å få simulate_rocket_launch til å fungere
launch.moving_particles(100000,1000,10**(-6))

#Alt er egen kode
class General_launch:
    def __init__(self, rotational_periods, semi_axis):
        self.rot_per = rotational_periods   #Periodene til alle planetene rundt sin egen akse
        self.semi_axis = semi_axis


    def planet_positions(self):
        #Henter ut de riktige planetposisjonene
        with np.load('planet_trajectories.npz') as f:
            times = f['times']
            exact_planet_positions = f['planet_positions']

        position_function = interpolate.interp1d(times, np.transpose(exact_planet_positions), axis = 0, bounds_error = False, fill_value = 'extrapolate')

        self.planet_positions = position_function(times)
        self.planet_velocities = np.gradient(self.planet_positions, axis=0) #Deriverer posisjonen for å få hastigheten
        self.t_planets = times

        np.save('plan_pos_interpol.npy', self.planet_positions)
        np.save('plan_vel_interpol.npy', self.planet_velocities)
        np.save('plan_times.npy', self.t_planets)


    def launch_window(self):
        #Finner store halvakse til transfer orbit, antar at radien til planetbanene er ca lik hele veien
        a_t = 0.5*(np.linalg.norm(self.planet_positions[0,0,:]) + np.linalg.norm(self.planet_positions[0,1,:]))

        #Finner perioden til transfer orbit
        P_t = np.sqrt(4*np.pi**2/(const.G_sol*(system.star_mass))*a_t**3) #Ignorerer

        #Finner tiden det vil ta å reise fra planeten til destinasjonsplaneten
        T_t = P_t/2
        print("Dette er transfer tid", T_t)

        #Finner perioden til destinasjonsplaneten
        P_1 = np.sqrt(4*(self.semi_axis[1]**3)*np.pi**2/(const.G_sol*(system.star_mass + system.masses[1])))

        #Finner vinkelen mellom planetene ved launch
        theta = np.deg2rad(180 - 360*T_t/P_1)

        r = self.planet_positions
        least_squares = np.zeros(len(self.planet_positions[:,0,:]))

        for i in range(len(self.planet_positions[:,0,:])):
            cos_theta = np.dot(r[i,0,:], r[i,1,:])/(np.linalg.norm(r[i,0,:], axis=0)*np.linalg.norm(r[i,1,:],axis=0))
            diff = (cos_theta - np.cos(theta))**2
            least_squares[i] = np.sum(diff)

        launch_index = np.argmin(least_squares)
        print("Dette er index", launch_index)
        self.launch_index = launch_index

        return self.launch_index #Bruker dette i del 5


    def generalized_initial_conditions(self, theta1): #Beholder den i tilfelle vi vil endre theta, må da fjerne utregnet theta
        '''Må regne om på vinkelen her'''
        x, y = self.planet_positions[self.launch_index,0,0], self.planet_positions[self.launch_index,0,1]
        theta = np.arctan2(y, x) + np.pi/2

        print("Dette er launch vinkel", np.rad2deg(theta))

        plan_rad = system.radii[0]*1000 #Radien til hjemplaneten i meter
        rot_per = self.rot_per[0]
        v_tan = 2*np.pi*plan_rad*1000/(rot_per*24*60*60) #Tangentiell hastighet fra jordas rotasjon, gjort om til m/s

        #Bytter om til polarkoordinater, kan nå plassere raketten hvor som helst
        self.r0 = np.array([plan_rad*np.cos(theta), plan_rad*np.sin(theta)])
        self.v0 = np.array([-v_tan*np.sin(theta), v_tan*np.cos(theta)])


    def convert_to_solar_system_frame(self):
        #Konverterer til et koordinatsystem med stjernen i origo
        r0 = self.r0/const.AU

        #Bruker r og v her, bruker de andre parametrene i set_parameters_and_verify
        r, v, self.t_rocket, remaining_fuel, self.fuel_con, self.tot_thrust, self.init_fuel = launch.simulate_rocket_launch(100000,24000,10**(12),1200,1000, self.r0, self.v0)

        #Dette er posisjon og hastighet til raketten etter launch
        self.r = r/const.AU             #Posisjonen gitt i meter, konverterer til AU
        self.v = v*const.c_AU_pr_yr     #Hastigheten gitt i m/s, konverterer til AU/year

        init_plan_pos = self.planet_positions[self.launch_index,0,:]
        init_plan_vel = self.planet_velocities[self.launch_index,0,:]

        self.rocket_position_before_launch =  np.array([init_plan_pos[0] + r0[0], init_plan_pos[1] + r0[1]])
        self.rocket_position_after_launch = np.array([init_plan_pos[0] + r[-1][0], init_plan_pos[1] + r[-1][1]]) #Posisjonen til planeten ift stjernen + raketten ift planeten
        self.rocket_velocity_after_launch = np.array([init_plan_vel[0] + v[-1][0], init_plan_vel[1] + v[-1][1]]) #Hastigheten til planeten ift stjernen + raketten ift planeten


    def set_parameters_and_verify(self):
        mass_loss_rate_per_box_times_number_of_boxes = self.fuel_con
        thrust_per_box_times_number_of_boxes = abs(self.tot_thrust)
        initial_fuel_mass = self.init_fuel
        launch_duration = self.t_rocket[-1] #Siste tidspunkt i tidsarrayen fra rocket launchen
        rocket_position_before_launch = self.rocket_position_before_launch #Denne er relativ til stjernen
        time_of_launch = self.t_planets[self.launch_index] #Behold denne launch indexen hvis vi bruker Hohmann transfer


        mission.set_launch_parameters(thrust_per_box_times_number_of_boxes, mass_loss_rate_per_box_times_number_of_boxes,
                                      initial_fuel_mass, launch_duration, rocket_position_before_launch, time_of_launch)

        #Brukt snarvei:
        mission.launch_rocket()
        consumed_fuel_mass, final_time, final_position, final_velocity = shortcuts.get_launch_results()
        mission.verify_launch_result(final_position)

        print("Drivstoff brukt:", consumed_fuel_mass, "endelig tid", final_time)

        np.save('final_position.npy', final_position)
        np.save('final_velocity.npy', final_velocity)


gen_launch = General_launch(system.rotational_periods, system.semi_major_axes)

if __name__ == '__main__':
    gen_launch.planet_positions()
    gen_launch.launch_window()
    gen_launch.generalized_initial_conditions(20) #Input her gjør ikke noe, men beholder den hvis vi vil ha en annen theta senere
    gen_launch.convert_to_solar_system_frame()
    gen_launch.set_parameters_and_verify()
