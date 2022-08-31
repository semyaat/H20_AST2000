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

codes = [79374]
shortcuts = SpaceMissionShortcuts(mission, codes)
shortcuts.place_spacecraft_in_stable_orbit(108.8, 140000, 0, 1)

#Egen kode
class Landing:
    def __init__(self):
        self.M = system.masses[1]*const.m_sun
        self.A = mission.lander_area
        self.m = mission.lander_mass
        self.R = system.radii[1]*1000 #Radien til planeten i meter
        rho_saved = np.load('rho.npy')
        rho_height_saved = np.load('rho_position.npy')
        self.rho_function = interpolate.interp1d(rho_height_saved, rho_saved, axis = 0, bounds_error = False, fill_value = 'extrapolate')
        pressure_saved = np.load('pressure.npy')
        self.pressure_function = interpolate.interp1d(rho_height_saved, pressure_saved, axis = 0, bounds_error = False, fill_value = 'extrapolate')

    def rho(self, height):
        return self.rho_function(height)

    def pressure(self, height):
        return self.pressure_function(height)

    def landing_simulation(self, r0, v0, delta_v, thrust_force, dt, N, A_p, thruster_height=500):
        r = np.zeros((N,2))
        v = np.zeros((N,2))
        t = np.zeros((N,1))
        r[0,:] = r0
        v[0,:] = v0 + delta_v #Gir en boost til raketten ved utskytning

        v_safe = 3 #Hastigheten vi ønsker å ha ved landing
        F_L = 0 #Kraft fra landing thrusters
        add_F_L = False #Skal i utgangspunktet ikke ha en kraft fra landing thrusters, den skrus på ved terminal velocity
        omega = 2*np.pi/(system.rotational_periods[1]*const.day)

        for i in range(N-1):
            r_hat = r[i,:]/np.linalg.norm(r[i,:]) #Enhetsvektoren til r
            height = np.linalg.norm(r[i,:]) - self.R #Høyde over bakken
            v_drag = -(omega*height*np.array([-r_hat[1], r_hat[0]]) - v[i,:]) #v_drag er tilsvarende -(w-v)

            F_d = 0.5*self.rho(height)*self.A*np.linalg.norm(v_drag)**2 * (-v_drag/np.linalg.norm(v_drag)) #Dragforce, virker i retning v_drag
            F_g = r_hat*const.G*self.m*self.M/(np.linalg.norm(r[i,:])**2) #Virker i radiell retning
            F_tot = F_d + F_L - F_g

            a = F_tot/self.m
            v[i+1,:] = v[i,:] + a*dt
            r[i+1,:] = r[i,:] + v[i+1,:]*dt

            if height < thruster_height and add_F_L == False:
                F_L = thrust_force*r_hat #Thrusters virker i radiell retning
                add_F_L = True #Settes til true, kjører da bare if-testen en gang

                if 250000 < np.linalg.norm(F_d):
                    print("Drag force exceeded 250.000 N, parachute failed")
                    break

                if 10**7 < self.pressure_function(height):
                    print("Drag pressure exceeded 10^7 Pa, lander burned to ashes")
                    break

            if height < 0.01: #Sjekker om vi er ganske nære bakken, må sette en toleranse da simuleringen ikke er nøyaktig
                self.stop_index = i
                if v_safe < np.linalg.norm(v[i+1,:]):
                    print("Crashed into the planet with a speed of", np.linalg.norm(v[i+1,:]))
                if np.linalg.norm(v[i+1,:]) <= v_safe:
                    print("Succesful landing! Speed:", np.linalg.norm(v[i+1,:]))
                break

        self.r_sim = r #Brukes til plotting

    def actual_landing(self, N, delta_vx, delta_vy, thrust_force, thrust_height=500, parachute_area=300, fall_time=1):
        r = np.zeros((N,3))
        v = np.zeros((N,3))
        t = np.zeros((N,1))

        landing.launch_lander(np.array([delta_vx, delta_vy, 0]))
        landing.adjust_landing_thruster(thrust_force, thrust_height)
        landing.adjust_parachute_area(parachute_area)

        # landing.start_video()
        # landing.look_in_direction_of_motion()

        for i in range(N):
            landing.fall(fall_time)
            t[i], r[i,:], v[i,:] = landing.orient()

        #landing.finish_video(filename="lander_video.xml",number_of_frames=500,radial_camera_offset=1000)
        self.r_landing = r

    def plot_planet(self, landing=False, simulation=False):
        figure, axes = plt.subplots()
        draw_planet = plt.Circle((0, 0), self.R, fill=True, color='darkseagreen')
        axes.set_aspect(1)
        axes.add_artist(draw_planet)

        if landing == True:
            plt.plot(self.r_landing[:,0], self.r_landing[:,1], color='dimgrey', label='Lander')
            plt.plot(self.r_landing[0,0], self.r_landing[0,1], 'ko', markersize=2)

        if simulation == True:
            plt.plot(self.r_sim[:self.stop_index,0], self.r_sim[:self.stop_index,1], color='rosybrown', label='Lander, simulated')
            plt.plot(self.r_sim[0,0], self.r_sim[0,1], 'ko', markersize=2)

        plt.xlabel('Distance [m]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.title('Trajectory of the lander')
        plt.show()

if __name__ == '__main__':
    #Setter initialverdiene fra startposisjon i snarveien
    landing = mission.begin_landing_sequence()
    t, r0, v0 = landing.orient()

    lan = Landing()
    lan.landing_simulation(r0[:2], v0[:2], delta_v=np.array([700,-950]), thrust_force=710, dt=0.1, N=100000, A_p=300) #FINAL VERSION
    lan.actual_landing(N=1427, delta_vx=700, delta_vy=-950, thrust_force=579.3813)
    lan.plot_planet(landing=True, simulation=True)
