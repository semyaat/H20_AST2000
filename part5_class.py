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

mission = SpaceMission.load('part4.bin') #Bruker denne verify_manual_orientation

#Egen kode
class PlannedTrajectory:
    def __init__(self):
        self.time_planets = np.load('plan_times.npy')
        self.r_p = np.load('plan_pos_interpol.npy')
        self.launch_index = np.load('launch_index.npy')
        self.r0 = np.load('final_position.npy')
        self.v0 = np.load('final_velocity.npy')
        self.e_v = self.v0/np.linalg.norm(self.v0, axis=0) #Lager en enhetsvektor av initialhastigheten

    def new_inital_position(self, theta):
        '''Bruker denne hvis vi trenger å oppdatere oppskytningsposisjonen på planeten.
        Denne gis som input til snarveien i del 1, verifiserer oppskytningen, og gir oss
        posisjon og hastighet til raketten når den er skutt opp. Disse lagres i 'final_position.npy'
        og 'final_velocity.npy', som hentes ut i planned_trajectory'''
        from gen_launch_codes import gen_launch
        gen_launch.planet_positions()
        launch_index = gen_launch.launch_window()
        gen_launch.generalized_initial_conditions(theta)
        gen_launch.convert_to_solar_system_frame()
        gen_launch.set_parameters_and_verify()
        np.save('launch_index.npy', launch_index)

    def planned_trajectory(self, dt):
        T = self.estimated_travel_time()
        N = int(T/dt)  #Antall utregninger, (antatt) sluttid delt på tidssteg
        r = np.zeros((N,2))
        v = np.zeros((N,2))
        t = np.zeros((N,1))

        r_p = self.r_p
        e_v = self.e_v

        r[0,:] = self.r0
        v[0,:] = self.v0
        t[0] = self.time_planets[self.launch_index] #Henter ut tidsarrayen til planetene, tid gitt i år

        planet_mass = system.masses #Henter ut massen til alle planetene
        Ms = system.star_mass #Solmasse
        r_p = self.r_p #Henter de interpolerte planetposisjonene

        for i in range(N-1):
            a_planet = 0
            for j in range(7):
                '''Itererer gjennom alle planetene for å summere kreftene fra dem på raketten.'''
                a_planet += const.G_sol*planet_mass[j]*(r[i,:] - r_p[self.launch_index:,j])/np.linalg.norm(r[i,:] - r_p[self.launch_index:,j], axis=0)**3

            a = -const.G_sol*Ms*r[i,:]/np.linalg.norm(r[i,:], axis=0)**3 - a_planet[0]

            #Boosts
            v[10,:] += -1.67*e_v
            v[420,:] += 0.69*e_v
            v[440,:] += 0.1*e_v
            v[450,:] += -0.05*e_v[0]
            v[460,:] += -0.04*e_v[0]

            v[i+1,:] = v[i,:] + a*dt
            r[i+1,:] = r[i,:] + v[i+1,:]*dt
            t[i+1] = t[i] + dt

            #Sjekker om vi er nære nok destinasjonsplaneten
            distance = np.linalg.norm(r[i+1,:], axis=0)
            l = distance*np.sqrt(planet_mass[1]/(10*Ms))
            destination_rocket_vec = np.linalg.norm(r[i+1,:] - r_p[181697,1,:], axis=0)

            if destination_rocket_vec <= l:
                print("Close enough B-)")
                break

        print(f'Rocket distance from destination planet: {destination_rocket_vec:.3f},\
              \nMinimum distance needed to perfor orbital injection maneuver: {l:.3f}')

        #Trenger disse til plotting
        self.r = r
        self.T = T

    def plot_planned_trajectory(self):
        stop_plot = self.launch_index + int(self.T/(self.time_planets[1] - self.time_planets[0])) #Deler sluttid på tidssteg, legger til launch_index
        from orbits import calc_orb
        r0 = self.r0
        r = self.r
        r_p = self.r_p

        #Henter fra del 2, plotter de analytiske planene
        calc_orb.plot_analytical_orbit()

        #Plotter de integrerete banene til hjemplaneten og destinasjonsplaneten
        color_arr = np.array(['c', 'r']) #Fargene til planetbanene
        for planet in range(2):
            plt.plot(r_p[self.launch_index:stop_plot, planet, 0], r_p[self.launch_index:stop_plot, planet, 1], color=color_arr[planet])

        '''Plotter raketten og destinasjonsplaneten ved launch, og så ved ankomst.
        Plotter også sluttposisjonen til hjemplanetenv vår'''
        plt.plot(r_p[self.launch_index,1,0], r_p[self.launch_index,1,1], "ro", markersize=4)
        plt.plot(r0[0], r0[1], "co", markersize=4, label='home planet')
        plt.plot(r_p[stop_plot,0,0], r_p[stop_plot,0,1], "co", markersize=4) #Plotter hjemplaneten der den stopper
        plt.plot(r_p[stop_plot,1,0], r_p[stop_plot,1,1], "ro", markersize=4, label='destination planet') #Plotter destinasonsplaneten der den stopper
        plt.plot(r[-1,0], r[-1,1], "ko", markersize=3) #Plotter raketten der den stopper, men feil indeksering

        #Plotter sånn ca når planeten vår evt er i nærheten
        plt.plot(r_p[173925+6700,1,0], r_p[173925+6700,1,1], "bo", markersize=3)
        plt.plot(r_p[173925+7400,1,0], r_p[173925+7400,1,1], "bo", markersize=3)

        '''Plotter bevegelsen til raketten, de analytiske banene til planetene, og de
        faktiske banene deres ila reisetiden'''
        plt.plot(r[:,0], r[:,1], color='k', label='Rocket trajectory') #Rakett
        plt.legend()

    def estimated_travel_time(self):
        '''Planetene beveger seg ca i sirkel, tar utgangspunkt i at avstand til sola er lik hele veien.
        Bruker derfor avstanden til sola ved t=0'''
        a_t = 0.5*(np.linalg.norm(self.r_p[0,0]) + np.linalg.norm(self.r_p[0,1])) #Store halvakse til transfer orbit
        P_t = np.sqrt((4*np.pi**2/(const.G_sol*(system.star_mass))*a_t**3)) #Perioden til transfer orbit, ignorerer massen til raketten, da den er veldig liten ift sola
        T = P_t/2
        print("Estimated time of travel", T)
        return T

    def after_launch_cheat(self):
        codes = [37938]
        shortcuts = SpaceMissionShortcuts(mission, codes) #Dette er for place_spacecraft_on_escape_trajectory

        #Lengden av vektoren mellom planetens sentrum og raketten, bruker for å finne høyde over planeten
        d = np.linalg.norm(self.r_p[self.launch_index,0,:] - self.r0, axis=0)*const.AU/1000 #Gjør om fra AU til km

        #Setter sånn ca verdier
        rocket_thrust = 12000
        rocket_mass_loss_rate = 0.5
        self.time = self.time_planets[self.launch_index]
        height_above_surface = d - system.radii[0]
        direction_angle = 230 #Regnet ut i gen_launch_codes
        remaining_fuel_mass = 25000
        shortcuts.place_spacecraft_on_escape_trajectory(rocket_thrust, rocket_mass_loss_rate, self.time, \
                                                        height_above_surface, direction_angle, remaining_fuel_mass)
        SpaceMission.save('part5.bin', mission)

    def interplanetary_travel(self, N, end_coast_time, restart_mission=False):
        pos_orient = np.zeros((N,2))
        vel_orient = np.zeros((N,2))
        time_orient = np.zeros((N,1))

        self.after_launch_cheat()
        end_coast_time = end_coast_time + self.time #VURDER Å SLETTE, bruker ikke?

        travel = mission.begin_interplanetary_travel()

        '''Kan filme reisen ved å se i flere forskjellige retninger ved å bruke dette.
        Videoen slutter når vi er ferdig å reise'''
        travel.start_video()
        travel.look_in_direction_of_planet(0)
        #travel.look_in_direction_of_motion()

        #År 0, orientering
        time_orient[0], pos_orient[0,:], vel_orient[0,:] = travel.orient() #Tid er gitt i år, posisjon og hastighet er i AU(/år) ift stjernen

        '''
        #Alt under er fra planen som ikke fungerte:
        #År 0.1
        travel.coast(0.1)
        time_orient[1], pos_orient[1,:], vel_orient[1,:] = travel.orient()
        travel.boost(-1.67*self.e_v)

        #År 4.2
        travel.coast(4.1)
        time_orient[2], pos_orient[2,:], vel_orient[2,:] = travel.orient()
        travel.boost(0.69*self.e_v)

        #År 4.4
        travel.coast(0.2)
        time_orient[3], pos_orient[3,:], vel_orient[3,:] = travel.orient()
        travel.boost(0.1*self.e_v)

        #År 4.5
        travel.coast(0.1)
        time_orient[4], pos_orient[4,:], vel_orient[4,:] = travel.orient()
        travel.boost(-0.05*self.e_v)

        #År 4.6, boost
        travel.coast(0.1)
        time_orient[5], pos_orient[5,:], vel_orient[5,:] = travel.orient()
        travel.boost(-0.04*self.e_v)
        '''

        #År 0.001, orientering
        travel.coast(0.001)
        time_orient[1], pos_orient[1,:], vel_orient[1,:] = travel.orient()

        #År 0.5, boost
        travel.coast(0.5)
        time_orient[2], pos_orient[2,:], vel_orient[2,:] = travel.orient()
        travel.boost(0.8*self.e_v)

        #År 1, orientering
        travel.coast(0.5)
        time_orient[3], pos_orient[3,:], vel_orient[3,:] = travel.orient()

        #År 1.5, boost
        travel.coast(0.5)
        time_orient[4], pos_orient[4,:], vel_orient[4,:] = travel.orient()
        travel.boost(-0.3*self.e_v)

        #År 2, boost
        travel.coast(0.5)
        time_orient[5], pos_orient[5,:], vel_orient[5,:] = travel.orient()
        travel.boost(-0.4*self.e_v)

        #År 2.5, orientering
        travel.coast(0.5)
        time_orient[6], pos_orient[6,:], vel_orient[6,:] = travel.orient()

        #År 3, boost
        travel.coast(0.5)
        time_orient[7], pos_orient[7,:], vel_orient[7,:] = travel.orient()
        travel.boost(0.5*self.e_v)

        #År 3.5, boost
        travel.coast(0.5)
        time_orient[8], pos_orient[8,:], vel_orient[8,:] = travel.orient()
        travel.boost(-0.7*self.e_v)

        #År 4, boost
        travel.coast(0.5)
        time_orient[9], pos_orient[9,:], vel_orient[9,:] = travel.orient()
        travel.boost(np.array([0,1.5*self.e_v[1]]))

        # travel.finish_video(filename='travel_video_launch.xml', number_of_frames=500)

        if restart_mission == True:
            travel.restart()

        #Bruker dette for å plotte senere
        self.pos_orient = pos_orient
        self.time_orient = time_orient

        travel.record_destination(1) #Lagrer posisjon til rakett og destinasjonsplaneten

    def plot_interplanetary_travel(self, N, dt):
        #Plotter den planlagte banen vår og de analytiske banene
        self.planned_trajectory(dt)
        self.plot_planned_trajectory()
        pos_orient = self.pos_orient
        time_orient = self.time_orient

        #Plotter punkter fra interplanetary_travel
        for i in range(N):
            plt.plot(pos_orient[i,0], pos_orient[i,1], color="k", marker='x', markersize=3)
            plt.annotate(f"{time_orient[i] - self.time}", (pos_orient[i,0], pos_orient[i,1]))

    def unstable_orbit(self):
        mission = SpaceMission.load('part5.bin')
        codes = [97520]
        shortcuts = SpaceMissionShortcuts(mission, codes)

        planet_idx = 1 #Destinasjonsplanet
        time = 108.8    #Tiden vi vil være fremme i år
        shortcuts.place_spacecraft_in_unstable_orbit(time, planet_idx)
        travel = mission.ongoing_interplanetary_travel

        #Lagrer posisjonen når vi er i bane rundt destinasjonsplaneten
        N = 250 #250 utregninger er ca en periode
        coast_time = 0.001
        position = np.zeros((N, 2))
        velocity = np.zeros((N, 2))
        time = np.zeros((N, 1))

        # travel.start_video()
        # travel.look_in_direction_of_planet(1)

        for i in range(N-1):
            travel.coast(coast_time)
            time[i], position[i,:], velocity[i,:] = travel.orient()
        #travel.finish_video(filename='travel_video_orbit_minivideo.xml', number_of_frames=100)

        np.save('position_in_orbit.npy', position)
        np.save('velocity_in_orbit.npy', velocity)
        np.save('times_in_orbit.npy', time)

    def change_frame_of_reference(self, plot=True):
        rocket_position = np.load('position_in_orbit.npy') #Posisjonen til raketten ift sola
        planet_times_orbit = np.load('times_in_orbit.npy') #Tiden raketten er i orbit

        planet_positions = np.load('planet_positions.npy') #Alle planetposisjoner, simulert over 21 perioder
        planet_times = np.load('planet_times.npy') #Tidsarrayen over 21 perioder

        #Interpolerer planetposisjonene og tar inn tidspunktene vi vil se på
        interpol_func = interpolate.interp1d(planet_times, planet_positions[:,1,:], axis = 0, bounds_error = False, fill_value = 'extrapolate')
        interpol_planet_orbit_pos = interpol_func(planet_times_orbit) #Den interpolerte posisjonen til destinasjonsplaneten ila tidspunktene der vi er i orbit rundt den

        rocket_pos_plan = rocket_position - interpol_planet_orbit_pos[:,0,:]  #Vektoren fra destinasonsplaneten til raketten

        #Plotter orbit til raketten rundt destinasjonsplaneten
        plt.plot(rocket_pos_plan[:-1,0], rocket_pos_plan[:-1,1], 'orange') #Må ta bort siste index for å unngå en strek
        plt.plot(0.0, 0.0, 'ko', markersize='12')
        plt.xlabel('Distance in AU'); plt.ylabel('Distance in AU')
        plt.title('Orbit Around Destination Planet (1 year)')
        plt.gca().set_aspect('equal',adjustable='box')
        if plot == True:
            plt.show()

        return rocket_pos_plan[:-1]

    def orbit_info(self):
        rocket_pos_plan = self.change_frame_of_reference(plot=False)
        rocket_vel_plan = np.load('velocity_in_orbit.npy')
        rocket_times_plan = np.load('times_in_orbit.npy')

        #Avstand til planeten
        abs_r = np.linalg.norm(rocket_pos_plan[0])

        #Radiell hastighet rett etter at vi er i orbit
        unit_r = rocket_pos_plan[0]/abs_r
        radial_vel = np.dot(rocket_vel_plan[0], unit_r)

        #Vinkelkomponenten av hastigheten
        delta_S = np.sqrt(np.linalg.norm(rocket_pos_plan[1])**2 - np.linalg.norm(rocket_pos_plan[0])**2) #AU
        delta_t = rocket_times_plan[1] - rocket_times_plan[0] #År
        angular_comp_vel = delta_S/delta_t

        #Finner store halvakse a
        periapsis = np.amin(np.linalg.norm(rocket_pos_plan, axis=1), axis=0)
        apoapsis = np.amax(np.linalg.norm(rocket_pos_plan, axis=1), axis=0)
        semi_major_axis = (periapsis + apoapsis)/2

        #Finner lille halvakse b og eksentrisiteten
        c = apoapsis - semi_major_axis
        eccentricity = c/semi_major_axis #e = c/a
        semi_minor_axis = semi_major_axis*np.sqrt(1 - eccentricity**2) #b = a*sqrt(1 - e^2)

        #Finner perioden
        period = 2*np.pi*np.sqrt(semi_major_axis**3/(const.G_sol*(system.masses[1] + mission.spacecraft_mass/const.m_sun)))

        #Sjekker at periapsis og apoapsis er riktig sted
        periapsis_pos = rocket_pos_plan[np.argmin(np.linalg.norm(rocket_pos_plan, axis=1))]
        apoapsis_pos = rocket_pos_plan[np.argmax(np.linalg.norm(rocket_pos_plan, axis=1))]

        plt.plot(periapsis_pos[0], periapsis_pos[1], 'ro')
        plt.plot(apoapsis_pos[0], apoapsis_pos[1], 'ro')
        plt.show()

        print("Distance from planet:", f'{abs_r*const.AU/1000:.2f}', "km")
        print("Radial velocity:", f'{radial_vel*17066:.2f}', "km/h")
        print("Angular velocity:", f'{angular_comp_vel[0]:.2f}')


if __name__ == '__main__':
    pl_tr = PlannedTrajectory()
    # #pl_tr.new_inital_position(150) #Kun kall på denne ved endring i initialposisjon
    #
    # pl_tr.planned_trajectory(0.01)
    # pl_tr.plot_planned_trajectory()
    # plt.show()
    #
    # N = 10 #Endrer denne ut i fra hvor mange orienteringer vi gjør
    # pl_tr.interplanetary_travel(N, 4.86)
    # pl_tr.plot_interplanetary_travel(N, 0.01)
    # plt.show()
    pl_tr.after_launch_cheat()

    #pl_tr.unstable_orbit() #Snarvei, OBS tar lang tid, men lagrer info i position_in_orbit.npy
    # pl_tr.change_frame_of_reference()
    # pl_tr.orbit_info()
