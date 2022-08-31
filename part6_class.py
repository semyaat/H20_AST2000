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

#Alt er egen kode

class SpectralLines:
    def __init__(self):
        u = 1.66053886e-27
        self.lambda0_list = [632, 690, 760, 720, 820, 940, 1400, 1600, 1660, 2200, 2340, 2870] #Nanometer
        self.mass_list = [15.999*u, 15.999*u, 15.999*u, 18.05*u, 18.05*u, 18.05*u, 44.01*u, 44.01*u, 16.04*u, 16.04*u, 58.93*u, 44.01*u] #Kg


    def read_file(self, fileinput, filename1, filename2=None):
        file = np.loadtxt(fileinput)
        np.save(f'{filename1}.npy', file[:,0])
        if filename2:
            np.save(f'{filename2}.npy', file[:,1])


    def plot_all(self):
        all_lambda = np.load('spectral_lines.npy')
        all_flux = np.load('flux_observed.npy')

        plt.plot(all_lambda, all_flux)
        plt.title("Spectral lines")
        plt.xlabel('Spectral lines [nm]')
        plt.ylabel('Flux (normalized) [W/m^2]')
        plt.show()


    def slice_arrays(self, lambda_ref):
        all_lambda = np.load('spectral_lines.npy')
        all_flux = np.load('flux_observed.npy')
        all_sigma = np.load('sigma_noise.npy')

        vmax = 10
        delta_lambda = (vmax/const.c_km_pr_s)*lambda_ref #Forskyvningen av spektrallinjene, km/s
        lambda_min = np.greater_equal(all_lambda, lambda_ref-delta_lambda)
        lambda_max = np.less_equal(all_lambda, lambda_ref+delta_lambda)
        lambda_range = np.logical_and(lambda_min, lambda_max)
        gas_index = np.where(lambda_range == True)
        lambda_sliced = all_lambda[gas_index]
        flux_sliced = all_flux[gas_index]
        sigma_sliced = all_sigma[gas_index]
        return lambda_sliced, flux_sliced, sigma_sliced


    def flux_mod_input(self, lambda0, mass, test_values=20):
        delta_lambda = (10/const.c_km_pr_s)*lambda0
        lambda0_values = np.linspace(lambda0 - delta_lambda, lambda0 + delta_lambda, test_values)
        T = np.linspace(150, 450, test_values)
        sigma_values = (lambda0/const.c)*np.sqrt(const.k_B*T/mass)
        flux_values = np.linspace(0.7, 1.0, test_values)
        return lambda0_values, sigma_values, flux_values


    def flux_mod(self, Fmin, lambda0, sigma, lambda_sliced):
        f_mod = np.zeros(len(lambda_sliced))
        for i in range(len(lambda_sliced)):
            f_mod[i] = 1 + (Fmin-1)*np.exp(-0.5*((lambda_sliced[i] - lambda0)/(sigma))**2)
        return f_mod


    def least_chi_squares(self, lambda0, mass, lambda_sliced, flux_sliced, sigma_sliced):
        lambda0_values, sigma_values, flux_values = self.flux_mod_input(lambda0, mass)

        #Initialverdi for X
        chi_previous = np.sum(((flux_sliced - self.flux_mod(flux_values[0], lambda0_values[0], sigma_values[0], lambda_sliced))/sigma_sliced)**2)

        for flux in flux_values:
            for lambda0 in lambda0_values:
                for sigma in sigma_values:
                    chi = np.sum(((flux_sliced - self.flux_mod(flux, lambda0, sigma, lambda_sliced))/sigma_sliced)**2)

                    if chi < chi_previous:
                        flux_optimal = flux
                        lambda0_optimal = lambda0
                        sigma_optimal = sigma
                        chi_previous = chi #Oppdaterer X-verdien
        return flux_optimal, lambda0_optimal, sigma_optimal


    def find_every_line(self):
        lambda0_list = self.lambda0_list #Nanometer
        mass_list = self.mass_list #Kg

        parameter_list = []

        for lambda0, mass in zip(lambda0_list, mass_list):
            lambda_sliced, flux_sliced, sigma_sliced = self.slice_arrays(lambda0)

            flux_optimal, lambda0_optimal, sigma_optimal = self.least_chi_squares(lambda0, mass, lambda_sliced, flux_sliced, sigma_sliced)
            parameters = lambda0_optimal, flux_optimal, sigma_optimal
            parameter_list.append(parameters)

        parameter_array = np.array(parameter_list)
        np.save('parameters_chi_square.npy', parameter_array)


    def plot_spectral_lines(self):
        lambda_ref = self.lambda0_list
        mass = self.mass_list
        parameters = np.load('parameters_chi_square.npy')
        lambda_opt, f_opt, s_opt = parameters[:,0], parameters[:,1], parameters[:,2]
        type_of_gas = ['Oxygen', 'Oxygen', 'Oxygen', 'Water', 'Water', 'Water', 'Carbon Dioxide', 'Carbon Dioxide', 'Methane', 'Methane', 'Carbon Monoxide', 'Nitrous Oxide']

        for i in range(12):
            lambda_sliced, flux_sliced, sigma_sliced = self.slice_arrays(lambda_ref[i])
            flux_model = self.flux_mod(f_opt[i], lambda_opt[i], s_opt[i], lambda_sliced)
            plt.title(type_of_gas[i])
            plt.xlabel('Spectral lines [nm]')
            plt.ylabel('Flux (normalized) [W/m^2]')
            plt.plot(lambda_sliced, flux_sliced, label='Measured flux')
            plt.plot(lambda_sliced, flux_model, label='Gaussian line profile')
            plt.legend()
            plt.show()


    def spectral_line_info(self):
        '''Skal finne hastighet og temperatur til partiklene i atmosfæren'''
        parameters = np.load('parameters_chi_square.npy')
        lambda0, flux, sigma = parameters[:,0], parameters[:,1], parameters[:,2]
        m = self.mass_list

        temperature = [] #Temperatures in Kelvin

        for i in range(12):
            T = m[i]/const.k_B*(const.c*sigma[i]/lambda0[i])**2
            temperature.append(T)

        v_max = []

        for i in range(12):
            delta_lambda = (10/const.c_km_pr_s)*lambda0[i]
            v = const.c*(delta_lambda/lambda0[i])
            v_max.append(v)


class Enter_orbit:
    def unstable_orbit(self):
        codes = [97520]
        shortcuts = SpaceMissionShortcuts(mission, codes)

        planet_idx = 1 #Destinasjonsplanet
        arrival_time = 108.8    #Tiden vi vil være fremme i år
        shortcuts.place_spacecraft_in_unstable_orbit(arrival_time, planet_idx)
        travel = mission.ongoing_interplanetary_travel

        N = 250 #250 utregninger er ca en periode
        coast_time = 0.001

        position = np.zeros((N, 2))
        velocity = np.zeros((N, 2))
        time = np.zeros((N, 1))

        time[0], position[0,:], velocity[0,:] = travel.orient()
        travel.boost(self.orbital_injection(travel, time, position[0,:], velocity[0,:])) #Utfører en orbital injection maneuver

        for i in range(1, N-1):
            travel.coast(coast_time)
            time[i], position[i,:], velocity[i,:] = travel.orient()

        return position, time


    def orbital_injection(self, travel, time, position0, velocity0):
        r_unit = position0/np.linalg.norm(position0)
        e_x = np.array([1, 0])
        e_y = np.array([0, 1])

        cos_theta = r_unit[0]
        sin_theta = r_unit[1]
        e_theta = -sin_theta*e_x + cos_theta*e_y

        v_stable = np.sqrt(const.G_sol*system.masses[1]/np.linalg.norm(position0))
        v_inj_boost = -e_theta*v_stable - velocity0 #Delta v = v - v0
        return v_inj_boost


    def change_frame_of_reference(self, plot=True):
        rocket_position, planet_times_orbit = self.unstable_orbit()

        planet_positions = np.load('planet_positions.npy') #Alle planetposisjoner, simulert over 21 perioder
        planet_times = np.load('planet_times.npy') #Tidsarrayen over 21 perioder

        #Interpolerer planetposisjonene og tar inn tidspunktene vi vil se på
        interpol_func = interpolate.interp1d(planet_times, planet_positions[:,1,:], axis = 0, bounds_error = False, fill_value = 'extrapolate')
        interpol_planet_orbit_pos = interpol_func(planet_times_orbit) #Den interpolerte posisjonen til destinasjonsplaneten ila tidspunktene der vi er i orbit rundt den

        rocket_pos_plan = rocket_position - interpol_planet_orbit_pos[:,0,:]  #Vektoren fra destinasonsplaneten til raketten

        #Plotter orbit til raketten rundt destinasjonsplaneten
        plt.plot(rocket_pos_plan[:-1,0], rocket_pos_plan[:-1,1], 'orange') #Må ta bort siste index for å unngå en strek
        plt.plot(0.0, 0.0, 'ko', markersize='12') #Her skal planeten  være etter bytting av ref.system
        plt.plot(rocket_position[0,0], rocket_position[0,1], 'ko') #Plotter rakettens posisjon ved start, ikke byttet referansesystem
        plt.plot(rocket_position[:-1,0], rocket_position[:-1,1], 'orange')
        plt.xlabel('Distance in AU'); plt.ylabel('Distance in AU')
        plt.title('Orbit Around Destination Planet (1 year)')
        plt.gca().set_aspect('equal',adjustable='box')
        if plot == True:
            plt.show()

        return rocket_pos_plan


    def stable_orbit_cheat(self, picturetime=None):
        codes = [79374]
        shortcuts = SpaceMissionShortcuts(mission, codes)
        shortcuts.place_spacecraft_in_stable_orbit(108.8, 140000, 0, 1)
        landing = mission.begin_landing_sequence()

        N = 500
        fall_time = 5
        position = np.zeros((N, 3))
        velocity = np.zeros((N, 3))
        time = np.zeros((N, 1))

        # landing.start_video()
        # landing.look_in_direction_of_motion()
        # landing.look_in_direction_of_planet()

        time[0], position[0,:], velocity[0,:] = landing.orient()

        for i in range(1, N-1):
            landing.fall(fall_time)
            time[i], position[i,:], velocity[i,:] = landing.orient()

            # Dette er bare for å finne et sted å lande
            # if time[i] >= picturetime:
            #     landing.start_video()
            #     landing.look_in_direction_of_planet()
            #     landing.fall(100)
            #     landing.finish_video(filename="pos_land_site.xml", number_of_frames=500)
            #     break

        # landing.finish_video(filename="close_orbit.xml", number_of_frames=500)
        return time, position


    def moving_coordinates(self, landing_time, delta_t, phi_0=0, theta_0=np.pi/2):
        omega = 2*np.pi/(system.rotational_periods[1]*const.day) #vinkelhastighet

        time, position = self.stable_orbit_cheat()

        landing_index = np.where(time == landing_time) #Finner indexen i tidsarrayen der tiden er lik tidspunktet vi sender ut landeren
        rocket_landing_position = position[landing_index,:][0][0]

        X = rocket_landing_position[0]
        Y = rocket_landing_position[1]

        phi_start = np.arctan(Y/X)
        phi_landing = phi_start + omega*delta_t

        theta = np.pi/2 #Konstant
        new_X = system.radii[1]*1000*np.sin(theta)*np.cos(phi_landing)
        new_Y = system.radii[1]*1000*np.sin(theta)*np.sin(phi_landing)


class Atmosphere_profile:
    def __init__(self, surface_temperature, planet_radius, surface_density, gamma):
        self.T0 = surface_temperature
        self.R = planet_radius*1000
        self.rho_0 = surface_density
        self.Y = gamma
        u = 1.66053886e-27
        self.mass_list = [18.05*u, 44.01*u, 44.01*u] #Kg

    def mean_molecular_weight(self):
        #Må kalles på før noe annet
        N = 3 #Har tre typer molekyler
        frac = 1/N
        self.mu = 0
        for i in range(N):
            self.mu += frac*self.mass_list[i]/const.m_H2

    def find_temp_and_density(self, N):
        mu = self.mu
        gamma = self.Y
        T0 = self.T0
        rho_0 = self.rho_0
        P_0 = rho_0*T0*const.k_B/(mu*const.m_H2)
        C = P_0**(1-gamma)*(T0**gamma)
        M = system.masses[1]*const.m_sun

        dr = 1

        r = np.zeros(N)
        g = np.zeros(N)
        P = np.zeros(N)
        T = np.zeros(N)
        rho = np.zeros(N)

        r[0] = self.R
        g[0] = M*const.G/(r[0]**2)
        T[0] = T0
        rho[0] = rho_0
        P[0] = P_0

        for i in range(N-1):
            g[i+1] = M*const.G/(r[i]**2)
            dP = -rho[i]*g[i]
            P[i+1] = P[i] + dP
            T[i+1] = (C*P[i+1]**(gamma-1))**(1/gamma)
            rho[i+1] = P[i+1]/(T[i+1]*const.k_B/(mu*const.m_H2))
            r[i+1] = r[i] + dr

            if T[i+1] < T0/2:
                g[i+1] = M*const.G/(r[i]**2)
                dP = -rho[i]*g[i]
                P[i+1] = P[i] + dP
                T[i+1] = T0/2
                rho[i+1] = P[i+1]/(T[i+1]*const.k_B/(mu*const.m_H2))
                r[i+1] = r[i] + dr

                # if P[i] < P_0/100:
                #     break

                if r[i+1] > 200000 + system.radii[1]*1000:
                    print(r[i+1])
                    break

        T = np.trim_zeros(T, 'b')/T0 #Fjerner overflødige nuller
        rho = np.trim_zeros(rho, 'b')/rho_0
        P = np.trim_zeros(P, 'b')/P_0
        r = np.trim_zeros(r, 'b')
        #print(r-system.radii[1]*1000)
        np.save('rho.npy', rho)
        np.save('rho_position.npy', r-system.radii[1]*1000)
        np.save('pressure.npy', P)

        return T, rho, P, r


    def visual_profile(self, N):
        T, rho, P, r = self.find_temp_and_density(N)

        x_isothermal = [6393035, 6393035]
        y_isothermal = [0, 1]
        plt.title("Atmospheric profile")
        plt.xlabel("Distance from planet surface")
        plt.plot(r, T, label=f"Temperature ($T/T_0$)", color='hotpink')
        plt.plot(r, rho, label=f"Density ($ ρ /  ρ_0$)", color='yellowgreen')
        plt.plot(r, P, label=f"Pressure ($P/P_0$)", color='steelblue')
        plt.plot(x_isothermal, y_isothermal, linestyle='dotted', color='k')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    spec_lin = SpectralLines()
    #Bare kjør dette for å gjøre om txt-filer til arrayer:
    # spec_lin.read_file('spectrum_seed09_600nm_3000nm.txt', 'spectral_lines', 'flux_observed')
    # spec_lin.read_file('sigma_noise.txt', 'sigma_noise')
    # spec_lin.find_every_line()
    # spec_lin.plot_spectral_lines()
    # spec_lin.spectral_line_info()
    # spec_lin.plot_all()
    #
    # entr_orb = Enter_orbit()
    # entr_orb.unstable_orbit()
    # entr_orb.change_frame_of_reference()
    # entr_orb.orbit_info()
    # entr_orb.stable_orbit_cheat(1500)
    # entr_orb.spherical_coordinates()
    # entr_orb.moving_coordinates(1500, 500)

    atm_pr = Atmosphere_profile(surface_temperature=271, planet_radius=system.radii[1], surface_density=system.atmospheric_densities[1], gamma=1.4)
    atm_pr.mean_molecular_weight()
    atm_pr.find_temp_and_density(1000000)
    #atm_pr.visual_profile(100000)
