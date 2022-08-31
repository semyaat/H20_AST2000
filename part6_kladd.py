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
mission = SpaceMission.load('part5.bin') #Bruker denne verify_manual_orientation

def unstable_orbit():
    #HER MÅ DET RYDDES MYE
    codes = [97520]
    shortcuts = SpaceMissionShortcuts(mission, codes)

    planet_idx = 1 #Destinasjonsplanet
    time = 108.8    #Tiden vi vil være fremme i år
    shortcuts.place_spacecraft_in_unstable_orbit(time, planet_idx)
    travel = mission.ongoing_interplanetary_travel


    #Lagrer posisjonen når vi er i bane rundt destinasjonsplaneten
    N = 250 #250 utregninger er ca en periode
    coast_time = 0.001
    # position = np.zeros((N, 2))
    # velocity = np.zeros((N, 2))
    # time = np.zeros((N, 1))

    position = np.zeros((N+200, 2))
    velocity = np.zeros((N+200, 2))
    time = np.zeros((N+200, 1))

    # travel.start_video()
    # travel.look_in_direction_of_planet(1)

    time[0], position[0,:], velocity[0,:] = travel.orient()
    travel.coast(0.0001)
    travel.boost(np.array([0.015*velocity[0,0], -0.025*velocity[0,1]]))


    for i in range(1, N-1):
        travel.coast(coast_time)
        time[i], position[i,:], velocity[i,:] = travel.orient()

        #Finner periapsis:
        if np.linalg.norm(position[i-1,:], axis=0) < np.linalg.norm(position[i,:], axis=0):
            travel.boost(np.array([0,-0.00005*velocity[i,1]]))

        # if np.linalg.norm(position[i,:], axis=0) < np.linalg.norm(position[i-1,:], axis=0):
        #     travel.boost(0.00005*velocity[i,:])

    for i in range(N-1, N+199):
        travel.coast(coast_time)
        time[i], position[i,:], velocity[i,:] = travel.orient()

    #travel.finish_video(filename='orbit_TEST.xml', number_of_frames=500)

    np.save('position_in_orbit_TEST.npy', position)
    np.save('velocity_in_orbit_TEST.npy', velocity)
    np.save('times_in_orbit_TEST.npy', time)

def change_frame_of_reference(plot=True):
    #BRUKER DENNE FOR Å PLOTTE BANEN
    rocket_position = np.load('position_in_orbit_TEST.npy') #Posisjonen til raketten ift sola
    planet_times_orbit = np.load('times_in_orbit_TEST.npy') #Tiden raketten er i orbit

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

    return rocket_pos_plan

def orbit_info():
    #BRUKER BARE DENNE FOR Å SJEKKE EKSENTRISITETEN
    rocket_pos_plan = change_frame_of_reference(plot=False)
    rocket_vel_plan = np.load('velocity_in_orbit_TEST.npy')
    rocket_times_plan = np.load('times_in_orbit_TEST.npy')

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
    print("DETTE ER E", eccentricity)
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

# unstable_orbit()
#change_frame_of_reference()
# orbit_info()


def read_file():
    with open('spectrum_seed09_600nm_3000nm.txt') as f:
        spectrum_list = [line.split() for line in f]
        spectral_lines = []
        flux_observed = []
        for i in range(len(spectrum_list)):
            spectral_lines.append(float(spectrum_list[i][0]))
            flux_observed.append(float(spectrum_list[i][1]))

    np.save('spectral_lines.npy', np.array(spectral_lines))
    np.save('flux_observed.npy', np.array(flux_observed))

    with open('sigma_noise.txt') as g:
        sigma_list = [line.split() for line in g]
        spectral_lines_sigma = []
        sigma_noise = []
        for i in range(len(sigma_list)):
            spectral_lines_sigma.append(float(sigma_list[i][0]))
            sigma_noise.append(float(sigma_list[i][1]))

    np.save('spectral_lines_sigma.npy', np.array(spectral_lines_sigma)) #BRUKER IKKE DENNE
    np.save('sigma_noise.npy', np.array(sigma_noise))



def flux_mod(Fmin, lambda0, sigma):
    lambda1 = np.load('spectral_lines.npy')
    delta_lambda = (10/const.c_km_pr_s)*lambda0 #km/s
    lambda_min = np.greater_equal(lambda1, lambda0-delta_lambda)
    lambda_max = np.less_equal(lambda1, lambda0+delta_lambda)
    lambda_range = np.logical_and(lambda_min, lambda_max)
    gas_index = np.where(lambda_range == True)
    lambda_sliced = lambda1[gas_index]

    f_mod = np.zeros(len(lambda_sliced))

    for i in range(len(lambda_sliced))
        f_mod[i] = 1 + (Fmin-1)*np.exp(-0.5*((lambda_sliced[i] - lambda0)/(sigma))**2)
    return f_mod

def least_chi_squares(Fmin, lambda0, sigma):
    flux_obs = np.load('flux_observed.npy')
    sigma_obs = np.load('sigma_noise.npy')
    # lambda1 = np.load('spectral_lines.npy')

    #Fjerne dette, putt i egen funksjon i klassen
    delta_lambda = (10/const.c_km_pr_s)*lambda0 #km/s
    lambda_min = np.greater_equal(lambda1, lambda0-delta_lambda)
    lambda_max = np.less_equal(lambda1, lambda0+delta_lambda)
    lambda_range = np.logical_and(lambda_min, lambda_max)
    gas_index = np.where(lambda_range == True)

    #Velger hvilke verdier vi skal teste
    #lambda_sliced = lambda1[gas_index]
    sigma_sliced = sigma_obs[gas_index] #TRENGER IKKE DENNE
    flux_sliced = flux_obs[gas_index] #TRENGER BARE TIL PLOTTING

    lambda0_values = np.linspace(lambda0 - delta_lambda, labda0 + delta_lambda, test_values)
    sigma_values = np.linspace(SIGMA MIN, SIGMA MAX, test_values)
    flux_values = np.linspace(FLUX MIN, FLUX MAX, test_values)
    FLUKS GÅR FRA 0.7 TIL 1
    SIGMA GÅR FRA (BRUK FORMEL MED TEMO FRA 150K TIL 450K)

    N = len(lambda_sliced) #Litt weird måte å gjøre ting på?? KANSKJE IKKE RIKTIG HELLER
    chi = np.zeros(N)

    for i in range(N): #Finn ut av N
        #flux_mod = 1 + (Fmin-1)*np.exp(-0.5*((lambda_sliced[i] - lambda0)/(sigma))**2)
        chi[i] += ((flux_sliced[i] - flux_mod(flux_values, lambda0_values, sigma_values))/sigma_values)**2

    least_value = np.amin(chi)
    least_value_index = np.argmin(chi)
    least_lambda = lambda_sliced[least_value_index]

    return least_value, least_lambda

def find_optimal_parameters(lambda0, test_values, mass):
    flux_min = np.amin(np.load('flux_observed.npy')) #IGNORER
    flux_max = np.amax(np.load('flux_observed.npy')) #IGNORER

    T = np.linspace(150, 450, test_values)
    sigma = (lambda0/const.c)*np.sqrt(const.k_B*T/mass)
    Fmin = np.linspace(flux_min, flux_max, test_values) #IGNORER

    least_value_list = []
    for f in Fmin:
        for s in sigma:
            least_val_element = [least_chi_squares(f, lambda0, s)[0], least_chi_squares(f, lambda0, s)[1], f, s]
            least_value_list.append(least_val_element)

    least_value_array = np.array(least_value_list)
    least_value_index = np.argmin(least_value_array[:,0])

    return least_value_array[least_value_index]


def test_all_lines():
    u = 1.66053886e-27 #Finn en bedre konstant?
    lambda0_list = [632, 690, 760, 720, 820, 940, 1400, 1600, 1660, 2200, 2340, 2870] #Meter
    mass_list = [15.999*u, 15.999*u, 15.999*u, 18.05*u, 18.05*u, 18.05*u, 44.01*u, 44.01*u, 16.04*u, 16.04*u, 58.93*u, 44.01*u] #Kg
    #lambda0_array = np.array([632, 690, 760, 720, 820, 940, 1400, 1600, 1660, 2200, 2340, 2870])
    #mass_array = np.array([15.999*u, 15.999*u, 15.999*u, 18.05*u, 18.05*u, 18.05*u, 44.01*u, 44.01*u, 16.04*u, 16.04*u, 58.93*u, 44.01*u]) #Kg
    parameter_array = np.zeros((len(lambda0_list), 4))

    i = 0
    for lambda0, mass in zip(lambda0_list, mass_list):
        parameter_array[i,:] = find_optimal_parameters(lambda0, 2, mass) #Lagrer minste forskjell, lambda, fluks, sigma
        i += 1

    # for i in range(12):
    #     parameter_array[i,:] = find_optimal_parameters(lambda0_array[i], 2, mass_array[i]) #Lagrer minste forskjell, lambda, fluks, sigma

    print(parameter_array)
    np.save('parameter_array.npy', parameter_array)

import time
start = time.time()

test_all_lines()

end = time.time()
print("Dette er tiden det tok lol", end - start)












# import time
# start = time.time()
#
# #KODE
#
# end = time.time()
# print("Dette er tiden det tok lol", end - start)
