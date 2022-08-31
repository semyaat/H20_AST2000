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
# shortcuts.place_spacecraft_in_stable_orbit(108.8, 140000, 0, 1)
shortcuts.place_spacecraft_in_stable_orbit(108.45, 140000, 0, 1)

rho_saved = np.load('rho.npy')
rho_height_saved = np.load('rho_position.npy')
rho_function = interpolate.interp1d(rho_height_saved, rho_saved, axis = 0, bounds_error = False, fill_value = 'extrapolate')

def rho(height):
    rho = rho_function(height)
    return rho


def landing_func(r0, v0, end_time, N):
    landing = mission.begin_landing_sequence()

    r = np.zeros((N,2))
    v = np.zeros((N,2))
    t = np.zeros((N,1))

    dt = end_time/N

    r[0,:] = r0
    v[0,:] = v0

    A = 0.3 #m^2
    M = 1100
    m = 90
    R = system.radii[1]*1000
    A_p = 150 #Parachute

    #omega = 2*np.pi*system.radii[1]*1000/(system.rotational_periods[1]*const.day)
    #v_drag = -(height*omega*np.array([-r_hat[1], r_hat[0]]) - v[i,:])


    for i in range(N-1):
        r_hat = r[i,:]/np.linalg.norm(r[i,:])
        height = np.linalg.norm(r[i,:])

        v_drag = -(height*np.array([-r_hat[1], r_hat[0]]) - v[i,:])
        F_d = 0.5*rho(height)*A*np.linalg.norm(v_drag)**2 * (-v_drag/np.linalg.norm(v_drag)) #Virker i fartsretning
        F_g = const.G*m*M/(np.linalg.norm(r[i,:])**2)*r_hat #Virker i radiell
        #print(F_g)
        F_tot = F_d - F_g
        #F_tot = -F_g

        a = F_tot/m
        v[i+1,:] = v[i,:] + a*dt
        r[i+1,:] = r[i,:] + v[i+1,:]*dt

        #print(np.linalg.norm(r[i+1,:]) - R)
        # if (np.linalg.norm(r[i+1,:]) - R) < 30:
        #         print(np.linalg.norm(v[i+1,:]))
        #         print("Dette er indeksen der den stopper", i)
        #         break

        v_t_update = True
        if (np.linalg.norm(r[i+1,:]) - R) < 500:
            r_hat = r[i,:]/np.linalg.norm(r[i,:])
            height = np.linalg.norm(r[i,:])

            if np.linalg.norm(F_tot) < 1 and v_t_update == True:
                v_t = np.sqrt(2*const.G*M*m/(R*(-rho(height))*A))
                v_t_update = False
            else:
                v_t = 0


            v_drag = -(height*np.array([-r_hat[1], r_hat[0]]) - v[i,:])
            F_d = 0.5*rho(height)*A*np.linalg.norm(v_drag)**2 * (-v_drag/np.linalg.norm(v_drag)) #Virker i fartsretning
            F_g = const.G*m*M/(np.linalg.norm(r[i,:])**2)*r_hat #Virker i radiell
            F_L = 0.5*rho(0)*A_p*(v_t - 3**2))
            F_tot = F_d + F_L - F_g

            a = F_tot/m
            v[i+1,:] = v[i,:] + a*dt
            r[i+1,:] = r[i,:] + v[i+1,:]*dt

            print(np.linalg.norm(r[i+1,:]) - R)

            if (np.linalg.norm(r[i+1,:]) - R) < 30:
                print(np.linalg.norm(v[i+1,:]))
                print("Dette er indeksen der den stopper", i)
                break


    theta = np.linspace(0, 2*np.pi, 100)
    radius = system.radii[1]*1000
    figure, axes = plt.subplots(1)
    axes.plot(radius*np.cos(theta), radius*np.sin(theta))

    plt.plot(r[0,0], r[0,1], "bo", markersize=3)
    plt.plot(r[:,0],r[:,1])
    plt.gca().set_aspect('equal',adjustable='box')
    plt.show()

landing = mission.begin_landing_sequence()
t, r0, v0 = landing.orient() #Dette tar utgangspunkt i at vi starter etter 108.8, må evt endres i snarveien
# landing_func(r0[:2], v0[:2], 100, 1000)
landing_func(r0[:2], v0[:2] + np.array([-2000,0]), 1000, 100000)



# N = 1260
#N = 20000
N = 1426
r = np.zeros((N,3))
v = np.zeros((N,3))
t = np.zeros((N,1))

landing = mission.begin_landing_sequence()
# landing.launch_lander(np.array([700,-1200,0]))
landing.launch_lander(np.array([700,-950,0]))

landing.adjust_landing_thruster(579.3813,500)
landing.adjust_parachute_area(350)

landing.start_video()
landing.look_in_direction_of_motion()

for i in range(N):
    landing.fall(1)
    t[i], r[i,:], v[i,:] = landing.orient()
    print(np.linalg.norm(r[i,:]) - system.radii[1]*1000)

landing.finish_video(filename="lander_VERYCLOSE.xml",number_of_frames=500,radial_camera_offset=3000)

theta = np.linspace(0, 2*np.pi, 100)
radius = system.radii[1]*1000
figure, axes = plt.subplots(1)
axes.plot(radius*np.cos(theta), radius*np.sin(theta))

plt.plot(r[0,0], r[0,1], "bo", markersize=3)
plt.plot(r[:,0], r[:,1])
plt.gca().set_aspect('equal',adjustable='box')
plt.show()
