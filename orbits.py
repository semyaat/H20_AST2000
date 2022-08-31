import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sc
from scipy import interpolate
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
from ast2000tools.shortcuts import SpaceMissionShortcuts

random.seed(None)
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)
mission = SpaceMission(seed)

class Orbits:
    def __init__(self, G, star_mass, number_of_planets, initial_positions, initial_velocities, planet_masses, rotational_periods, semi_major_axis):
        self.G = G
        self.star_mass = star_mass
        self.num_plan = number_of_planets
        self.init_pos = initial_positions
        self.init_vel = initial_velocities
        self.plan_mass = planet_masses
        self.rot_per = rotational_periods   #Periodene til alle planetene rundt sin egen akse
        self.semi_axis = semi_major_axis    #Store halvakse til alle planetene

    #Egen kode
    def integrate(self, t_end, N, show, transposed):
        #Dette er posisjon som funksjon av tid
        G = self.G
        star_mass = self.star_mass
        num_plan = self.num_plan    #Antall planeter i solsystemet vårt
        init_pos = self.init_pos    #Startposisjonene til alle planetene
        init_vel = self.init_vel    #Starthastighetene til alle planetene
        plan_mass = self.plan_mass  #Massene til alle planetene
        semi_axis = self.semi_axis  #Store halvakse til alle planetene

        #Lager en array med initalbetingelsene
        r0 = np.zeros((num_plan,2))
        v0 = np.zeros((num_plan,2))

        for i in range(num_plan):
            r0[i] = np.array(init_pos[:,i])
            v0[i] = np.array(init_vel[:,i])

        P = np.sqrt(((semi_axis[0]**3)*4*np.pi**2)/(G*(star_mass))) #Tar utgangspunkt i perioden til planet 0
        r = np.zeros((N*t_end,num_plan,2),float) #iterasjoner X antall planeter X antall kolonner (x,y)
        v = np.zeros((N*t_end,num_plan,2),float)
        t = np.linspace(0, P*t_end, N*t_end)
        dt = t[1] - t[0]

        r[0] = r0
        v[0] = v0

        a_i = np.transpose(-(G*(star_mass)/np.linalg.norm(r[0,:,:],axis=1)**3)*np.transpose(r[0,:,:]))
        for i in range(N*t_end-1):    #Gjør utregningene
            r[i+1,:,:] = r[i,:,:] + v[i,:,:]*dt + 0.5*a_i*dt**2
            a_ii = np.transpose(-(G*(star_mass)/np.linalg.norm(r[i+1,:,:],axis=1)**3)*np.transpose(r[i+1,:,:]))
            v[i+1,:,:] = v[i,:,:] + 0.5*(a_i + a_ii)*dt
            a_i = a_ii

        #Kommenterer ut for videre bruk i general launch
        if show == True:
            for planet in range(num_plan):
                plt.plot(r[:-1,planet,0],r[:-1,planet,1])

            plt.gca().set_aspect('equal',adjustable='box')
            plt.legend()
            plt.show()

        self.transposed_positions = np.transpose(r)
        self.P = P
        self.t = t
        self.r = r
        self.v = v
        self.dt = dt

        if transposed == False:
            return self.t, self.r, self.v
        if transposed == True:
            return self.t, self.r, self.v, self.transposed_positions, self.P


    #Egen kode
    def assert_kepler(self, t_start1, t_stop1, t_start2, t_stop2, planet):
        #Kan sjekke for hvilken som helst planet, men ønsker å sjekke vår. Bruk da planet=0.
        epsilon = 0.1
        dA = np.zeros(2) #Lager en tom array til utregning av areal
        sum_v = 0
        arc_length = 0

        #Regner ut arealet som sveipes ut over to tidspunkter for å sjekke om de er like
        for i in range(t_start1,t_stop1):
            P = np.sqrt((4*np.pi/(const.G_sol*(self.star_mass + self.plan_mass[planet]))))*self.semi_axis[planet]**3 #Bruker Keplers tredje lov for å finne perioden
            dA[0] += 0.5*((np.linalg.norm(self.r[i,planet])**2)*2*np.pi/P)*(self.t[i+1]-self.t[i])

        for i in range(t_start2,t_stop2):
            P = np.sqrt((4*np.pi/(const.G_sol*(self.star_mass + self.plan_mass[planet]))))*self.semi_axis[planet]**3
            dA[1] += 0.5*((np.linalg.norm(self.r[i,planet])**2)*2*np.pi/P)*(self.t[i+1]-self.t[i])

        if abs(dA[1]-dA[0]) < epsilon:
            print("The areas swept out are equal")
        else:
            print("The areas swept out are not equal")


        for i in range(t_start1,t_stop1):
            #Brukes til å regne midlere hastighet
            sum_v += np.linalg.norm(self.v[i,planet])

            #Regner ut avstanden de har beveget seg, antar at buelengden tilsvarer høyden i en rettvinklet trekant når tidsstegene blir små
            arc_length += (2*np.pi/self.rot_per[planet])*(self.t[i+1]-self.t[i])

        #Regner ut midlere hastighet
        midl_v = sum_v/t_stop1
        tot_time = self.dt*t_stop1 #Ganger opp tidssteget i v-arrayen med antall utregninger for å finne ut hvor langt planeten har beveget seg.
        print(f"The mean velocity from t=0 years to t={0.2*tot_time} years is {midl_v:.2f} AU/year, and the planet covered a distance of {arc_length:.2f} AU over a time period of {tot_time} years")

        period = np.zeros(8)
        #Sjekker om Keplers tredje lov holder
        for i in range(8):
            if abs(np.sqrt(self.semi_axis[i]**3) - np.sqrt((4*np.pi/(const.G_sol*(self.star_mass + self.plan_mass[i]))))*self.semi_axis[i]**3) < epsilon:
                print("Keplers third law checks out for planet", i)
            else:
                period[i] = np.sqrt((4*np.pi/(const.G_sol*(self.star_mass + self.plan_mass[i]))))*self.semi_axis[i]**3
                #print(f"Keplers third law does not hold up for planet {i}, and the difference in P is {abs(np.sqrt(self.semi_axis[i]**3) - np.sqrt((4*np.pi/(const.G_sol*(self.star_mass + self.plan_mass[i]))))*self.semi_axis[i]**3):.5} years")
                print(f"The period of planet {i} is {period[i]:.2f}") #Bruker bare denne til bloggen

    #Egen kode
    def find_analytical_orbit(self,N_points,a,e):
        angles = np.linspace(0,2*np.pi,N_points)
        r_pos = a*(1-e**2)/(1+e*np.cos(angles))

        self.x_analytic = r_pos*np.cos(angles)
        self.y_analytic = r_pos*np.sin(angles)

    #Egen kode
    def plot_analytical_orbit(self):
        #plt.figure(figsize=(10,10))
        plt.title("Noon Universe")
        plt.xlabel("Distance in AU")
        plt.ylabel("Distance in AU")
        for i in range(system.number_of_planets):
            calc_orb.find_analytical_orbit(1000,system.semi_major_axes[i], system.eccentricities[i])
            # plt.plot(self.x_analytic, self.y_analytic, label=f'{system.types[i]}',linestyle="dashed") #Kommenterer ut denne pga del 5
            plt.plot(self.x_analytic, self.y_analytic, linestyle="dashed")
        plt.gca().set_aspect('equal',adjustable='box')
        #plt.legend()
        #plt.savefig('filename.png', dpi=300)
        #plt.show()


    def interpolate(self, year):
        position_function = interpolate.interp1d(self.t, self.r, axis = 0, bounds_error = False, fill_value = 'extrapolate')
        velocity_function = interpolate.interp1d(self.t, self.v, axis = 0, bounds_error = False, fill_value = 'extrapolate')

        self.vel = velocity_function(year)
        self.pos = position_function(year)

        return self.vel, self.pos, self.t


calc_orb = Orbits(const.G_sol, system.star_mass, system.number_of_planets, system.initial_positions, system.initial_velocities, system.masses, system.rotational_periods, system.semi_major_axes)

#Integrerer posisjon og plotter mot den analytiske løsningen
#calc_orb.plot_analytical_orbit()
# t, planet_positions, planet_velocities = calc_orb.integrate(20, 10000, False, False)

'''
#Bruker dette i del 5:
t, planet_positions, planet_velocities = calc_orb.integrate(20, 10000, False, False)
np.save('planet_positions.npy', planet_positions)
np.save('planet_velocities.npy', planet_velocities)
np.save('planet_times.npy', t)
'''


# calc_orb.assert_kepler(0, 1000, 2000, 3000, 0)
#vel, pos = calc_orb.interpolate(1)

if __name__ == '__main__':
    #Bruker bare denne til å lagre filen med posisjoner
    codes = [71466]
    shortcuts = SpaceMissionShortcuts(mission, codes) #Bruker denne for å hente ut riktige planetbaner
    t, planet_positions, planet_velocities, transposed_positions, P = calc_orb.integrate(21, 10000, False, True)
    planet_positions_shortcut = shortcuts.compute_planet_trajectories(t)
    mission.verify_planet_positions(21*P, planet_positions_shortcut)

    SpaceMission.save('part3.bin', mission)


# mission.verify_planet_positions(20*P, transposed_positions)
#mission.generate_orbit_video(t, transposed_positions, filename='orbit_video.xml')



















'''
#Lager bevegelig plot:
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = plt.subplot(1,1,1)

data_skip = 50

def init_func():
    ax.clear()
    plt.xlabel("Avstand")
    plt.ylabel("Avstand")

def update_plot(i):
    #ax.scatter(planet_positions[i,0,0], planet_positions[i,0,1], marker='o', color='r')
    ax.plot(planet_positions[i:i+data_skip,0,0], planet_positions[i:i+data_skip,0,1], color='r')
    ax.plot(planet_positions[i:i+data_skip,1,0], planet_positions[i:i+data_skip,1,1], color='b')
    plt.xlim((-60,60))
    plt.ylim((-60,60))

calc_orb.plot_analytical_orbit()
anim = FuncAnimation(fig, update_plot, frames=np.arange(0,200000,data_skip), init_func=init_func, interval = 1)
plt.show()
'''
