import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit
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


class Alien_orbits:
    def __init__(self, G, star_mass, star_radii, plan_radii, number_of_planets, initial_positions, initial_velocities, planet_masses, rotational_periods, semi_major_axis, eccentricities):
        self.G = G
        self.star_mass = star_mass
        self.star_radii = star_radii
        self.plan_radii = plan_radii
        self.num_plan = number_of_planets
        self.init_pos = initial_positions
        self.init_vel = initial_velocities
        self.plan_mass = planet_masses
        self.rot_per = rotational_periods   #Periodene til alle planetene
        self.semi_axis = semi_major_axis    #Store halvakse til alle planetene
        self.ecc = eccentricities
        
     
    #Egen kode
    def change_frame_of_reference(self, N, t_end):
        G = self.G
        star_mass = self.star_mass
        init_pos = self.init_pos[:,6]    #Startposisjonen til planeten
        init_vel = self.init_vel[:,6]    #Starthastigheten til planeten
        plan_mass = self.plan_mass[6]    #Massene til planeten
        
        #Endrer startposisjonene til sola og planeten så CM er i origo
        r0_sol = -plan_mass*init_pos/(star_mass + plan_mass) #R-vektor i negativ retning, se notater
        r0_planet = init_pos + r0_sol #Opprinnelig posisjonsvektor minus R-vektor
        
        v0_sol = 0
        v0_planet = init_vel 
        
        a0_sol = (G*(plan_mass)/np.linalg.norm(r0_planet-r0_sol)**3)*(r0_planet-r0_sol)
        a0_planet = (G*(star_mass)/np.linalg.norm(r0_sol-r0_planet)**3)*(r0_sol-r0_planet)
        
        P = np.sqrt(((system.semi_major_axes[6]**3)*4*np.pi**2)/(G*(star_mass))) #Regner ut perioden til planet 6
        
        r = np.zeros((N*t_end,2,2),float) #N iterasjoner, 2 legemer, 2 koordinater
        v = np.zeros((N*t_end,2,2),float)
        a = np.zeros((N*t_end,2,2),float)
        t = np.linspace(0, P*t_end, N*t_end)
        dt = t[1] - t[0]
        
        r[0,0,:] = r0_sol
        r[0,1,:] = r0_planet
        v[0,0,:] = v0_sol
        v[0,1,:] = v0_planet
        a[0,0,:] = a0_sol
        a[0,1,:] = a0_planet
        
        for i in range(N*t_end-1):
            r_rel = r[i,1,:] - r[i,0,:] #Vektor fra sol til planet
            t[i+1] = t[i] + dt
            a[i,0,:] = (G*(plan_mass)/np.linalg.norm(r_rel)**3)*(r_rel) #Oppdaterer kreftene på sola, har delt på planetmasse
            a[i,1,:] = (G*(star_mass)/np.linalg.norm(-r_rel)**3)*(-r_rel) #Oppdaterer kreftene på planet, har delt på solmasse
            r[i+1,:,:] = r[i,:,:] + v[i,:,:]*dt + 0.5*a[i,:,:]*dt**2
            r_rel = r[i+1,1,:] - r[i+1,0,:]
            a[i+1,0,:] = (G*(plan_mass)/np.linalg.norm(r_rel)**3)*(r_rel) #Oppdaterer kreftene på sola, har delt på planetmasse
            a[i+1,1,:] = (G*(star_mass)/np.linalg.norm(-r_rel)**3)*(-r_rel) #Oppdaterer kreftene på planet, har delt på solmasse
            v[i+1,:,:] = v[i,:,:] + 0.5*(a[i,:,:] + a[i+1,:,:])*dt
            
           
        planet_names = np.array(["Sun", "Planet"])    
        for i in range(2):
            plt.plot(r[:,i,0],r[:,i,1], label=f"{planet_names[i]}")
         
         
        plt.title("Sun and planet orbiting a common center of mass") 
        plt.xlabel("Distance [AU]")
        plt.ylabel("Distance [AU]")
        plt.legend()
        plt.gca().set_aspect('equal',adjustable='box')
        plt.show()
            
        self.t = t
        self.r = r
        self.v = v
        self.t_end = t_end
        self.P = P
        return self.t, self.r, self.v
    
    #Egen kode
    def add_more_planets(self, N, t_end):
        #Bruker planet nr #2,5,6
        G = self.G
        P = self.P #Perioden til planet nr 6
        star_mass = self.star_mass
        init_pos = np.array([self.init_pos[:,2], self.init_pos[:,5], self.init_pos[:,6]])   #Startposisjonen til planetene
        init_vel = np.array([self.init_vel[:,2], self.init_vel[:,5], self.init_vel[:,6]])  #Starthastigheten til planetene
        plan_mass = np.array([self.plan_mass[2], self.plan_mass[5], self.plan_mass[6]])    #Massene til planetene

        #Lager store R-vektor
        M = star_mass #Setter initialbetingelse til stjernemassen, da denne skal legges til, men ønsker å gjøre dette utenfor for-løkken
        R = 0
        for i in range(3):
            M += plan_mass[i] 
            R += plan_mass[i]*init_pos[i]
        R = R/M #Vektoren fra origo til massesenteret
        
        #Lager en array med initalbetingelsene
        r0 = np.zeros((4,2))
        v0 = np.zeros((4,2))
        a0 = np.zeros((4,2))
        
        #Setter initialbetingelsene til sola
        r0[0] = -R
        v0[0] = 0
        a0[0] = (G*(M/star_mass)/np.linalg.norm(R)**3)*R
        
        #Setter initialbetingelsene til planetene
        for i in range(1,3):
            r0[i] = np.array(init_pos[i] - R)
            v0[i] = np.array(init_vel[i])
            a0[i] = -(G*(star_mass)/np.linalg.norm(r0[i])**3)*r0[i]
        
        #N iterasjoner, 4 legemer (sol + tre planeter), 2 koordinater
        r = np.zeros((N*t_end,4,2),float) 
        v = np.zeros((N*t_end,4,2),float)
        a = np.zeros((N*t_end,4,2),float)
        t = np.linspace(0, P*t_end, N*t_end)
        dt = t[1] - t[0]
 
        r[0,0,:] = r0[0]
        r[0,1:4,:] = r0[1:4]
        v[0,0,:] = v0[0]
        v[0,1:4,:] = v0[1:4]
        a[0,0,:] = a0[0]
        a[0,1:4,:] = a0[1:4]
        
        for i in range(N*t_end-1):
            for j in range(1,3):
                r_rel = r[i,j,:] - r[i,0,:] #Vektor fra alle planetene til sola
                t[i+1] = t[i] + dt
                a[i,0,:] = G*sum(plan_mass)/np.linalg.norm(r_rel)*r_rel #Sol
                a[i,j,:] += (G*star_mass)/(np.linalg.norm(-r_rel)**3)*(-r_rel) #Planetene
                r[i+1,:,:] = r[i,:,:] + v[i,:,:]*dt + 0.5*a[i,:,:]*dt**2
                r_rel = r[i+1,j,:] - r[i+1,0,:]
                a[i+1,0,:] = G*sum(plan_mass)/np.linalg.norm(r_rel)*r_rel
                a[i+1,j,:] = (G*star_mass)/(np.linalg.norm(-r_rel)**3)*(-r_rel)
                v[i+1,:,:] = v[i,:,:] + 0.5*(a[i,:,:] + a[i+1,:,:])*dt
         
        self.t_2 = t
        self.r_2 = r
        self.v_2 = v
        return self.t_2, self.r_2, self.v_2
        
    #Egen kode
    def radial_velocity_curve(self, t_end, N, v, t, num_plan, new_title, least_squares_plot):
        #Lager radial velocity curve for sola
        v_max = np.max(v[0:N,0,0]) #Maksimalt utslag av vx
        my = 0
        sigma = 0.2*v_max  
        gauss_noise = np.random.normal(my,sigma,N) 
        signal = v[0:N,0,0] + gauss_noise #Ser på hastigheten i x-retning
        
        plt.plot(t[0:N],signal)          #Plotter tid mot kurven + støy
        plt.plot(t[0:N], v[0:N,0,0])     #Plotter tid mot selve kurven
        
        if new_title == True:
            plt.title(f"Radial velocity with {num_plan} planets")
        else:
            plt.title("Radial velocity")
            
            
        plt.xlabel("Time [years]")
        plt.ylabel("Velocity in x-direction [AU/year]")
        
        if least_squares_plot == False: #Ønsker ikke å plotte her hvis vi bruker least_squares
            plt.show()
        
        self.signal = signal
        return self.signal       
    
    #Egen kode
    def check_analytic(self, N_points, plot_or_not, epsilon):
        mass_p = self.plan_mass[6]     #Massen til planeten
        mass_s = self.star_mass        #Massen til sola
        p_ax = self.semi_axis[6]       #Store halvakse til planeten
        ecc = self.ecc[6]              #Eksentrisiteten til planeten
        v_s = self.v[:,0,:]            #Hastigheten til sola
        v_p = self.v[:,1,:]            #Hastigheten til planeten
        r_s = self.r[:,0,:]            #Posisjonen til sola
        r_p = self.r[:,1,:]            #Posisjonen planeten
        
        my = (mass_p*mass_s)/(mass_p + mass_s)
        a_s = my*p_ax/mass_s
        a_p = my*p_ax/mass_p
        angles = np.linspace(0, 2.*np.pi, N_points)
        r_sol = a_s*(1 - ecc**2)/(1 + ecc*np.cos(angles))
        r_planet = a_p*(1 - ecc**2)/(1 + ecc*np.cos(angles))
        
        #Setter koordinatene:
        self.x1_analytic = r_sol*np.cos(angles)
        self.y1_analytic = r_sol*np.sin(angles)
        self.x2_analytic = r_planet*np.cos(angles)
        self.y2_analytic = r_planet*np.sin(angles)
        
        if plot_or_not == True:
            self.plot_analytical_orbit()
   
        #Store R-vektor
        R = r_p - r_s 
        E1 = 0.5*mass_s*np.linalg.norm(v_s[0])**2 + 0.5*mass_p*np.linalg.norm(v_p[0])**2 - const.G*mass_s*mass_p/np.linalg.norm(R[0])
        E2 = 0.5*mass_s*np.linalg.norm(v_s[-1])**2 + 0.5*mass_p*np.linalg.norm(v_p[-1])**2 - const.G*mass_s*mass_p/np.linalg.norm(R[-1])
        
        if abs(E1 - E2) < epsilon:
            print("The energy is conserved")
        else:
            print("The energy is not conserved, and the difference in energy is", abs(E1 - E2) )
        
    #Egen kode   
    def plot_analytical_orbit(self):
        plt.title("Analytic solution")
        plt.xlabel("Distance in AU")
        plt.ylabel("Distance in AU")
        #alien_orb.check_analytic(1000)
        plt.plot(self.x1_analytic, self.y1_analytic, label='Sun', linestyle="dashed")
        plt.plot(self.x2_analytic, self.y2_analytic, label='Planet', linestyle="dashed")
        plt.gca().set_aspect('equal',adjustable='box')
        plt.legend()
        #plt.savefig('filename.png', dpi=300)
        plt.show()
        
    #Egen kode
    def least_squares(self, v_i, v_r, P, t0):
        N = len(v_i)
        print("Lengde av v_i", N)
        v_mod = np.zeros(N)
        print("Lengden av v modellert",len(v_mod))

        for i in range(N):
            v_mod[i] = v_r*np.cos((2*np.pi/P)*(self.t[i] - t0))
        
        delta_v = 0
        for i in range(N):
            delta_v += (v_i[i] - v_mod[i])**2
        

        plt.plot(self.t[:N], v_mod)
        plt.title(f"Approximated radial velocity, \n $v_*$ = {v_r}, P = {P}, $t_0$ = {t0} \n $\Delta$ = {delta_v}", fontsize=18)
        plt.xlabel("Time [years]")
        plt.ylabel("Velocity in x-direction [AU/year]")
        plt.ylim(-0.0005,0.0006)
        plt.show()
            
        print("Delta v er gitt ved", delta_v)
        
    #Kjører ikke    
    def light_curve(self):
        r_s = self.star_radii #Radius i km
        r_p = self.plan_radii #Radius i km

        period = np.sqrt((self.semi_axis[6]**3)*4*np.pi**2/(const.G_sol*(self.plan_mass[6] + self.star_mass)))
        midl_dist = const.AU #Setter noe sånn ca
        orb_speed = 2*np.pi*midl_dist/period
        
        r0s = r_p         #Setter en tilfeldig verdi, her starter planeten å bevege seg inn foran sola
        r1s = r0s + 2*r_s #Her har planeten beveget seg ut av sola
        dt = 0.1
        r = np.arange(0, 2*r0s + 2*r_s, dt)
        N = len(r)
        A_s = np.zeros(N)
        A_p = np.zeros(N)
        Fluks = np.zeros(N)
        Fluks[0] = 1
        
        A_s = (2*r_s)**2 #Arealet til sol
     
        
        '''
        #Setter fluksen til 1 når planeten er utenfor sola
        fluks_init = np.logical_and(np.less(r, r0s), np.greater(r,r1s))
        fluks_init_indices = np.where(fluks_init == True)
        A_p[fluks_init_indices] = 0
        Fluks[fluks_init_indices] = 1
        print(Fluks)
        
        #Planetene er på vei inn
        fluks_var1 = np.logical_and(np.greater(r, r0s), np.less(r, r0s + 2*r_p))
        fluks_var1_indices = np.where(fluks_var1 == True)
        A_p[fluks_var1_indices] = (r[fluks_var1_indices] - r0s)*2*r_p
        Fluks[fluks_var1_indices] = (A_s - A_p[fluks_var1_indices])/A_s
                                  
        #Planeten er foran sola
        fluks_const = np.logical_and(np.greater(r, r0s + 2*r_p), np.less(r, r1s - 2*r_p))
        fluks_const_indices = np.where(fluks_const == True)
        Fluks[fluks_const_indices] = A_p[fluks_var1_indices][-1]
        
        #Planetene er på vei ut
        fluks_var2 = np.logical_and(np.greater(r, r1s - 2*r_p), np.less(r, r1s))
        fluks_var2_indices = np.where(fluks_var2 == True)
        A_p[fluks_var2_indices] = (r[fluks_var2_indices] - r0s)*2*r_p
        Fluks[fluks_var2_indices] = (A_s + A_p[fluks_var2_indices])/A_s
        
        print(Fluks)
        '''
        
    
    
alien_orb = Alien_orbits(const.G_sol, system.star_mass, system.star_radius, system.radii[6], system.number_of_planets, system.initial_positions, system.initial_velocities, system.masses, system.rotational_periods, system.semi_major_axes, system.eccentricities)
t, r, v = alien_orb.change_frame_of_reference(10000, 20) #OBS! N og N_points må være like
alien_orb.check_analytic(10000, False, 0.000001)
# alien_orb.plot_analytical_orbit()

#Lager radial velocity curve med en planet:
t, r, v = alien_orb.change_frame_of_reference(10000, 70) #Har denne for å sjekke least_squares bare
# signal = alien_orb.radial_velocity_curve(70, 10000, v, t, "1", False, False)

#Sjekk for ulike verdier:
#signal = alien_orb.radial_velocity_curve(70, 10000, v, t, "1", False, True)
#alien_orb.least_squares(signal, 0.0004, 50, 12) 
#alien_orb.least_squares(signal, 0.00035, 55, 12) 
#alien_orb.least_squares(signal, 0.0003166, 52.1, 13.2) 

#Lager radial velocity curve med flere planeter:
t_2, r_2, v_2 = alien_orb.add_more_planets(10000,70)
#radial_velocity_curve(self, t_end, N, v, t, num_plan, new_title, least_squares_plot) FJERN DETTE
alien_orb.radial_velocity_curve(70,10000, v_2, t_2, "three", True, False)
