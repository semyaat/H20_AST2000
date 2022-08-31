import numpy as np
import random
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
random.seed(None)
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)
mission = SpaceMission(seed)

#Snarvei:
codes = [87556]
shortcuts = SpaceMissionShortcuts(mission, codes)


class Calculations:
    #Bruker bare sigma herfra, se andre program for sannsynlighetsfordeling og integrasjon
    def __init__(self, my, temp, mass_par, k):
        self.my = my
        self.T = temp
        self.m = mass_par
        self.k = k

    def beregn_sigma(self):
        self.sigma = np.sqrt((self.k*self.T)/self.m)
        return self.sigma

    def prob_distr_gauss(self,x):
        #f(my,sigma,x), gaussisk distribusjon
        sigma = self.beregn_sigma()
        my = self.my
        self.gauss_distr = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((x-my)/sigma)**2)
        return self.gauss_distr

    def prob_distr_maxbo(self,x):
        #Obs! Gjelder kun for positive verdier av v, brukes gjerne til absoluttverdien av v-vektor
        sigma = self.beregn_sigma()
        my = self.my
        self.mb_distr = (4*np.pi*x**2)*((1/(np.sqrt(2*np.pi)*sigma))**2)*np.exp(-0.5*((x-my)/sigma)**2)
        return self.mb_distr

    def integrate(self,f,a,b,N):
        #Fungerer ikke
        tot = 0
        dx = (b-a)/N
        for n in range(N):
            tot += f[n]*dx
        self.tot = tot
        return self.tot

vx = np.linspace(-2.5,2.5,10000)
vel = Calculations(0,3000,const.m_H2,const.k_B)
P_1 = vel.prob_distr_gauss(vx)
sigma = vel.beregn_sigma()
#print(vel.integrate(P_1,-sigma,sigma,10000)) #Tester integrasjonen, den kjører, men gir feil verdier


class Rocket_launch(Calculations):
    def __init__(self, my,temp,mass_par,k,G,plan_mass,plan_radii,sat_mass,rotational_period,plan_vel,box_dim):
        Calculations.__init__(self,my,temp,mass_par,k) #Usikker på om dette er nødvendig? Bruker bare utregningen for sigma
        self.G = G
        self.plan_mass = plan_mass*const.m_sun      #Konverterer til kg
        self.plan_radii = plan_radii*1000           #Konverterer radien til meter
        self.sat_mass = sat_mass                    #Massen er gitt i kg
        self.rot_per = rotational_period*24*60*60   #Konverterer perioden til sekunder
        self.mass_par = mass_par                    #Partikkelmasse gitt i kg
        self.box_dim = box_dim                      #Lengden på sidene på boksen
        self.plan_vel = plan_vel*4740.57            #Initialhastigheten til planeten i y-retning, omregnet til m/s

    def moving_particles(self, N, ant_iter,box_dim):
        #N er antall partikler, ant_iter er antall utregninger
        sigma = self.beregn_sigma()
        m = self.mass_par

        #Initialbetingelser for posisjon og hastighet
        r0 = np.random.uniform(0,box_dim,size=(int(N),3))   #N antall rader/partikler, 3 kolonner. Tilfeldige posisjoner
        v = np.random.normal(0,sigma,size=(int(N),3))       #Array for hastigheter, normalfordelt

        #Tom array for å beregne nye posisjoner
        r = np.zeros((ant_iter,int(N),3)) #Antall utregninger, antall partikler, antall kolonner
        r[0] = r0                         #Setter initialbetingelse
        dt = 0.01                         #Tidssteg

        #Utregninger som skal gjøres
        momentum = 0          #Bevegelsmengde
        esc_par_count = 0     #Teller partikler som har sluppet unna for hver utregning
        esc_par = 0           #Totalt antall partikler som slipper unna

        for i in range(ant_iter-1):
            r[i+1,:,:] = r[i,:,:] + v[:,:]*dt #Burde være i+1?

            #Sjekker om partiklene har gått ut av hullet
            z_component = np.less_equal(r[i,:,2], 0)
            y_component = np.less_equal(r[i,:,1], 0.5*box_dim)
            x_component = np.less_equal(r[i,:,0], 0.5*box_dim)
            lost_par = np.logical_and(x_component,np.logical_and(z_component, y_component))
            new_par_indices = np.where(lost_par == True)

            #Alt som har å gjøre med partiklene når de har sluppet ut
            esc_par_count = np.count_nonzero(lost_par)  #Registrerer partikler som slipper ut
            r[i,new_par_indices] = np.random.uniform(0,box_dim,size=(int(esc_par_count),3)) #Setter inn nye partikler
            esc_par += esc_par_count    #Legger til i totalt antall partikler som slipper unna

            #Regner ut bevegelsesmengden
            momentum += np.sum(m*v[:,2])

            #Snur partikler som kolliderer
            collision_points = np.logical_or(r[i,:,:] < 0, r[i,:,:] > box_dim)
            v[collision_points] *= -1

        self.momentum, self.esc_par = momentum, esc_par
        return self.esc_par, self.momentum


    def escape_velocity(self):
        #Regner ut unnslipningshastighet, basert på at kinetisk og potensiell energi skal være likt
        self.esc_vel = np.sqrt(2*self.G*self.plan_mass/self.plan_radii)
        return self.esc_vel

    def engine_performance(self, N, boost, init_fuel, ant_iter, no_boxes):
        #Oppgave 1D
        #Tester potensialet til motoren, ikke brukt til selve oppskytningen. Ikke tatt hensyn til gravitasjon eller planetens rotasjon
        #Denne biten funker ikke helt, får gale verdier
        box_dim = self.box_dim
        esc_par, momentum = self.moving_particles(N,ant_iter,box_dim)   #Henter verdier, N er antall partikler
        mass_par = self.mass_par         #Massen til partiklene
        sat_mass = self.sat_mass         #Massen til raketten
        vel_esc = self.escape_velocity() #Henter den utregnede unnslipningshastigheten

        calc_t = 10**(-9)                    #Tiden antall unnslupne partikler ble regnet ut over. 1000 tidssteg med dt=10^(-12) = 10*(-9) sek
        esc_par_sec  = esc_par/calc_t        #Partikler som slipper unna per sekund, delt på tiden
        fuel_con = esc_par_sec*mass_par      #Brensel brukt per sekund, oppgitt i kg
        init_tot_mass = sat_mass + init_fuel #Total masse til raketten og drivstoff

        #Krefter:
        thrust = abs(momentum)/calc_t        #Kraften fra bevegelsesmengden per boks, dp/dt
        tot_thrust = thrust*no_boxes         #Total kraft fra hele motoren

        #Verdier brukt i utregningen:
        self.time = 0               #Initierer tidsvariabelen
        dt = 0.01                   #Setter tidssteget
        v = 0                       #Setter startfarten til 0


        for i in range(ant_iter):
        #Regner ut den nye farten med gitt boost og mengde brensel brukt
            a = tot_thrust/init_tot_mass    #Regner ut akselerasjon ved å bruke a=F/m
            v += boost + a*dt               #Regner ut farten med Euler-Cromer
            init_tot_mass -= fuel_con*dt    #Mengde drivstoff per sekund trekkes fra massen til rakett+drivstoff
            self.time += dt
            if init_tot_mass < sat_mass:    #Drivstoff er oppbrukt når den totale massen er mindre enn massen til raketten
                print("Tom for drivstoff")
            elif v < 0:
                print("Gravitasjonen er sterkere enn motoren")
            elif v >= vel_esc:              #Stopper hvis raketten når unnslipningshastighet
                self.fuel_needed = (sat_mass + init_fuel) - init_tot_mass
                print("Unnslipningshastighet nådd. Drivstoff brukt: ",self.fuel_needed, "Tid brukt: ", self.time)
                return self.fuel_needed, self.time

        return 0., 0.

    def simulate_rocket_launch(self, N, init_fuel, no_boxes, max_time, ant_iter):
        #Oppgave 1E
        #Tatt hensyn til gravitasjon og planetens rotasjon.
        box_dim = self.box_dim
        esc_par, momentum = self.moving_particles(N,ant_iter,box_dim)
        mass_par = self.mass_par         #Massen til partiklene
        sat_mass = self.sat_mass         #Massen til raketten
        rot_per = self.rot_per           #Perioden til planeten
        plan_radii= self.plan_radii      #Radien til planeten
        plan_mass =  self.plan_mass      #Massen til planeten
        G = self.G                       #Gravitasjonskonstant
        vel_esc = self.escape_velocity() #Henter den utregnede unnslipningshastigheten
        plan_vel = self.plan_vel

        #Drivstoff
        calc_t = 10**(-9)                             #Tiden antall unnslupne partikler ble regnet ut over. 1000 tidssteg med dt=10^(-12) = 10*(-9) sek
        esc_par_sec  = esc_par/calc_t                 #Partikler som slipper unna per sekund, delt på tiden
        fuel_con = esc_par_sec*mass_par*no_boxes      #Brensel brukt per sekund totalt i motoren, oppgitt i kg
        init_tot_mass = sat_mass + init_fuel          #Total masse til raketten og drivstoff, oppdateres i for-løkka
        print("DETTE ER FUEL CON", fuel_con)


        #Lager tomme arrayer for posisjon, hastighet og tid
        r = np.zeros((ant_iter,2),float)
        v = np.zeros((ant_iter,2),float)
        t = np.linspace(0,max_time,ant_iter)


        #Initialbetingelser
        v_tangent = (2*np.pi/rot_per)*plan_radii        #Regner ut hastigheten i vx-retningen basert på planetens rotasjon
        #v0 = np.array([0,v_tangent + plan_vel])         #Initialhastighet er 0 i x-retning og vy i y-retning
        v0 = np.array([0,0])
        r0 = np.array([plan_radii, 0])                  #Setter origo i sentrum av planeten, derav planetens radius i x-retning
        r[0,:] = r0
        v[0,:] = v0

        #Krefter:
        calc_t = 10**(-9)                    #Tiden antall unnslupne partikler ble regnet ut over
        thrust = abs(momentum)/calc_t        #Kraften fra bevegelsesmengden per boks, dp/dt
        tot_thrust = thrust*no_boxes         #Total kraft fra hele motoren
        print("DETTE ER THRUST", tot_thrust)
        print(t)

        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            #total_system_mass = init_tot_mass + plan_mass                         #Massen til planeten, raketten og drivstoffet. init_tot_mass er raketten + drivstoffet
            gravity = (G*(plan_mass*init_tot_mass))/np.linalg.norm(r[i,:][0])**2  #Gravitasjon, vil variere ettersom raketten mister masse og avstanden til planetens sentrum øker
            a = np.array([(tot_thrust - gravity)/init_tot_mass, 0])               #Antar det kun virker krefter i x-retning, altså veien raketten skal skytes opp. Bruker at a = F/m
            v[i+1,:] = v[i,:] + a*dt
            r[i+1,:] = r[i,:] + v[i+1,:]*dt
            t[i+1] = t[i] + dt
            init_tot_mass -= fuel_con*dt    #Har drivstoff pr sekund, må regne drivstoff pr tidssteg

            #print(r[i+1,:])

            if r[i+1,0] < plan_radii + 100:
                print("Raketten kan ikke gå innover i planeten")
                break

            if init_tot_mass < sat_mass:           #Om massen til raketten og drivstoffet er mindre enn eller lik massen til raketten, er vi tomme for drivstoff
                print(f"Tom for drivstoff etter {t[i],i} sekunder/iterasjoner")
                break


            if vel_esc < np.linalg.norm(v[i]):             #Raketten har nådd escape velocity
                print("Dette er farten til raketten ",np.linalg.norm(v[i]),"og dette er unnslipningshastighet",vel_esc)
                print("Ute i verdensrommet! Tid brukt: ", t[i], "sekunder, eller", t[i]/60, "minutter. Drivstoff brukt: ", init_tot_mass, "kg, høyde over planeten: ", (r[i,:][0]-plan_radii)/1000, "kilometer fra overflaten, hastighet rett opp: ", v[i,:][0], " meter per sekund") #Bruker element nr 2 i siste rad for å printe hastighet og posisjon
                print("---------------------------------------------")
                break

        self.r = r  #Finner slutthøyde i arrayen (final_height_above_surface)
        self.v = v  #Finner sluttfart i arrayen (final_upward_speed)
        self.t = t  #Finner tiden oppskytningen tar i arrayen (launch_duration)
        self.remaining_fuel = init_tot_mass #Finner gjenværende drivstoff (fuel_mass_after_launch)
        self.fuel_con = fuel_con
        self.tot_thrust = tot_thrust
        self.init_fuel = init_fuel
        return self.r, self.v, self.t, self.remaining_fuel, self.fuel_con, self.tot_thrust, self.init_fuel


    def convert_to_solar_system_frame_simple(self):
        #Oppgave 1F
        #Konverterer til et koordinatsystem med stjernen i origo. Antar at stjernen ligger i massesenteret
        r = self.r/const.AU             #Posisjonen gitt i meter, konverterer til AU
        v = self.v*const.c_AU_pr_yr     #Hastigheten gitt i m/s, konverterer til AU/year

        init_plan_pos = system.initial_positions[:,0]         #Posisjonen til planeten ift stjernen
        init_plan_vel = system.initial_velocities[:,0]        #Hastigheten til planeten ift stjernen

        self.rocket_position_before_launch =  np.array([init_plan_pos[0] + r[0][0], init_plan_pos[1] + r[0][1]])
        self.rocket_position_after_launch = np.array([init_plan_pos[0] + r[-1][0], init_plan_pos[1] + r[-1][1]]) #Posisjonen til planeten ift stjernen + raketten ift planeten
        self.rocket_velocity_after_launch = np.array([init_plan_vel[0] + v[-1][0], init_plan_vel[1] + v[-1][1]]) #Hastigheten til planeten ift stjernen + raketten ift planeten

        return self.rocket_position_before_launch, self.rocket_position_after_launch, self.rocket_velocity_after_launch


    def set_parameters_and_verify(self):
        mass_loss_rate_per_box_times_number_of_boxes = self.fuel_con
        thrust_per_box_times_number_of_boxes = abs(self.tot_thrust)
        initial_fuel_mass = self.init_fuel
        launch_duration = self.t[-1] #HER GÅR DET GALT?
        rocket_position_before_launch = self.rocket_position_before_launch
        rocket_position_after_launch = self.rocket_position_after_launch
        time_of_launch = 0

        mission.set_launch_parameters(thrust_per_box_times_number_of_boxes, mass_loss_rate_per_box_times_number_of_boxes,
                                      initial_fuel_mass, launch_duration, rocket_position_before_launch, time_of_launch)

        #Brukt snarvei:
        mission.launch_rocket()
        consumed_fuel_mass, final_time, final_position, final_velocity = shortcuts.get_launch_results()
        mission.verify_launch_result(final_position)



#Input: my, temp, mass_par, k, G, plan_mass, plan_radii, sat_mass, rotational_period, plan_vel, box_dim
launch = Rocket_launch(0, 10000, const.m_H2, const.k_B, const.G, system.masses[0], system.radii[0], mission.spacecraft_mass, system.rotational_periods[0],system.initial_velocities[1][0],10**(-6)) #Har justert størrelsen på hullet i boksen

esc_par, momentum = launch.moving_particles(100000,1000,10**(-6))
print("Antall partikler som slipper unna: ", esc_par, "og bevegelsesmengde per boks: ", momentum)

#Input: N, boost, init_fuel, ant_iter, no_boxes
#launch.engine_performance(10000,50,10000,1000,10**(12)) #Funker ikke helt, får ikke riktige resultater

#Input: simulate_rocket_launch(self, N, init_fuel, no_boxes, max_time, ant_iter)
launch.simulate_rocket_launch(100000,24000,2*10**(11),1200,10)
launch.convert_to_solar_system_frame_simple()
launch.set_parameters_and_verify()
