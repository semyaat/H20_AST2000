from PIL import Image
import numpy as np
from ast2000tools.shortcuts import SpaceMissionShortcuts
from ast2000tools.space_mission import SpaceMission
import ast2000tools.constants as const
import ast2000tools.utils as utils
from ast2000tools.solar_system import SolarSystem
seed = utils.get_seed('mariehav')
system = SolarSystem(seed)

mission = SpaceMission.load('part1.bin')
#Bruker snarvei, men kun posisjon som ikke fungerer
codes = [64662]
shortcuts = SpaceMissionShortcuts(mission, codes)

from orbits import calc_orb

#Egen kode
class Orientation:
    def __init__(self, alpha, theta_0, phi_0):
        self.alpha = np.deg2rad(alpha)
        self.theta_0 = np.deg2rad(theta_0)
        self.phi_0 = np.deg2rad(phi_0) #Vurder å fjerne denne


    def full_range(self):
        alpha = self.alpha
        X_range = 2*np.sin(alpha/2)/(1 + np.cos(alpha/2))
        Y_range = 2*np.sin(alpha/2)/(1 + np.cos(alpha/2))

        X_min = -X_range
        X_max = X_range
        Y_min = -Y_range
        Y_max = Y_range

        return X_min, X_max, Y_min, Y_max


    def grid(self):
        X_min, X_max, Y_min, Y_max = self.full_range()

        img = Image.open('sample0000.png')
        pixels = np.array(img)
        width = np.shape(pixels)[1]
        height = np.shape(pixels)[0]

        x = np.linspace(X_min, X_max, width)
        y = np.linspace(Y_min, Y_max, height)
        X, Y = np.meshgrid(x, y)

        return X, Y, width, height


    def theta_phi(self, phi_0):
        X, Y, width, height = self.grid()

        rho = np.sqrt(X**2 + Y**2)
        beta = 2*np.arctan(rho/2)

        phi = phi_0 + np.arctan(X*np.sin(beta)/(rho*np.sin(self.theta_0)*np.cos(beta) - Y*np.cos(self.theta_0)*np.sin(beta)))

        return phi


    def sky_picture(self, phi_0, save=True):
        X, Y, width, height = self.grid()
        theta, phi = self.theta_phi(phi_0)

        full_sky_image = np.load('himmelkule.npy')
        RGB = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(width):
            for j in range(height):
                index = mission.get_sky_image_pixel(theta[j,i], phi[j,i])
                RGB[j,i] = full_sky_image[index][2:]

        RGB = np.flipud(RGB)

        if save:
            img2 = Image.fromarray(RGB)
            img2.save('bilde.png')

        return RGB


    def sky_picture_360(self):
        for i in range(360):
            RGB = self.sky_picture(np.deg2rad(i), False)
            img = Image.fromarray(RGB)
            img.save(f'himmelkule_grad_test{i}.png')


    def image_analysis(self, picture):
        img = Image.open(picture) #Henter ut bilde vi har tatt
        sat_pic = np.array(img)

        delta_pic = np.zeros(360)

        for i in range(360):
            img2 = Image.open(f'himmelkule_grad_test{i}.png')
            sky_pic = np.array(img2)

            diff = (sat_pic - sky_pic)**2
            delta_pic[i] = np.sum(diff)

        phi = np.argmin(delta_pic)

        return phi


    def doppler_shift_analysis(self, delta_star, delta_rocket):
        lambda_0 =  656.3
        delta_tot = delta_star - delta_rocket
        v_r = const.c_AU_pr_yr*(delta_tot)/lambda_0

        return v_r


    def transform_to_xy(self):
        lambda_sun1, lambda_sun2 = mission.star_doppler_shifts_at_sun
        lambda_star1, lambda_star2 = mission.measure_star_doppler_shifts()
        phi = mission.star_direction_angles

        v1 = self.doppler_shift_analysis(lambda_sun1, lambda_star1)
        v2 = self.doppler_shift_analysis(lambda_sun2, lambda_star2)

        phi1 = np.deg2rad(phi[0])
        phi2 = np.deg2rad(phi[1])
        frac = 1/np.sin(phi2 - phi1)

        vx = frac*(v1*np.sin(phi2) - v2*np.sin(phi1))
        vy = frac*(-v1*np.cos(phi2) + v2*np.cos(phi1))

        return vx, vy


    def trilateration(self, time):
        #Denne gir feil resultat
        #Finner posisjonen til raketten ved t=0
        planet_vel, planet_pos, planet_time = calc_orb.interpolate(time)
        distance = np.array(mission.measure_distances())

        rs, r1, r2 = distance[-1], distance[3], distance[7]     #Avstand mellom sol + to planeter og satelliten
        x1, x2, y1, y2 = planet_pos[3,0], planet_pos[3,1], planet_pos[7,0], planet_pos[7,1]    #Posisjonen til planetene ift sola
        #print(f"Our home planet is positioned at ({planet_pos[0,0]:.5f}, {planet_pos[0,1]:.5f})")

        A = 2*x1
        B = 2*y1
        C = rs**2 - r1**2 + x1**2 + y1**2
        D = 2*(x2 - x1)
        E = 2*(y2 - y1)
        F = r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2

        x = (C*E - F*B)/(E*A - B*D)
        y = (F*A - D*C)/(E*A - B*D)

        return x, y


'''
IKKE KJØR DENNE, lager 360 bilder
space_orientation.sky_picture_360()
'''
#mission.take_picture(filename = 'space_pic.png')

if __name__ == '__main__':
    space_orientation = Orientation(70, 90, 0) #Alpha, theta, phi
    space_orientation.theta_phi(np.deg2rad(0))
    space_orientation.sky_picture(np.deg2rad(0))

    velocity_after_launch = space_orientation.transform_to_xy()
    position_after_launch = space_orientation.trilateration(mission.time_after_launch)
    angle_after_launch = space_orientation.image_analysis('space_pic.png')


    position_after_launch, velocity_after_launch, angle_after_launch = shortcuts.get_orientation_data()
    mission.verify_manual_orientation(position_after_launch, velocity_after_launch, angle_after_launch)
    SpaceMission.save('part4.bin', mission)
