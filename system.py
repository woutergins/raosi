import numpy as np
import scipy.optimize as optimize
import collections
import matplotlib.pyplot as plt
import lmfit
from mayavi import mlab
from mayavi.filters.transform_data import TransformData
from mayavi.modules.surface import Surface
import mayavi
import vtk
from .material import Material

__all__ = ['GlassObject', 'Lens', 'Window', 'Detector', 'Aperture', 'OpticalSystem', 'RaySource']

class GlassObject(object):
    """docstring for GlassObject"""
    def __init__(self, material, aperture, clear_aperture, thickness):
        super(GlassObject, self).__init__()
        self.material = Material(material)
        self.aperture = aperture
        self.clear_aperture = clear_aperture
        self.thickness = thickness

    def normal_vector(self, x, surface=1):
        f = getattr(self, 'give_surface_'+str(surface))
        x = np.vstack([np.array([0, 1, 0]), x])
        p1 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])
        x[:, 1] = x[:, 1]+1e-12
        p2 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])
        x[:, 1] = x[:, 1]-1e-12
        x[:, 2] = x[:, 2]+1e-12
        p3 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])
        v1 = p2-p1
        v2 = p3-p1
        v1 = v1 / np.atleast_2d(((v1**2).sum(axis=1)**0.5)).T
        v2 = v2 / np.atleast_2d(((v2**2).sum(axis=1)**0.5)).T
        direc = np.cross(v2, v1)
        return_value = direc/np.atleast_2d((direc**2).sum(axis=1)).T**0.5
        return_value = return_value[1:, :]
        return return_value

    def set_refractive_index(self, lamda):
        self.n = self.material.sellmeier(lamda)

    def refract(self, n, ray_direc, ray_position, surface=1):
        if surface == 1:
            n1 = n
            n2 = self.n
        elif surface == 2:
            n1 = self.n
            n2 = n
        index_ratio = n1/n2
        normal_vector = self.normal_vector(ray_position, surface=surface)
        try:
            cost = np.atleast_2d(np.einsum('ai,ai->a', -normal_vector, ray_direc)).T
        except ValueError:
            cost = np.atleast_2d(np.dot(-normal_vector, ray_direc.T)).T
        new_direc = index_ratio*ray_direc+(index_ratio*cost-np.sqrt(1-index_ratio*index_ratio*(1-cost*cost)))*normal_vector
        return new_direc

    def intensity_after_passage(self, lamda, distance):
        return self.material.intensity(lamda, distance)

class Lens(GlassObject):
    """docstring for Lens"""
    def __init__(self, parameters, material, aperture, clear_aperture, thickness, surface_type='asphere'):
        super(Lens, self).__init__(material, aperture, clear_aperture, thickness)
        self.parameters = parameters
        if isinstance(surface_type, str):
            self.surface_1 = surface_type.lower()
            self.surface_2 = surface_type.lower()
        else:
            self.surface_1 = surface_type[0].lower()
            self.surface_2 = surface_type[1].lower()
        self.__mapping__ = {'asphere': self.sag}

    def sag(self, arr, parameters):
        try:
            C = 1/parameters['R']
        except ZeroDivisionError:
            return arr[:, 0]*0
        Ap = parameters['Ap']
        K = parameters['K']
        A2 = parameters['A2']
        A4 = parameters['A4']
        A6 = parameters['A6']
        A8 = parameters['A8']
        A10 = parameters['A10']
        A12 = parameters['A12']
        A14 = parameters['A14']
        A16 = parameters['A16']

        y = arr[:, 0]
        z = arr[:, 1]
        rs = y*y+z*z
        r = rs**0.5
        sag = C*rs/(1+(1-(1+K)*C*C*rs)**0.5)+A2*rs+A4*rs**2+A6*rs**3+A8*rs**4+A10*rs**5+A12*rs**6+A14*rs**7+A16*rs**8
        rs = (Ap/2)**2
        sag[r>Ap/2] = C*rs/(1+(1-(1+K)*C*C*rs)**0.5)+A2*rs+A4*rs**2+A6*rs**3+A8*rs**4+A10*rs**5+A12*rs**6+A14*rs**7+A16*rs**8
        return sag

    def give_surface_1(self, array):
        return self.__mapping__[self.surface_1](array, self.parameters[0])

    def give_surface_2(self, array):
        return self.thickness - self.__mapping__[self.surface_2](array, self.parameters[1])

class Window(GlassObject):
    """docstring for Window"""
    def __init__(self, material, aperture, clear_aperture, thickness):
        super(Window, self).__init__(material, aperture, clear_aperture, thickness)

    def give_surface_1(self, array):
        return 0*array[:, 0]

    def give_surface_2(self, array):
        return self.thickness - 0*array[:, 0]

class Detector(object):
    """docstring for Window"""
    def __init__(self, aperture, slit=0):
        super(Detector, self).__init__()
        self.aperture = aperture
        self.slit = slit

    def give_surface_1(self, array):
        return 0*array[:, 0]

class Aperture(object):
    """docstring for Window"""
    def __init__(self, aperture):
        super(Aperture, self).__init__()
        self.aperture = aperture

    def give_surface_1(self, array):
        return 0*array[:, 0]

class OpticalSystem(object):
    """docstring for OpticalSystem"""
    def __init__(self, wavelength):
        super(OpticalSystem, self).__init__()
        self.parameters = lmfit.Parameters()
        self.objects = []
        self.ray_sources = []
        self.object_number = 0
        self.stepping = 20
        self.n = 1.0
        self.wavelength = wavelength

    def add_lens(self, lens_selection, position, material, reference=1):
        parameters = {
        'ACL108U':   [collections.defaultdict(float, ), collections.defaultdict(float, Ap=10, R=4.185, K=-0.6027, A4=2.21e-4), 10, 8, 5.8],
        'ACL1210U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=12, R=5.492, K=-0.6230, A4=8.7e-5), 12, 10, 5.8],
        'ACL1512U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=15, R=6.277, K=-0.6139, A4=6.8E-5), 15, 13, 8.0],
        'ACL1815U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=18, R=7.818, K=-1.817, A4=2.93E-4), 18, 16, 8.2],
        'ACL2018U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=20, R=9.415, K=-0.6392, A4=1.7E-5), 20, 18, 8.0],
        'ACL2520U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=25, R=10.462, K=-0.6265, A4=1.5E-5), 25, 23, 12.0],
        'ACL3026U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=30, R=13.551, K=-0.6301, A4=5.5E-6), 30, 28, 11.9],
        'ACL5040U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=50, R=20.923, K=-0.6405, A4=2.0E-6), 50, 48, 21.0],
        'ACL7560U':  [collections.defaultdict(float, ), collections.defaultdict(float, Ap=75, R=31.384, K=-1.911, A4=5.0E-6), 75, 73, 30.0],
        'A100-100LPX-S-U': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=100,
                                                   R=1/0.01956181533646322378716744913928,
                                                   K=-1.023,
                                                   A4=4.4278927e-07,
                                                   A6=2.8715019e-11,
                                                   A8=1.9201195e-15,
                                                   A10=9.2124803e-20,
                                                   A12=-1.6052264e-24,
                                                   A14=-5.8638374e-28,
                                                   A16 = -3.0821914e-31,), 100, 98, 36.0],
        'AFL50-60-S-U': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=50,
                                                   R=1/3.384438352455410000e-2,
                                                   K=-6.6e-1,
                                                   A4=5.01e-7,
                                                   A6=1.24e-10,), 50, 48, 17.5],
        'LA4464-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=27.6), 2*25.4, 2*25.4-2, 19.8],
        'LA4078-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=34.5), 2*25.4, 2*25.4-2, 14.2],
        'LA4464-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=27.6), 2*25.4, 2*25.4-2, 19.8],
        'LA4545-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=46), 2*25.4, 2*25.4-2, 10.7],
        'LA4904-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=69.0), 2*25.4, 2*25.4-2, 7.8],
        'LA4984-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=92.0), 2*25.4, 2*25.4-2, 6.6],
        'LA4538-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=115.0), 2*25.4, 2*25.4-2, 5.8],
        'LA4855-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=138.0), 2*25.4, 2*25.4-2, 5.4],
        'LA4782-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=230.0), 2*25.4, 2*25.4-2, 4.4],
        'LA4745-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=345.1), 2*25.4, 2*25.4-2, 3.9],
        'LA4337-UV': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=2*25.4,
                                                   R=460.1), 2*25.4, 2*25.4-2, 3.7],
        'AFL50-80-S-U': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=50,
                                                   K=-0.67,
                                                   R=1/2.538328764341557500E-002,
                                                   A4=2.11e-7,
                                                   A6=2.47e-11), 50, 48, 14],
        'AFL50-60-S-U-mod': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=75,
                                                   R=1/3.384438352455410000e-2,
                                                   K=-6.6e-1,
                                                   A4=5.01e-7,
                                                   A6=1.24e-10,), 75, 73, 35],
        'AFL50-80-S-U-mod': [collections.defaultdict(float, ), collections.defaultdict(float,
                                                   Ap=75,
                                                   K=-0.67,
                                                   R=1/2.538328764341557500E-002,
                                                   A4=2.11e-7,
                                                   A6=2.47e-11), 75, 73, 25],
        }
        if reference == 1:
            params = [parameters[lens_selection][0], parameters[lens_selection][1]]
        elif reference == 2:
            params = [parameters[lens_selection][1], parameters[lens_selection][0]]
        else:
            raise ValueError
        aperture = parameters[lens_selection][2]
        clear_aperture = parameters[lens_selection][3]
        thickness = parameters[lens_selection][4]
        lens_object = Lens(params, material, aperture, clear_aperture, thickness)
        self.object_number += 1
        self.parameters.add('Lens_'+str(self.object_number), value=position, brute_step=0.5)
        self.objects.append(['Lens', lens_object])

    def add_window(self, position, material, thickness, aperture, clear_aperture):
        window_object = Window(material, aperture, clear_aperture, thickness)
        self.object_number += 1
        self.objects.append(['Window', window_object])
        self.parameters.add('Window_'+str(self.object_number), value=position, brute_step=0.5)

    def add_detector(self, position, aperture, slit=0):
        detector_object = Detector(aperture, slit=slit)
        self.object_number += 1
        self.objects.append(['Detector', detector_object])
        self.parameters.add('Detector', value=position, brute_step=0.5)

    def add_ray_source(self, N, positions, distributed, laserbeamsize):
        source = RaySource(N, positions, distributed, laserbeamsize)
        self.ray_sources.append(source)

    def clear_ray_sources(self):
        self.ray_sources = []

    def generate_rays(self, distribution='pi'):
        for s in self.ray_sources:
            rp, rd = s.generate_rays(distribution)
            try:
                ray_pos = np.vstack([ray_pos, rp])
                ray_direc = np.vstack([ray_direc, rd])
            except UnboundLocalError:
                ray_pos = rp
                ray_direc = rd
        self.original_rays = ray_pos
        self.original_ray_direcs = ray_direc
        self.original_intensities = np.ones(self.original_rays.shape[0])

    def prepare_objects(self):
        for obj in range(len(self.objects)):
            if self.objects[obj][0].lower() in ['lens', 'window']:
                self.objects[obj][1].set_refractive_index(self.wavelength)

    def propagate_to_end(self, ):
        self.propagate_to_object(self.object_number-1)

    def jacobian(self, x, ray_pos, ray_direc):
        return -np.eye(x.shape[0])

    def propagate_to_object(self, object_number, debug=False):
        self.rays = [self.original_rays]
        self.ray_direcs = [self.original_ray_direcs]
        self.ray_intensities = [self.original_intensities]

        for obj in range(object_number+1):
            ray_pos = self.rays[-1]
            ray_direc = self.ray_direcs[-1]

            if self.objects[obj][0].lower() in ['lens', 'window']:
                l = self.parameters[self.objects[obj][0]+'_'+str(obj+1)].value

                def intersection1(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_1(new_ray[:, 1:3])-l
                    return return_val

                def intersection2(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_2(new_ray[:, 1:3])-l
                    return return_val

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l - ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection1, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]), jac=self.jacobian)
                    if not result.success:
                        initial = 2 * initial
                        result = optimize.root(intersection1, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]), jac=self.jacobian)
                        result.success = True
                        if not result.success:
                            distance = np.atleast_2d(result.x).T
                            raise ValueError
                    try:
                        distance = np.vstack([distance, (np.atleast_2d(result.x).T)])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                try:
                    ray_pos = ray_pos + distance*ray_direc
                    ray_direc = self.objects[obj][1].refract(self.n, ray_direc, ray_pos, surface=1)
                    r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                    mask = np.logical_and.reduce([r < self.objects[obj][1].clear_aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])

                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1])
                    self.rays = [r[mask] for r in self.rays]
                    self.ray_direcs = [r[mask] for r in self.ray_direcs]
                    self.ray_intensities = [r[mask] for r in self.ray_intensities]

                    del distance

                except UnboundLocalError:
                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1])

                ray_pos = self.rays[-1]
                ray_direc = self.ray_direcs[-1]

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l+self.objects[obj][1].thickness-ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection2, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                    if not result.success:
                        initial = 2* initial
                        result = optimize.root(intersection2, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                        result.success = True
                        if not result.success:
                            raise ValueError
                    try:
                        distance = np.vstack([distance, np.atleast_2d(result.x).T])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                try:
                    ray_pos = ray_pos + distance*ray_direc
                    ray_direc = self.objects[obj][1].refract(self.n, ray_direc, ray_pos, surface=2)
                    ray_intensity = self.objects[obj][1].intensity_after_passage(self.wavelength, distance.flatten())
                    r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                    mask = np.logical_and.reduce([r < self.objects[obj][1].clear_aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])

                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1]*ray_intensity)
                    self.rays = [r[mask] for r in self.rays]
                    self.ray_direcs = [r[mask] for r in self.ray_direcs]
                    self.ray_intensities = [r[mask] for r in self.ray_intensities]
                    
                    del distance

                except UnboundLocalError:
                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1])

            elif self.objects[obj][0].lower() in ['detector']:
                try:
                    l = self.parameters[self.objects[obj][0]+'_'+str(obj+1)].value
                except KeyError:
                    l = self.parameters[self.objects[obj][0]].value

                def intersection(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_1(new_ray[:, 1:3])-l
                    return return_val

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l - ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                    try:
                        distance = np.vstack([distance, np.atleast_2d(result.x).T])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                try:
                    ray_pos = ray_pos + distance*ray_direc
                    r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                    slit = self.objects[obj][1].slit
                    if not slit > 0:
                        mask = np.logical_and.reduce([r <= self.objects[obj][1].aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] >= 0])
                    else:
                        z = ray_pos[:, 2]
                        mask = np.logical_and.reduce([r <= self.objects[obj][1].aperture/2, np.abs(z)<slit/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] >= 0])

                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1])
                    self.rays = [r[mask] for r in self.rays]
                    self.ray_direcs = [r[mask] for r in self.ray_direcs]
                    self.ray_intensities = [r[mask] for r in self.ray_intensities]

                    del distance

                except UnboundLocalError:
                    self.rays.append(ray_pos)
                    self.ray_direcs.append(ray_direc)
                    self.ray_intensities.append(self.ray_intensities[-1])

    def propagate_from_object_to(self, starting_object, end_object):
        should_have = 1
        for i in range(starting_object + 1):
            if self.objects[i][0].lower() in ['lens', 'window']:
                should_have += 2
            else:
                should_have += 1
        if len(self.rays) != should_have:
            self.propagate_to_object(starting_object)

        self.rays = self.rays[:should_have]
        self.ray_direcs = self.ray_direcs[:should_have]
        self.ray_intensities = self.ray_intensities[:should_have]

        ray_pos = self.rays[-1]
        ray_direc = self.ray_direcs[-1]
        ray_intensities = self.ray_intensities[-1]

        for obj in range(starting_object + 1, end_object+1):
            if self.objects[obj][0].lower() in ['lens', 'window']:
                l = self.parameters[self.objects[obj][0]+'_'+str(obj+1)].value

                def intersection1(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_1(new_ray[:, 1:3])-l
                    return return_val

                def intersection2(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_2(new_ray[:, 1:3])-l
                    return return_val

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l - ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection1, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                    if not result.success:
                        initial = 2* initial
                        result = optimize.root(intersection1, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                        if not result.success:
                            print(self.objects[obj], obj, result)
                            print(ray_pos[i:i+self.stepping])
                            raise ValueError
                    try:
                        distance = np.vstack([distance, (np.atleast_2d(result.x).T)])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                ray_pos = ray_pos + distance*ray_direc
                ray_direc = self.objects[obj][1].refract(self.n, ray_direc, ray_pos, surface=1)
                r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                mask = np.logical_and.reduce([r < self.objects[obj][1].aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])

                ray_pos = ray_pos[mask]
                ray_direcs = ray_direcs[mask]
                ray_intensities = ray_intensities[mask]

                del distance

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l+self.objects[obj][1].thickness-ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection2, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                    if not result.success:
                        initial = 2* initial
                        result = optimize.root(intersection2, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                        if not result.success:
                            raise ValueError
                    try:
                        distance = np.vstack([distance, np.atleast_2d(result.x).T])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                ray_pos = ray_pos + distance*ray_direc
                ray_direc = self.objects[obj][1].refract(self.n, ray_direc, ray_pos, surface=2)
                ray_intensity = self.objects[obj][1].intensity_after_passage(self.wavelength, distance.flatten())
                r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                mask = np.logical_and.reduce([r < self.objects[obj][1].aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])

                ray_pos = ray_pos[mask]
                ray_direcs = ray_direcs[mask]
                ray_intensities = ray_intensities[mask]

                del distance
            elif self.objects[obj][0].lower() in ['detector']:
                try:
                    l = self.parameters[self.objects[obj][0]+'_'+str(obj+1)].value
                except KeyError:
                    l = self.parameters[self.objects[obj][0]].value

                def intersection(x, ray_pos, ray_direc):
                    new_ray = ray_pos + (np.atleast_2d(x).T)*ray_direc
                    return_val = new_ray[:, 0] - self.objects[obj][1].give_surface_1(new_ray[:, 1:3])-l
                    return return_val

                for i in range(0, ray_pos.shape[0], self.stepping):
                    initial = (l - ray_pos[i:i+self.stepping, 0])/ray_direc[i:i+self.stepping, 0]
                    result = optimize.root(intersection, np.ones(ray_pos[i:i+self.stepping].shape[0])*initial, args=(ray_pos[i:i+self.stepping], ray_direc[i:i+self.stepping]))
                    try:
                        distance = np.vstack([distance, np.atleast_2d(result.x).T])
                    except (ValueError, UnboundLocalError):
                        distance = np.atleast_2d(result.x).T
                ray_pos = ray_pos + distance*ray_direc
                r = (ray_pos[:, 1]**2+ray_pos[:, 2]**2)**0.5

                slit = self.objects[obj][1].slit
                if not slit > 0:
                    mask = np.logical_and.reduce([r < self.objects[obj][1].aperture/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])
                else:
                    z = ray_pos[:, 2]
                    mask = np.logical_and.reduce([r < self.objects[obj][1].aperture/2, np.abs(z)<slit/2, ~np.isnan(ray_direc[:, 0]), distance[:, 0] > 0])

                ray_pos = ray_pos[mask]
                ray_direc = ray_direc[mask]
                ray_intensities = ray_intensities[mask]

                del distance
        return ray_pos, ray_direc, ray_intensities

    def parallel_after_object(self, object_number, method='powell'):
        desired_direction = np.array([[1, 0, 0]]).T

        def cost_function(params):
            self.parameters = params
            self.propagate_to_object(object_number)
            rd = self.ray_direcs[-1]
            angle = np.arccos(np.dot(rd, desired_direction))
            return_value = np.abs(angle).sum()
            print(params, return_value)
            return return_value

        minimizer = lmfit.Minimizer(cost_function, self.parameters)
        result = minimizer.minimize(method=method)
        self.parameters = result.params
        self.parameters.pretty_print()
        self.propagate_to_object(object_number)

    def focus_at_object(self, object_number, method='powell'):

        def cost_function(params):
            self.parameters = params
            self.propagate_to_object(object_number)
            positions = self.rays[-1]
            y = positions[:, 1]
            z = positions[:, 2]
            spotsize = (y.std()**2+z.std()**2)**0.5
            absorbed_eff, eff = self.efficiency()
            return_value = spotsize - absorbed_eff
            print(params, return_value)
            return return_value

        minimizer = lmfit.Minimizer(cost_function, self.parameters)
        result = minimizer.minimize(method=method)
        self.parameters = result.params
        self.parameters.pretty_print()
        self.propagate_to_object(object_number)

    def efficiency(self):
        intensity = self.ray_intensities[-1]
        return intensity.sum() / self.original_intensities.sum() * 100, intensity.shape[0] / self.original_intensities.sum() * 100

    def show_distribution(self, ax=None):
        detector = self.rays[-1]
        detector_y, detector_z = detector[:, 1], detector[:, 2]
        intensity = self.ray_intensities[-1]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1.0, 1.0])
        else:
            fig = ax.figure

        grid_y, grid_z = int(detector_y.ptp()), int(detector_z.ptp())
        ax.hexbin(detector_y, detector_z, C=intensity, reduce_C_function=np.sum, bins='log', gridsize=(grid_y, grid_z))
        return fig, ax

    def show_ray_paths(self, stepping=10, r_steps=30, theta_steps=40, colormap='viridis', camera_kwargs={'azimuth': 0, 'elevation': 0, 'distance': 180}, filename=None, filename_kwargs={}):
        try:
            if not len(self.rays) == len(self.objects):
                self.propagate_to_end()
        except AttributeError:
            self.propagate_to_end()

        x = np.hstack([r[::stepping, 0] for r in self.rays])
        y = np.hstack([r[::stepping, 1] for r in self.rays])
        z = np.hstack([r[::stepping, 2] for r in self.rays])
        intensity = np.hstack([r[::stepping] for r in self.ray_intensities])

        number_rays = self.rays[0][::stepping, 0].shape[0]
        points = len(self.rays)
        connections = np.vstack([
            np.arange(0, (points-1)*number_rays, number_rays),
            np.arange(number_rays, points*number_rays, number_rays),
            ]).T
        connections = np.vstack([
                                connections + n for n in range(number_rays)
                                ])

        mlab.figure(1, size=(1920, 1080), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        vtk.vtkObject.GlobalWarningDisplayOff()
        mlab.clf()

        s = x
        from scipy import stats
        kde = stats.gaussian_kde(np.vstack([x, y, z]), weights=intensity)
        s = kde(np.vstack([x, y, z]))
        s = np.log10(s)
        vmin = s.min()
        vmax = s.max()

        src = mlab.pipeline.scalar_scatter(x, y, z, s, vmin=vmin, vmax=vmax)

        src.mlab_source.dataset.lines = connections
        src.name = 'Ray data'
        src.update()

        lines = mlab.pipeline.tube(src, tube_radius=0.25, tube_sides=4)
        surf_tubes = mlab.pipeline.surface(lines, colormap=colormap, line_width=1, opacity=0.4)
        surf_tubes.name = 'Rays'

        # glyph = mlab.pipeline.glyph(lines, colormap=colormap)
        # glyph.glyph.glyph.scale_mode = 'data_scaling_off'
        # glyph.glyph.glyph.range = np.array([0., 1.])
        # glyph.glyph.scale_mode = 'data_scaling_off'

        for k, obj in enumerate(self.objects):
            if obj[0].lower() in ['detector']:
                continue
            try:
                l = self.parameters[self.objects[k][0]+'_'+str(k+1)].value
            except KeyError:
                l = self.parameters[self.objects[k][0]].value
            r = np.linspace(0, obj[1].aperture / 2, r_steps)
            r = r[0:]
            theta = np.linspace(0, 2*np.pi, theta_steps)
            r, theta = np.meshgrid(r, theta)
            x = (r*np.cos(theta)).flatten()
            y = (r*np.sin(theta)).flatten()
            x = np.hstack([0, x])
            y = np.hstack([0, y])
            z = np.zeros(x.shape)
            for i, (X, Y) in enumerate(zip(x, y)):
                z[i] = obj[1].give_surface_1(np.vstack([X, Y]).T)
            z = z + l
            if obj[0].lower() in ['lens', 'window']:
                z2 = np.zeros(x.shape)
                for i, (X, Y) in enumerate(zip(x, y)):
                    z2[i] = obj[1].give_surface_2(np.vstack([X, Y]).T)
                z2 = z2 + l
                x = np.vstack([x, x])
                y = np.vstack([y, y])
                z = np.vstack([z, z2])
            x, y, z = x.flatten(), y.flatten(), z.flatten()
            x, y, z = z, x, y
            s = kde(np.vstack([x, y, z]))
            s = np.log10(s)
            vtk_source = mlab.pipeline.scalar_scatter(x, y, z, s)
            vtk_source.name = obj[0] + ' ' + str(k+1) + ' data'
            delaunay = mlab.pipeline.delaunay3d(vtk_source)
            s = mlab.pipeline.surface(delaunay, opacity=0.8, colormap=colormap, vmin=vmin, vmax=vmax)
        if filename is not None:
            engine = mlab.get_engine()
            scene = engine.scenes[0]

            poly_data_reader = engine.open(filename, scene)
            mlab.pipeline.surface(poly_data_reader, **filename_kwargs)
        mlab.view(azimuth=camera_kwargs['azimuth'], elevation=camera_kwargs['elevation'], distance=camera_kwargs['distance'], focalpoint='auto')

class RaySource(object):
    """docstring for RaySource"""
    def __init__(self, N, positions, distributed, laserbeamsize, direction='x'):
        super(RaySource, self).__init__()
        self.N = N
        self.positions = positions
        self.distributed = distributed
        self.laserbeamsize = laserbeamsize
        self.direction = direction
        
    def hit_and_miss_3d_dipole_donut(self, n):
        pos = 3*np.random.rand(n, 3)-1.5
        # pos[:, 0] = np.abs(pos[:, 0])

        r = (pos**2).sum(axis=1)**0.5
        pol = np.arccos(pos[:, 2]/r)
        azi = np.arctan2(pos[:, 1], pos[:, 0])

        accepted = r<=np.sin(pol)**2
        while not np.all(accepted):
            naccepted = ~accepted
            naccepted_num = naccepted.sum()
            pos[naccepted] = 3*np.random.rand(naccepted_num, 3)-1.5
            # pos[naccepted, 0] = np.abs(pos[naccepted, 0])
            r[naccepted] = (pos[naccepted]**2).sum(axis=1)**0.5
            pol[naccepted] = np.arccos(pos[naccepted, 2]/r[naccepted])
            azi[naccepted] = np.arctan2(pos[naccepted, 1], pos[naccepted, 0])
            accepted[naccepted] = r[naccepted] <= np.sin(pol[naccepted])**2
        return pos

    def hit_and_miss_3d_dipole_dumbbell(self, n):
        pos = 5*np.random.rand(n, 3)-2.5
        # pos[:, 0] = np.abs(pos[:, 0])

        r = (pos**2).sum(axis=1)**0.5
        pol = np.arccos(pos[:, 2]/r)
        azi = np.arctan2(pos[:, 1], pos[:, 0])

        accepted = r<=(1+np.cos(pol)**2)
        while not np.all(accepted):
            naccepted = ~accepted
            naccepted_num = naccepted.sum()
            pos[naccepted] = 5*np.random.rand(naccepted_num, 3)-2.5
            # pos[naccepted, 0] = np.abs(pos[naccepted, 0])
            r[naccepted] = (pos[naccepted]**2).sum(axis=1)**0.5
            pol[naccepted] = np.arccos(pos[naccepted, 2]/r[naccepted])
            azi[naccepted] = np.arctan2(pos[naccepted, 1], pos[naccepted, 0])
            accepted[naccepted] = r[naccepted] <= (1+np.cos(pol[naccepted])**2)
        return pos

    def hit_and_miss_3d_uniform(self, n):
        pos = 5*np.random.rand(n, 3)-2.5
        # pos[:, 0] = np.abs(pos[:, 0])

        r = (pos**2).sum(axis=1)**0.5
        pol = np.arccos(pos[:, 2]/r)
        azi = np.arctan2(pos[:, 1], pos[:, 0])

        accepted = r<=1
        while not np.all(accepted):
            naccepted = ~accepted
            naccepted_num = naccepted.sum()
            pos[naccepted] = 5*np.random.rand(naccepted_num, 3)-2.5
            # pos[naccepted, 0] = np.abs(pos[naccepted, 0])
            r[naccepted] = (pos[naccepted]**2).sum(axis=1)**0.5
            pol[naccepted] = np.arccos(pos[naccepted, 2]/r[naccepted])
            azi[naccepted] = np.arctan2(pos[naccepted, 1], pos[naccepted, 0])
            accepted[naccepted] = r[naccepted] <= 1
        return pos

    def generate_rays(self, distribution='pi'):
        if distribution.lower() == 'pi':
            function = self.hit_and_miss_3d_dipole_donut
        elif distribution.lower() == 'sigma':
            function = self.hit_and_miss_3d_dipole_dumbbell
        elif distribution.lower() == 'uniform':
            function = self.hit_and_miss_3d_uniform

        x_pos = np.array([])
        y_pos = np.array([])
        z_pos = np.array([])

        if not self.distributed:
            for p in self.positions:
                if self.laserbeamsize > 0:
                    pos = self.laserbeamsize*np.random.rand(self.N, 2)-self.laserbeamsize/2
                    while np.any(((pos**2).sum(axis=1))**0.5>self.laserbeamsize):
                        not_ok = ((pos**2).sum(axis=1))**0.5>self.laserbeamsize
                        pos[not_ok] = self.laserbeamsize*np.random.rand(not_ok.sum(), 2)-self.laserbeamsize/2
                else:
                    pos = np.zeros((self.N, 2))
                x_pos = np.append(x_pos, pos[:, 0])
                y_pos = np.append(y_pos, np.zeros(self.N)+p)
                z_pos = np.append(z_pos, pos[:, 1])

        else:
            low, high = np.min(self.positions), np.max(self.positions)
            y_pos = (high-low)*np.random.rand(self.N)+low
            if self.laserbeamsize > 0:
                pos = self.laserbeamsize*np.random.rand(self.N, 2)-self.laserbeamsize/2
                while np.any(((pos**2).sum(axis=1))**0.5>self.laserbeamsize):
                    not_ok = ((pos**2).sum(axis=1))**0.5>self.laserbeamsize
                    pos[not_ok] = self.laserbeamsize*np.random.rand(not_ok.sum(), 2)-self.laserbeamsize/2
            else:
                pos = np.zeros((self.N, 2))
            x_pos = pos[:, 0]
            z_pos = pos[:, 1]

        direc = function(len(x_pos))
        direc = direc / np.atleast_2d((direc**2).sum(axis=1)**0.5).T
        x_direc, y_direc, z_direc = direc[:, 0], direc[:, 1], direc[:, 2]

        ray_pos = np.vstack([x_pos, y_pos, z_pos]).T
        ray_direc = np.vstack([
            x_direc,
            y_direc,
            z_direc,
            ]).T

        return ray_pos, ray_direc
