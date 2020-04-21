import logging

logger = logging.getLogger(__name__)

import numpy as np
import scipy.optimize as optimize
from .material import Material
from .rays import Bundle

__all__ = ["OpticObject", "GlassObject", "Lens", "Window", "Detector", "Aperture"]


class OpticObject(object):
    """docstring for OpticObject"""

    def __init__(self):
        self.stepping = 20

    def transfer_rays(self, rays, ray_direcs, wavelength):
        raise NotImplemented


class GlassObject(OpticObject):
    """docstring for GlassObject"""

    def __init__(self, material, aperture, clear_aperture, thickness, location):
        super().__init__()
        self.material = Material(material)
        self.aperture = aperture
        self.clear_aperture = clear_aperture
        self.thickness = thickness
        self.location = location

    def transfer_rays(self, bundle, n1=1.0, n2=1.0):

        if bundle.isempty():
            return bundle.clone(), bundle.clone()
        ray_pos = bundle.positions
        ray_direc = bundle.directions
        wavelength = bundle.wavelengths

        def intersection1(x, ray_pos, ray_direc):
            new_ray = ray_pos + (np.atleast_2d(x).T) * ray_direc
            return_val = (
                new_ray[:, 0] - self.give_surface_1(new_ray[:, 1:3]) - self.location
            )
            return return_val

        def intersection2(x, ray_pos, ray_direc):
            new_ray = ray_pos + (np.atleast_2d(x).T) * ray_direc
            return_val = (
                new_ray[:, 0] - self.give_surface_2(new_ray[:, 1:3]) - self.location
            )
            return return_val

        def jacobian(x, ray_pos, ray_direc):
            return -np.eye(x.shape[0])

        for i in range(0, ray_pos.shape[0], self.stepping):
            # logger.debug('Transferring rays ({:d} to {:d} of {:d}) to first surface.'.format(i, min(i+self.stepping, ray_pos.shape[0]), ray_pos.shape[0]))
            initial = (self.location - ray_pos[i : i + self.stepping, 0]) / ray_direc[
                i : i + self.stepping, 0
            ]
            result = optimize.root(
                intersection1,
                initial,
                args=(ray_pos[i : i + self.stepping], ray_direc[i : i + self.stepping]),
                jac=jacobian,
            )
            if not result.success:
                initial = 2 * initial
                result = optimize.root(
                    intersection1,
                    initial,
                    args=(
                        ray_pos[i : i + self.stepping],
                        ray_direc[i : i + self.stepping],
                    ),
                    jac=jacobian,
                )
                result.success = True
                if not result.success:
                    distance = np.atleast_2d(result.x).T
                    raise ValueError
            try:
                distance = np.vstack([distance, (np.atleast_2d(result.x).T)])
            except (ValueError, UnboundLocalError):
                distance = np.atleast_2d(result.x).T

        b1 = bundle.clone()
        ray_pos = ray_pos + distance * ray_direc
        ray_direc = self.refract(n1, ray_direc, ray_pos, wavelength, surface=1)
        b1.positions = ray_pos
        b1.directions = ray_direc

        r = (ray_pos[:, 1] ** 2 + ray_pos[:, 2] ** 2) ** 0.5
        mask = np.logical_and.reduce(
            [
                r < self.clear_aperture / 2,
                ~np.isnan(ray_direc[:, 0]),
                distance[:, 0] > 0,
            ]
        )

        b1.mask(mask)

        if b1.isempty():
            return b1, b1.clone()

        del distance

        b2 = b1.clone()
        ray_pos = b1.positions
        ray_direc = b1.directions
        wavelength = b1.wavelengths

        for i in range(0, ray_pos.shape[0], self.stepping):
            # logger.debug('Transferring rays ({:d} to {:d} of {:d}) to second surface.'.format(i, min(i+self.stepping, ray_pos.shape[0]), ray_pos.shape[0]))
            initial = self.thickness / ray_direc[i : i + self.stepping, 0]
            result = optimize.root(
                intersection2,
                initial,
                args=(ray_pos[i : i + self.stepping], ray_direc[i : i + self.stepping]),
                jac=jacobian,
            )
            if not result.success:
                initial = 2 * initial
                result = optimize.root(
                    intersection2,
                    initial,
                    args=(
                        ray_pos[i : i + self.stepping],
                        ray_direc[i : i + self.stepping],
                    ),
                    jac=jacobian,
                )
                result.success = True
                if not result.success:
                    raise ValueError
            try:
                distance = np.vstack([distance, np.atleast_2d(result.x).T])
            except (ValueError, UnboundLocalError):
                distance = np.atleast_2d(result.x).T
        ray_pos = ray_pos + distance * ray_direc
        ray_direc = self.refract(n2, ray_direc, ray_pos, wavelength, surface=2)
        ray_intensity = self.intensity_after_passage(wavelength, distance.flatten())
        b2.positions = ray_pos
        b2.directions = ray_direc
        b2.intensity = b2.intensity * ray_intensity

        r = (ray_pos[:, 1] ** 2 + ray_pos[:, 2] ** 2) ** 0.5
        mask = np.logical_and.reduce(
            [
                r < self.clear_aperture / 2,
                ~np.isnan(ray_direc[:, 0]),
                distance[:, 0] > 0,
            ]
        )

        b2.mask(mask)

        return b1, b2

    def normal_vector(self, x, surface=1):
        f = getattr(self, "give_surface_" + str(surface))

        x = np.vstack([np.array([0, 1, 0]), x])

        p1 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])

        x[:, 1] = x[:, 1] + 1e-12
        p2 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])
        x[:, 1] = x[:, 1] - 1e-12

        x[:, 2] = x[:, 2] + 1e-12
        p3 = np.hstack([np.atleast_2d(f(x[:, 1:3])).T, x[:, 1:]])

        v1 = p2 - p1
        v1 = v1 / np.atleast_2d(((v1 ** 2).sum(axis=1) ** 0.5)).T
        v2 = p3 - p1
        v2 = v2 / np.atleast_2d(((v2 ** 2).sum(axis=1) ** 0.5)).T
        direc = np.cross(v2, v1)

        return_value = direc / np.atleast_2d((direc ** 2).sum(axis=1)).T ** 0.5
        return_value = return_value[1:, :]

        return return_value

    def refractive_index(self, lamda):
        return self.material.sellmeier(lamda)

    def refract(self, n, ray_direc, ray_position, wavelength, surface=1):
        if surface == 1:
            n1 = n
            n2 = self.refractive_index(wavelength)
        elif surface == 2:
            n1 = self.refractive_index(wavelength)
            n2 = n
        index_ratio = np.atleast_2d(n1 / n2).T
        normal_vector = self.normal_vector(ray_position, surface=surface)
        try:
            cost = np.atleast_2d(np.einsum("ai,ai->a", -normal_vector, ray_direc)).T
        except ValueError:
            cost = np.atleast_2d(np.dot(-normal_vector, ray_direc.T)).T
        new_direc = (
            index_ratio * ray_direc
            + (
                index_ratio * cost
                - np.sqrt(1 - index_ratio * index_ratio * (1 - cost * cost))
            )
            * normal_vector
        )
        return new_direc

    def intensity_after_passage(self, lamda, distance):
        return self.material.intensity(lamda, distance)


class Lens(GlassObject):
    """docstring for Lens"""

    def __init__(
        self,
        parameters,
        material,
        aperture,
        clear_aperture,
        thickness,
        location,
        surface_type="asphere",
    ):
        super(Lens, self).__init__(
            material, aperture, clear_aperture, thickness, location
        )
        self.parameters = parameters
        if isinstance(surface_type, str):
            self.surface_1 = surface_type.lower()
            self.surface_2 = surface_type.lower()
        else:
            self.surface_1 = surface_type[0].lower()
            self.surface_2 = surface_type[1].lower()
        self.__mapping__ = {"asphere": self.sag}

    def sag(self, arr, parameters):
        try:
            C = 1 / parameters["R"]
        except ZeroDivisionError:
            return arr[:, 0] * 0
        Ap = parameters["Ap"]
        K = parameters["K"]
        A2 = parameters["A2"]
        A4 = parameters["A4"]
        A6 = parameters["A6"]
        A8 = parameters["A8"]
        A10 = parameters["A10"]
        A12 = parameters["A12"]
        A14 = parameters["A14"]
        A16 = parameters["A16"]

        y = arr[:, 0]
        z = arr[:, 1]
        rs = y * y + z * z
        r = rs ** 0.5
        sag = (
            C * rs / (1 + (1 - (1 + K) * C * C * rs) ** 0.5)
            + A2 * rs
            + A4 * rs ** 2
            + A6 * rs ** 3
            + A8 * rs ** 4
            + A10 * rs ** 5
            + A12 * rs ** 6
            + A14 * rs ** 7
            + A16 * rs ** 8
        )
        rs = (Ap / 2) ** 2
        sag[r > Ap / 2] = (
            C * rs / (1 + (1 - (1 + K) * C * C * rs) ** 0.5)
            + A2 * rs
            + A4 * rs ** 2
            + A6 * rs ** 3
            + A8 * rs ** 4
            + A10 * rs ** 5
            + A12 * rs ** 6
            + A14 * rs ** 7
            + A16 * rs ** 8
        )
        return sag

    def give_surface_1(self, array):
        return self.__mapping__[self.surface_1](array, self.parameters[0])

    def give_surface_2(self, array):
        return self.thickness - self.__mapping__[self.surface_2](
            array, self.parameters[1]
        )


class Window(GlassObject):
    """docstring for Window"""

    def give_surface_1(self, array):
        return 0 * array[:, 0]

    def give_surface_2(self, array):
        return self.thickness - 0 * array[:, 0]


class Detector(OpticObject):
    """docstring for Window"""

    def __init__(self, aperture, location, slit=0):
        super().__init__()
        self.aperture = aperture
        self.location = location
        self.slit = slit

    def transfer_rays(self, bundle, n1=1.0, n2=1.0):
        b1 = bundle.clone()
        ray_pos = b1.positions
        ray_direc = b1.directions

        def intersection(x, ray_pos, ray_direc):
            new_ray = ray_pos + (np.atleast_2d(x).T) * ray_direc
            return_val = (
                new_ray[:, 0] - self.give_surface_1(new_ray[:, 1:3]) - self.location
            )
            return return_val

        def jacobian(x, ray_pos, ray_direc):
            return -np.eye(x.shape[0])

        for i in range(0, ray_pos.shape[0], self.stepping):
            # logger.debug('Transferring rays ({:d} to {:d} of {:d}) to detector surface.'.format(i, min(i+self.stepping, ray_pos.shape[0]), ray_pos.shape[0]))
            initial = (self.location - ray_pos[i : i + self.stepping, 0]) / ray_direc[
                i : i + self.stepping, 0
            ]
            result = optimize.root(
                intersection,
                initial,
                args=(ray_pos[i : i + self.stepping], ray_direc[i : i + self.stepping]),
                jac=jacobian,
            )
            try:
                distance = np.vstack([distance, np.atleast_2d(result.x).T])
            except (ValueError, UnboundLocalError):
                distance = np.atleast_2d(result.x).T
        ray_pos = ray_pos + distance * ray_direc
        b1.positions = ray_pos

        r = (ray_pos[:, 1] ** 2 + ray_pos[:, 2] ** 2) ** 0.5

        slit = self.slit
        if not slit > 0:
            mask = np.logical_and.reduce(
                [
                    r <= self.aperture / 2,
                    ~np.isnan(ray_direc[:, 0]),
                    distance[:, 0] >= 0,
                ]
            )
        else:
            z = ray_pos[:, 2]
            mask = np.logical_and.reduce(
                [
                    r <= self.aperture / 2,
                    np.abs(z) < slit / 2,
                    ~np.isnan(ray_direc[:, 0]),
                    distance[:, 0] >= 0,
                ]
            )

        b1.mask(mask)
        return (b1,)

    def give_surface_1(self, array):
        return 0 * array[:, 0]


class Aperture(OpticObject):
    """docstring for Window"""

    def __init__(self, aperture):
        super(Aperture, self).__init__()
        self.aperture = aperture

    def give_surface_1(self, array):
        return 0 * array[:, 0]
