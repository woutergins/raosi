import logging
logger = logging.getLogger(__name__)

import numpy as np
import copy

__all__ = ['Bundle', 'RaySource']

class Bundle(object):
    """"Docstring for Bundle."""
    def __init__(self, positions=np.array([]), directions=np.array([]), intensity=np.array([]), wavelength=np.array([]), indices=None):
        self.positions = positions
        self.directions = directions
        self.intensity = intensity
        self.wavelengths = wavelength
        if indices is None:
            self.generate_indices()
        else:
            self.indices = indices
            self.index_max = self.indices.max()

    def intersect(self, other_bundle):
        _, index1, index2 = np.intersect1d(self.indices, other_bundle.indices, return_indices=True)
        self.positions = self.positions[index1]
        self.directions = self.directions[index1]
        self.intensity = self.intensity[index1]
        self.wavelengths = self.wavelengths[index1]
        self.indices = self.indices[index1]

    def isempty(self):
        return self.positions.shape[0] == 0

    def generate_indices(self):
        self.indices = np.arange(0, self.positions.shape[0], 1)
        self.index_max = self.positions.shape[0]

    def add_rays(self, positions, directions, intensity, wavelength):
        add_indices = np.arange(self.index_max, positions.shape[0], 1)
        self.index_max += positions.shape[0]

        self.positions = np.vstack([self.positions, positions])
        self.directions = np.vstack([self.directions, directions])
        self.intensity = np.vstack([self.intensity, intensity])
        self.wavelengths = np.vstack([self.wavelengths, wavelength])
        self.indices = np.vstack([self.indices, indices])

    def merge(self, bundle):
        positions = np.vstack([self.positions, bundle.positions])
        directions = np.vstack([self.directions, bundle.directions])
        intensity = np.vstack([self.intensity, bundle.intensity])
        wavelength = np.vstack([self.wavelengths, bundle.wavelength])
        indices = np.vstack([self.indices, bundle.indices])

        return_value = self.__class__(positions, directions, intensity, wavelength, indices)
        return_value.index_max = max(self.index_max, bundle.index_max)
        return return_value

    def mask(self, mask, inline=True):
        if inline:
            self.positions = self.positions[mask]
            self.directions = self.directions[mask]
            self.intensity = self.intensity[mask]
            self.wavelengths = self.wavelengths[mask]
            self.indices = self.indices[mask]
        else:
            return_value = self.__class__(self.positions[mask], self.directions[mask], self.intensity[mask], self.wavelengths[mask], self.indices[mask])
            return_value.index_max = self.index_max
            return return_value

    def clone(self):
        return copy.deepcopy(self)


class RaySource(object):
    """docstring for RaySource"""
    def __init__(self, N, positions, distributed, laserbeamsize, wavelength, direction='x'):
        super(RaySource, self).__init__()
        self.N = N
        self.positions = positions
        self.distributed = distributed
        self.laserbeamsize = laserbeamsize
        self.direction = direction
        self.wavelength = wavelength
        
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

        return_rays = Bundle(ray_pos, ray_direc, np.ones(ray_pos.shape[0]), np.ones(ray_pos.shape[0])*self.wavelength)
        return return_rays
