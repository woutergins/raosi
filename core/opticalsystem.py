import logging
logger = logging.getLogger(__name__)

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
from .objects import Lens, Window, Detector

# __all__ = ['GlassObject', 'Lens', 'Window', 'Detector', 'Aperture', 'OpticalSystem', 'RaySource']

class OpticalSystem(object):
    """docstring for OpticalSystem"""
    def __init__(self):
        super(OpticalSystem, self).__init__()
        self.parameters = lmfit.Parameters()
        self.objects = []
        self.ray_sources = []
        self.stepping = 20
        self.n = 1.0

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
        lens_object = Lens(params, material, aperture, clear_aperture, thickness, position)
        self.parameters.add('Lens_'+str(len(self.objects)+1), value=position, brute_step=0.5)
        self.objects.append(['Lens', lens_object])

    def add_window(self, position, material, thickness, aperture, clear_aperture):
        window_object = Window(material, aperture, clear_aperture, thickness, position)
        self.parameters.add('Window_'+str(len(self.objects)+1), value=position, brute_step=0.5)
        self.objects.append(['Window', window_object])

    def add_detector(self, position, aperture, slit=0):
        detector_object = Detector(aperture, position, slit=slit)
        self.parameters.add('Detector_'+str(len(self.objects)+1), value=position, brute_step=0.5)
        self.objects.append(['Detector', detector_object])

    def add_bundle(self, bundle):
        self.original_bundle = bundle

    def clear_bundle(self):
        self.original_bundle = None
        self.bundles = []

    def propagate_to_end(self):
        self.propagate_to_object(len(self.objects)-1)

    def propagate_to_object(self, object_number):
        bundles = [self.original_bundle.clone()]
        for obj in self.objects:
            b = bundles[-1]
            logger.debug('Transferring rays to object {}.'.format(obj[0]))
            return_bundles = obj[1].transfer_rays(b, self.n, self.n)
            bundles.extend(return_bundles)
            for bundle in bundles[:-1]:
                bundle.intersect(bundles[-1])
        self.bundles = bundles

    def parallel_after_object(self, object_number, method='nelder'):
        desired_direction = np.array([[1, 0, 0]]).T

        def cost_function(params):
            self.parameters = params
            for p in self.parameters.keys():
                l = self.parameters[p].value
                obj = int(p.split('_')[1])-1
                self.objects[obj][1].location = l
            self.propagate_to_object(object_number)
            rd = self.bundles[-1].directions
            angle = np.arccos(np.dot(rd, desired_direction))
            return_value = np.abs(angle).sum()
            if np.isnan(return_value):
                return_value = 1e99
            logger.info('Making parallel, f()={:+0.3e}'.format(return_value))
            return return_value

        minimizer = lmfit.Minimizer(cost_function, self.parameters)
        result = minimizer.minimize(method=method)
        return result

    def focus_at_object(self, object_number, method='nelder'):

        def cost_function(params):
            self.parameters = params
            for p in self.parameters.keys():
                l = self.parameters[p].value
                obj = int(p.split('_')[1])-1
                self.objects[obj][1].location = l
            self.propagate_to_object(object_number)
            positions = self.bundles[-1].positions
            y = positions[:, 1]
            z = positions[:, 2]
            spotsize = (y.std()**2+z.std()**2)**0.5
            return_value = spotsize
            if np.isnan(return_value):
                return_value = 1e99
            logger.info('Making focus, f()={:+0.3e}'.format(return_value))
            return return_value

        minimizer = lmfit.Minimizer(cost_function, self.parameters)
        result = minimizer.minimize(method=method)
        return result

    def efficiency(self):
        intensity = self.bundles[-1].intensity
        return intensity.sum() / self.original_bundle.intensity.sum() * 100, intensity.shape[0] / self.original_bundle.intensity.sum() * 100

    def show_distribution(self, ax=None):
        detector = self.bundles[-1].positions
        detector_y, detector_z = detector[:, 1], detector[:, 2]
        intensity = self.bundles[-1].intensity

        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1.0, 1.0])
        else:
            fig = ax.figure

        grid_y, grid_z = int(detector_y.ptp()), int(detector_z.ptp())
        ax.hexbin(detector_y, detector_z, C=intensity, reduce_C_function=np.sum, bins='log', gridsize=(grid_y, grid_z))
        return fig, ax

    def show_ray_paths(self, percentage=100, r_steps=30, theta_steps=40, colormap='viridis', camera_kwargs={'azimuth': 0, 'elevation': 0, 'distance': 180}, filename=None, filename_kwargs={}):
        try:
            if not len(self.rays) == len(self.objects):
                self.propagate_to_end()
        except AttributeError:
            self.propagate_to_end()

        final_rays = self.bundles[-1].positions.shape[0]
        ray_number = min(int(final_rays / 100 * percentage), 1000)
        stepping = int(final_rays / ray_number)
        x = np.hstack([r.positions[::stepping, 0] for r in self.bundles])
        y = np.hstack([r.positions[::stepping, 1] for r in self.bundles])
        z = np.hstack([r.positions[::stepping, 2] for r in self.bundles])
        intensity = np.hstack([r.intensity[::stepping] for r in self.bundles])
        indices = np.hstack([r.indices[::stepping] for r in self.bundles])

        for v in indices:
            locations = np.where(indices==v)[0]
            connect = np.vstack([locations[:-1], locations[1:]]).T
            try:
                connections = np.vstack([connections, connect])
            except UnboundLocalError:
                connections = np.array(connect)

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

        glyph = mlab.pipeline.glyph(lines, colormap=colormap)
        glyph.glyph.glyph.scale_mode = 'data_scaling_off'
        glyph.glyph.glyph.range = np.array([0., 1.])
        glyph.glyph.scale_mode = 'data_scaling_off'

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
