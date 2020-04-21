import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import raosi

np.random.seed(1)
lens_material = "silica"
window_material = "silica"
positions = [0]

N = 1000
distributed = False
laserbeamsize = 0
lens_selection = "ACL7560U"

wavelength = 360

window_location = 112.5 + 35.6  # Flange location + distance from flange to window
window_aperture = 63
window_thickness = 6.4

pmt_distance = 15.49
pmt_size = 47.5

system = raosi.OpticalSystem()

generator = raosi.RaySource(N, positions, distributed, laserbeamsize, wavelength)
system.add_bundle(generator.generate_rays(distribution="pi"))

lens_position = 37.5  # B270 lens

# Doublet of lenses
system.add_lens(lens_selection, lens_position, lens_material, reference=1)
system.parameters["Lens_1"].min = 40
system.parameters["Lens_1"].max = 50

thickness = system.objects[0][1].thickness
r = system.parallel_after_object(0, method="brute")
system.parameters["Lens_1"].vary = False

lens_position = system.parameters["Lens_1"].value
system.add_lens(
    lens_selection, lens_position + thickness + 40, lens_material, reference=2
)
thickness2 = system.objects[1][1].thickness

system.parameters["Lens_2"].min = lens_position + thickness
system.parameters["Lens_2"].max = window_location - thickness2

system.add_window(
    window_location, window_material, window_thickness, window_aperture, window_aperture
)
system.parameters["Window_3"].vary = False

system.add_detector(window_location + window_thickness + pmt_distance, pmt_size, slit=6)
system.parameters["Detector_4"].vary = False

r = system.focus_at_object(4, method="brute")
system.parameters.pretty_print()

aperture = system.objects[0][1].aperture

N = 100000
positions = [-aperture / 2, aperture / 2]
distributed = True
laserbeamsize = 0

# system.clear_bundle()
# generator = raosi.RaySource(N, positions, distributed, laserbeamsize, wavelength)
# system.add_bundle(generator.generate_rays(distribution='uniform'))
# system.plot_efficiency('Lens_1')
# plt.show()

system.show_ray_paths(
    percentage=100,
    camera_kwargs={"azimuth": 0, "elevation": 0, "distance": 250},
    original=25,
)
mlab.show()
# system.show_distribution()
# plt.show()
