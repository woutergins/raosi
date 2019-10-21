import unittest
import numpy as np
from core import OpticalSystem

class TestRayTracer(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.lens_selection = 'ACL7560U'
        self.lens_material = 'b270'
        self.system = OpticalSystem(wavelength=360)
        self.system.add_ray_source(100, [0], False, 0)
        self.system.generate_rays(distribution='uniform')
        self.lens_position = 40

    def test_parallel(self):
        self.system.add_lens(self.lens_selection, self.lens_position, self.lens_material, reference=1)
        self.system.prepare_objects()
        self.system.parameters['Lens_1'].min = 30
        self.system.parameters['Lens_1'].max = 41
        r = self.system.parallel_after_object(0, method='brute')
        self.assertTrue(np.isclose(r.params['Lens_1'].value, 37.5), 'Wrong point for parallel beam!')

    def test_point(self):
        self.system.add_lens(self.lens_selection, self.lens_position, self.lens_material, reference=1)
        self.system.prepare_objects()
        self.system.parameters['Lens_1'].min = 30
        self.system.parameters['Lens_1'].max = 41
        self.system.parallel_after_object(0, method='nelder')
        self.system.parameters['Lens_1'].vary = False
        self.system.add_lens(self.lens_selection, 150, self.lens_material, reference=2)
        thickness = self.system.objects[0][1].thickness
        self.system.parameters['Lens_2'].min = self.lens_position + thickness
        self.system.add_detector(200, 47.5)
        self.system.parameters['Detector'].vary = False
        self.system.prepare_objects()
        r = self.system.focus_at_object(2, method='nelder')
        self.assertTrue(np.isclose(r.params['Lens_2'].value, 132.67185058158572), 'Wrong focal point!')

if __name__ == '__main__':
    unittest.main()
