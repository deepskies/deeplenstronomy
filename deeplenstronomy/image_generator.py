"""Generate images from the organized user inputs."""

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.LensModel.Solver import lens_equation_solver
import numpy as np

import deeplenstronomy.distributions as distributions

from deeplenstronomy.utils import dict_select, dict_select_choose, select_params


class ImageGenerator():
    def __init__(self, return_planes=False, solve_lens_equation=False):
        """
        This is an internal class which calls lenstronomy functions based on parsed user inputs.
        
        Args:
            return_planes (bool): Automatically passed from deeplenstronomy.make_dataset args
            solve_lens_equation (bool): Automatically passed from deeplenstronomy.make_dataset args

        """
        self.return_planes = return_planes
        self.solve_lens_equation = solve_lens_equation
        return
    
    def sim_image(self, info_dict):
        """
        Simulate an image based on specifications in sim_dict
        
        Args:
            info_dict (dict): A single element from the list produced interanlly by input_reader.Organizer.breakup(). 
                Contains all the properties of a single image to generate.
        """
        output_image = []
        if self.return_planes:
            output_source, output_lens, output_point_source, output_noise = [], [], [], []
        output_metadata = []
        
        #set the cosmology
        cosmology_info = ['H0', 'Om0', 'Tcmb0', 'Neff', 'm_nu', 'Ob0']
        cosmo = FlatLambdaCDM(**dict_select_choose(list(info_dict.values())[0], cosmology_info))
        
        for band, sim_dict in info_dict.items():
            ### Geometry-independent image properties
            
            # Single Band Info
            single_band_info = ['read_noise', 'pixel_scale', 'ccd_gain', 'exposure_time',
                                'sky_brightness', 'seeing', 'magnitude_zero_point',
                                'num_exposures', 'data_count_unit', 'background_noise']
            kwargs_single_band = dict_select_choose(sim_dict, single_band_info)
            
            # Extinction
            extinction_class = None  #figure this out later
            
            # Numerics -- only use single number kwargs
            numerics_info = ['supersampling_factor', 'compute_mode', 'supersampling_convolution',
                             'supersampling_kernel_size', 'point_source_supersampling_factor', 
                             'convolution_kernel_size']
            kwargs_numerics = dict_select_choose(sim_dict, numerics_info)
            
            ### Geometry-dependent image properties
            
            # Dictionary for physical model
            kwargs_model = {'lens_model_list': [],  # list of lens models to be used
                            'lens_redshift_list': [],  # list of redshift of the deflections
                            'lens_light_model_list': [],  # list of unlensed light models to be used
                            'source_light_model_list': [],  # list of extended source models to be used
                            'source_redshift_list': [],  # list of redshfits of the sources in same order as source_light_model_list
                            'point_source_model_list': [], # list of point source models
                            'cosmo': cosmo,
                            'z_source': 0.0}

            # Lists for model kwargs
            kwargs_source_list, kwargs_lens_model_list, kwargs_lens_light_list, kwargs_point_source_list = [], [], [], []
            
            for plane_num in range(1, sim_dict['NUMBER_OF_PLANES'] + 1):
                
                for obj_num in range(1, sim_dict['PLANE_{0}-NUMBER_OF_OBJECTS'.format(plane_num)] + 1):
                    
                    prefix = 'PLANE_{0}-OBJECT_{1}-'.format(plane_num, obj_num)
                    
                    ### Point sources
                    if sim_dict[prefix + 'HOST'] != 'None':
                        
                        # Foreground objects
                        if sim_dict[prefix + 'HOST'] == 'Foreground':
                            kwargs_model['point_source_model_list'].append('UNLENSED')
                            ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra_image', 'dec_image', 'magnitude']]
                            ps_dict_info = dict_select(sim_dict, ps_info)
                            kwargs_point_source_list.append({'ra_image': [ps_dict_info[prefix + 'ra_image']], 'dec_image': [ps_dict_info[prefix + 'dec_image']], 'magnitude': [ps_dict_info[prefix + 'magnitude']]})
                        # Real point sources
                        else:
                            if plane_num < sim_dict['NUMBER_OF_PLANES']:
                                # point sources in the lens
                                kwargs_model['point_source_model_list'].append('LENSED_POSITION')
                                ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra', 'dec', 'magnitude']]
                                ps_dict_info = dict_select(sim_dict, ps_info)
                                kwargs_point_source_list.append({'ra_image': [ps_dict_info[prefix + 'ra']],
                                                                 'dec_image': [ps_dict_info[prefix + 'dec']],
                                                                 'magnitude': [ps_dict_info[prefix + 'magnitude']]})
                            elif plane_num == sim_dict['NUMBER_OF_PLANES']:
                                # point sources in the source plane
                                kwargs_model['point_source_model_list'].append('SOURCE_POSITION')
                                ps_info = ['PLANE_{0}-OBJECT_{1}-{2}'.format(plane_num, obj_num, x) for x in ['ra', 'dec', 'magnitude']]
                                ps_dict_info = dict_select(sim_dict, ps_info)
                                kwargs_point_source_list.append({'ra_source': ps_dict_info['PLANE_{0}-OBJECT_{1}-ra'.format(plane_num, obj_num)], 
                                                                 'dec_source': ps_dict_info['PLANE_{0}-OBJECT_{1}-dec'.format(plane_num, obj_num)], 
                                                                 'magnitude': ps_dict_info['PLANE_{0}-OBJECT_{1}-magnitude'.format(plane_num, obj_num)]})
                            else:
                                #should never get here
                                assert False
                                
                        # the number of profiles will be zero for point sources, so just skip ahead
                        continue
                    
                    ### Model keywords
                    # All planes except last one - treat as lens
                    if plane_num < sim_dict['NUMBER_OF_PLANES']:
                        for light_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_LIGHT_PROFILES'] +1):
                            kwargs_model['lens_light_model_list'].append(sim_dict[prefix + 'LIGHT_PROFILE_{0}-NAME'.format(light_profile_num)])
                            #kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_lens_light_list.append(select_params(sim_dict, prefix + 'LIGHT_PROFILE_{0}-'.format(light_profile_num)))
                        for mass_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_MASS_PROFILES'] +1):
                            kwargs_model['lens_model_list'].append(sim_dict[prefix + 'MASS_PROFILE_{0}-NAME'.format(mass_profile_num)])
                            kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            mass_params = select_params(sim_dict, prefix + 'MASS_PROFILE_{0}-'.format(mass_profile_num))
                            kwargs_lens_model_list.append(mass_params)
                            if 'sigma_v' in mass_params.keys(): # save simga_v locations so that we can calculate theta_E for the metadata
                                output_metadata.append({'PARAM_NAME':  prefix + 'MASS_PROFILE_{0}-sigma_v-{1}'.format(mass_profile_num, band),
                                                        'PARAM_VALUE': mass_params['sigma_v'],
                                                        'LENS_MODEL_IDX': len(kwargs_lens_model_list) - 1})
                        for shear_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_SHEAR_PROFILES'] +1):
                            kwargs_model['lens_model_list'].append(sim_dict[prefix + 'SHEAR_PROFILE_{0}-NAME'.format(shear_profile_num)])
                            kwargs_model['lens_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            mass_params = select_params(sim_dict, prefix + 'SHEAR_PROFILE_{0}-'.format(shear_profile_num))
                            kwargs_lens_model_list.append(select_params(sim_dict, prefix + 'SHEAR_PROFILE_{0}-'.format(shear_profile_num)))
                                                   
                    # Last Plane - treat as source
                    elif plane_num == sim_dict['NUMBER_OF_PLANES']:
                        kwargs_model['z_source'] = sim_dict[prefix + 'REDSHIFT']
                        for light_profile_num in range(1, sim_dict[prefix + 'NUMBER_OF_LIGHT_PROFILES'] +1):
                            kwargs_model['source_light_model_list'].append(sim_dict[prefix + 'LIGHT_PROFILE_{0}-NAME'.format(light_profile_num)])
                            kwargs_model['source_redshift_list'].append(sim_dict[prefix + 'REDSHIFT'])
                            kwargs_source_list.append(select_params(sim_dict, prefix + 'LIGHT_PROFILE_{0}-'.format(light_profile_num)))
                        
                    else:
                        # Should never get here
                        assert False

            # Make image
            sim = SimAPI(numpix=sim_dict['numPix'], 
                         kwargs_single_band=kwargs_single_band, 
                         kwargs_model=kwargs_model)

            imSim = sim.image_model_class(kwargs_numerics)
            
            kwargs_lens_light_list, kwargs_source_list, kwargs_point_source_list = sim.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light_list,
                                                                                                           kwargs_source_mag=kwargs_source_list,
                                                                                                           kwargs_ps_mag=kwargs_point_source_list)
            
            kwargs_lens_model_list = sim.physical2lensing_conversion(kwargs_mass=kwargs_lens_model_list)
            # Save theta_E (and sigma_v if used)
            for ii in range(len(output_metadata)):
                output_metadata.append({'PARAM_NAME': output_metadata[ii]['PARAM_NAME'].replace('sigma_v', 'theta_E'),
                                        'PARAM_VALUE': kwargs_lens_model_list[output_metadata[ii]['LENS_MODEL_IDX']]['theta_E'],
                                        'LENS_MODEL_IDX': output_metadata[ii]['LENS_MODEL_IDX']})
                
            try:
                image = imSim.image(kwargs_lens=kwargs_lens_model_list,
                                    kwargs_lens_light=kwargs_lens_light_list,
                                    kwargs_source=kwargs_source_list,
                                    kwargs_ps=kwargs_point_source_list)
            except Exception:
                # Some sort of lenstronomy error
                print("kwargs_numerics", kwargs_numerics)
                print("kwargs_single_band", kwargs_single_band)
                print("kwargs_model", kwargs_model)
                print("kwargs_lens_model_list", kwargs_lens_model_list)
                print("kwargs_lens_light_list", kwargs_lens_light_list)
                print("kwargs_source_list", kwargs_source_list)
                print("kwargs_point_source_list", kwargs_point_source_list)
                assert False
                                
            # Solve lens equation if desired
            if self.solve_lens_equation:
                solver = lens_equation_solver.LensEquationSolver(imSim.LensModel)
                x_mins, y_mins = solver.image_position_from_source(sourcePos_x=kwargs_source_list[0]['center_x'],
                                                                   sourcePos_y=kwargs_source_list[0]['center_y'],
                                                                   kwargs_lens=kwargs_lens_model_list)
                num_source_images = len(x_mins)
            
            # Add noise
            image_noise = np.zeros(np.shape(image))
            for noise_source_num in range(1, sim_dict['NUMBER_OF_NOISE_SOURCES'] + 1):
                image_noise += self._generate_noise(sim_dict['NOISE_SOURCE_{0}-NAME'.format(noise_source_num)],
                                                    np.shape(image),
                                                    select_params(sim_dict, 'NOISE_SOURCE_{0}-'.format(noise_source_num)))
            image += image_noise
                
            # Combine with other bands
            output_image.append(image)

            # Store plane-separated info if requested
            if self.return_planes:
                output_lens.append(imSim.lens_surface_brightness(kwargs_lens_light_list))
                output_source.append(imSim.source_surface_brightness(kwargs_source_list, kwargs_lens_model_list))
                output_point_source.append(imSim.point_source(kwargs_point_source_list, kwargs_lens_model_list))
                output_noise.append(image_noise)
        
        # Return the desired information in a dictionary
        return_dict = {'output_image': np.array(output_image),
                       'output_lens_plane': None,
                       'output_source_plane': None,
                       'output_point_source_plane': None,
                       'output_noise_plane': None,
                       'x_mins': None,
                       'y_mins': None,
                       'num_source_images': None,
                       'additional_metadata': output_metadata}
        if self.return_planes:
            return_dict['output_lens_plane'] = np.array(output_lens)
            return_dict['output_source_plane'] = np.array(output_source)
            return_dict['output_point_source_plane'] = np.array(output_point_source)
            return_dict['output_noise_plane'] = np.array(output_noise)
        if self.solve_lens_equation:
            return_dict['x_mins'] = x_mins
            return_dict['y_mins'] = y_mins
            return_dict['num_source_images'] = num_source_images

        return return_dict

    def _generate_noise(self, name, shape, params):
        """
        Add noise to image based on input yaml by targeting specified distribution.
        
        :param name: name of the distribution to target
        :param shape: shape of image to add noise to
        :param params: dictionary of additional parameters needed by distributions.name()
        :return: noise_image: noise from targeted distribution for the image
        """
        return eval('distributions.{0}(shape, **params)'.format(name.lower()))

    

                        
