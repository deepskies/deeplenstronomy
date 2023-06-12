# CONFIGURATION TEMPLATE DICTIONARY 

general_config_template = {
    'DATASET': {
        'NAME': 'value',
        'SIZE': 100,
        'OUTDIR': 'path/to/directory',
        'SEED': 42
    },
    'COSMOLOGY': {
        'H0': 0, 
        'Om0': 0, 
        'Tcmb0': 0, 
        'Neff': 0, 
        'm_nu': 0, 
        'Ob0': 0
    },
    'IMAGE': {
        'EXPOSURE_TIME': 90,
        'NUM_PIX': 100,
        'PIXEL_SCALE': 0.263,
        'PSF_TYPE': 'GAUSSIAN',
        'READ_NOISE': 7,
        'CCD_GAIN': 6.083
    },
    'SURVEY': {
        'BANDS': ['g', 'r', 'i', 'z', 'Y'],
        'SEEING': 0.9,
        'MAGNITUDE_ZERO_POINT': 30.0,
        'SKY_BRIGHTNESS': 23.5,
        'NUM_EXPOSURES': 10
    },

    'SPECIES': {
        'GALAXY_1': {
            'NAME': 'LENS',
            'LIGHT_PROFILE_1':
                {
                    'NAME': 'SERSIC_ELLIPSE',
                    'MAGNITUDE': 19.5,
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'R_sersic': 10,
                    'n_sersic': 4,
                    'e1': 0.2,
                    'e2': -0.1,
                },
            'LIGHT_PROFILE_2': 
                {
                    'NAME': 'SERSIC_ELLIPSE',
                    'DISTRIBUTION':{
                            'NAME': 'uniform',
                            'MINIMUM': 0,
                            'MAXIMUM': 100
                        },
                    'center_x': 0.0,
                    'center_y': 0.0,
                    'R_sersic': 3,
                    'n_sersic': 8,
                    'e1': 0.05,
                    'e2': -0.05,
                },
            'MASS_PROFILE_1':
                {
                    'NAME': 'SIE' ,
                    'theta_E': 2.0,
                    'e1': 0.1,
                    'e2': -0.1,
                    'center_x': 0.0,
                    'center_y': 0.0,
                },
            'SHEAR_PROFILE_1':
                {
                    'NAME': 'SHEAR',
                    'gamma1': 0.08,
                    'gamma2': 0.01,
                }
        },
        'GALAXY_2':
        {
            'NAME': 'SOURCE',
            'LIGHT_PROFILE_1':
            {
                'NAME': 'SERSIC_ELLIPSE',
                'magnitude': 21.5,
                'center_x': 0.0,
                'center_y': 0.0,
                'R_sersic': 6,
                'n_sersic': 5,
                'e1': 0.2,
                'e2': -0.1
            },
            'SHEAR_PROFILE_1':{
                'NAME': 'SHEAR',
                'gamma1': 0.08,
                'gamma2': 0.01, 
            } 
        },              
        'POINTSOURCE_1':{
            'NAME': 'AGN',
            'HOST': 'SOURCE',
            'magnitude': '16',
            },
        'POINTSOURCE_2':{
            'NAME': 'SUPERNOVA',
            'HOST': 'SOURCE',
            'magnitude': 21.0,
            'sep': 2.0,
            'sep_unit': 'arcsec',
            },
        'POINTSOURCE_3':{
            'NAME': 'STAR',
            'HOST': 'Foreground',
            'magnitude': '14.0'
            },
        'NOISE_1':{
            'NAME': 'POISSON_NOISE',
            'mean': 2.0
            }
    },
    'GEOMETRY':{
        'CONFIGURATION_1':{
            'NAME': 'GALAXY_AGN',
            'FRACTION': 0.25,
            'PLANE_1':{
                'OBJECT_1': 'LENS',
                'REDSHIFT': 0.2  
                },                
            'PLANE_2':{
                'OBJECT_1': 'SOURCE',
                'OBJECT_2': 'AGN',
                'REDSHIFT': 0.7     
                },             
            'NOISE_SOURCE_1': 'POISSON_NOISE'
        }
    },
    'DISTRIBUTIONS':{
        'USERDIST_1':{
            'FILENAME': 'distribution_file.txt',
            'MODE': 'interpolate'
            }
        },
    'BACKGROUNDS':{ 
        'PATH': 'path/to/background/image',
        'CONFIGURATIONS': ['CONFIGURATION_1']
    }
}

