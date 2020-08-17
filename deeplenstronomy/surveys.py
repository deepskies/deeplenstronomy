# Configuration Dicts for large astronomical surveys

def des():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time:
            DISTRIBUTION:
                NAME: des_exposure_time
                PARAMETERS: None
        numPix: 100
        pixel_scale: 0.263
        psf_type: 'GAUSSIAN'
        read_noise: 7
        ccd_gain:
            DISTRIBUTION:
                NAME: des_ccd_gain
                PARAMETERS: None
SURVEY:
    PARAMETERS:
        BANDS: g,r,i,z,Y
        seeing:
            DISTRIBUTION:
                NAME: des_seeing
                PARAMETERS: None
        magnitude_zero_point: 30.0
        sky_brightness:
            DISTRIBUTION:
                NAME: des_sky_brightness
                PARAMETERS: None
        num_exposures:
            DISTRIBUTION:
                NAME: des_num_exposures
                PARAMETERS: None
"""
    return info
