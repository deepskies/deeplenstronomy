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
        magnitude_zero_point:
            DISTRIBUTION:
                NAME: des_magnitude_zero_point
                PARAMETERS: None
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

def delve():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time:
            DISTRIBUTION:
                NAME: delve_exposure_time
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
        BANDS: g,r,i,z
        seeing:
            DISTRIBUTION:
                NAME: delve_seeing
                PARAMETERS: None
        magnitude_zero_point:
            DISTRIBUTION:
                NAME: delve_magnitude_zero_point
                PARAMETERS: None
        sky_brightness:
            DISTRIBUTION:
                NAME: delve_sky_brightness
                PARAMETERS: None
        num_exposures: 1
"""
    return info

def lsst():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time:
            DISTRIBUTION:
                NAME: lsst_exposure_time
                PARAMETERS: None
        numPix: 100
        pixel_scale: 0.2
        psf_type: 'GAUSSIAN'
        read_noise: 10
        ccd_gain: 2.3
SURVEY:
    PARAMETERS:
        BANDS: u,g,r,i,z,Y
        seeing:
            DISTRIBUTION:
                NAME: lsst_seeing
                PARAMETERS: None
        magnitude_zero_point:
            DISTRIBUTION:
                NAME: lsst_magnitude_zero_point
                PARAMETERS: None
        sky_brightness:
            DISTRIBUTION:
                NAME: lsst_sky_brightness
                PARAMETERS: None
        num_exposures:
            DISTRIBUTION:
                NAME: lsst_num_exposures
                PARAMETERS: None
"""
    return info

def hst():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time: 5400.0
        numPix: 100
        pixel_scale: 0.08
        psf_type: 'GAUSSIAN'
        read_noise: 4
        ccd_gain: 2.5
SURVEY:
    PARAMETERS:
        BANDS: F160W
        seeing: 0.08
        magnitude_zero_point: 25.96
        sky_brightness: 22.3
        num_exposures: 1
"""
    return info

def euclid():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time: 565.0
        numPix: 100
        pixel_scale: 0.101
        psf_type: 'GAUSSIAN'
        read_noise: 4.2
        ccd_gain: 3.1
SURVEY:
    PARAMETERS:
        BANDS: VIS
        seeing: 0.16
        magnitude_zero_point: 24.0
        sky_brightness: 22.35
        num_exposures: 4
"""
    return info

def ztf():
    info = """
IMAGE:
    PARAMETERS:
        exposure_time: 30.0
        numPix: 100
        pixel_scale: 1.01
        psf_type: 'GAUSSIAN'
        read_noise: 10.3
        ccd_gain: 5.8
SURVEY:
    PARAMETERS:
        BANDS: g,r,i
        seeing:
            DISTRIBUTION:
                NAME: ztf_seeing
                PARAMETERS: None
        magnitude_zero_point:
            DISTRIBUTION:
                NAME: ztf_magnitude_zero_point
                PARAMETERS: None
        sky_brightness: 
            DISTRIBUTION:
                NAME: ztf_sky_brightness
                PARAMETERS: None
        num_exposures: 24
"""
    return info
