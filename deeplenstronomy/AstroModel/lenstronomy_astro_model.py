

class AstroModel(object):
    """
    This class defines the list of astrophysical properties for a single lens
    """

    def __init__(self,Einstein_Radius ):
    Einstein_Radius = 1.
    Velocity_Dispersion = 250.
    Apparent_Magnitude_Lens = [21., 21., 21., 21., 21]
    Absolute_Magnitude_Lens = [21., 21., 21., 21., 21]
    Shear_External = 1.
    Redshift_Lens = 0.5
    Redshift_Source = 1.0
    Lens_position_x = 0.0
    Lens_position_y = 0.0
    Source_position_x = 0.0
    Source_position_y = 0.0
    Source_magnification = 2.0
    Apparent_Magnitude_Source = [21., 21., 21., 21., 21]
    Absolute_Magnitude_Source = [21., 21., 21., 21., 21]
    Position_angle_Lens = 0.0
    Position_angle_Source = 0.0
    Halflight_Radius_Lens = 1.0
    Halflight_Radius_Source = 1.0
    Signal_to_Noise_model = 5.0
    Halo_mass_Lens = 10.
    Halo_mass_Lens_slope = 1.0
    Halo_mass_Lens_core = 1.0 # e.g., NFW
    Cross_section = 1.0

