#!/usr/bin/env python3

import numpy as np

def expand_sim_params(sim_params):
    '''
    Compute additional parameters for the S4 simulations.
    '''
    sim_params['field_height']         = sim_params['post_height']
    sim_params['half_cell_width']      = sim_params['cell_width'] / 2.
    sim_params['half_post_width']      = sim_params['post_width'] / 2.
    sim_params['excitation_frequency'] = 1. / sim_params['wavelength']

    return sim_params

def hex_lattice_points(lattice_constant, radius):
    '''
    Given the lattice constant of a hexagonal lattice,
    this function returns as many lattice points as fit
    in a circle of the given radius.
    Parameters
    ----------
    lattice_constant: float
        The lattice constant of the hex lattice.
    radius: float
        The radius of the circle.
    Returns
    -------
    lattice_points: np.ndarray
        The lattice points (x,y) within the circle.
    '''
    v1 = lattice_constant*np.array([1, 0])
    v2 = lattice_constant*np.array([np.cos(2 * np.pi / 3),
                                    np.sin(2 * np.pi / 3)])
    reps = int(2.5 * radius / lattice_constant)
    half_reps = reps // 2
    lattice_points = np.array([i*v1 + j*v2 
                               for i in range(-half_reps, half_reps+1) 
                               for j in range(-half_reps, half_reps+1)])
    lattice_points = lattice_points[np.sum(lattice_points**2, axis=1) < radius**2]
    return lattice_points

def fresnel_fun_gen(focal_length, wavelength, mod_2pi=True):
    '''
    Parameters
    ----------
    focal_length: float
        The focal length of the lens in μm.
    wavelength: float
        The free space wavelength of the light in μm.
    Returns
    -------
    fresnel_fun: function
        A function that takes x and y coordinates and returns
        the phase of the light at that point.
    '''
    if mod_2pi:
        def fresnel_fun(x,y):
            r = np.sqrt(x**2+y**2)
            return np.mod((2 * np.pi / wavelength) * (focal_length - np.sqrt(focal_length**2 + r**2)), 2*np.pi)
    else:
        def fresnel_fun(x,y):
            r = np.sqrt(x**2+y**2)
            return (2 * np.pi / wavelength) * (focal_length - np.sqrt(focal_length**2 + r**2))
    return fresnel_fun

