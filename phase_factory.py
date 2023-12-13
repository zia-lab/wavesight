#!/users/jlizaraz/anaconda/S4env/bin/python

# ┌──────────────────────────────────────────────────────────┐
# │                                                          │
# │       _   _   _   _   _     _   _   _   _   _   _   _    │
# │      / \ / \ / \ / \ / \   / \ / \ / \ / \ / \ / \ / \   │
# │     ( p | h | a | s | e ) ( f | a | c | t | o | r | y )  │
# │      \_/ \_/ \_/ \_/ \_/   \_/ \_/ \_/ \_/ \_/ \_/ \_/   │
# │                                                          │
# │        This script uses S4 to estimate the phase         │
# │        differences that different sub-wavelength         │
# │      cylindrical posts would produce to a normally       │
# │                   incident plane wave.                   │
# │     When run from the CLI this produces a json file      │
# │                    with the results.                     │
# │     To estimate this a periodic structure with posts     │
# │     of the given geometry is solved through the RCWA     │
# │                  implementation of S4.                   │
# │           Assume that the length units are um.           │
# │                                                          │
# └──────────────────────────────────────────────────────────┘

try:
    import S4
except:
    print("S4 not found, this script must be run from the S4env conda environment.")
import json
import argparse
import numpy as np
from time import time
from printech import *

def meta_field(sim_params):
    '''
    Parameters
    ----------
    sim_params : dict with keys
        'n_refractive' (float): refractive index
        'lattice_const' (float): lattice constant of hex lattice
        'post_width' (float): width of the post
        'numG' (int): number of basis vectors
        's_amp' (float): amplitude of the s-polarized field
        'p_amp' (float): amplitude of the p-polarized field
        'wavelength' (float): equivalent wavelength in vacuum
    
    Returns
    -------
    E_field, H_field : tuple of triples of complex numbers
    '''
    n_refractive                = sim_params['n_refractive']
    numG                        = sim_params['numG']
    (x_coord, y_coord, z_coord) = sim_params['coords']
    lattice_const                  = sim_params['lattice_const']
    post_width                  = sim_params['post_width']
    wavelength                  = sim_params['wavelength']
    post_height                 = sim_params['post_height']
    (s_amp, p_amp)              = sim_params['s_amp'], sim_params['p_amp']
    half_post_width             = post_width/2
    half_cell_width             = lattice_const/2
    excitation_frequency        = 1/wavelength
    epsilon                     = n_refractive**2

    sim = S4.New(((lattice_const,0), (0,lattice_const)), numG)
    sim.AddMaterial("vacuum", 1)
    sim.AddMaterial("substrate", epsilon)
    sim.AddLayer('bottom',
                0,
                'substrate')
    sim.AddLayer('forest',
                post_height,
                'vacuum')
    sim.AddLayer('top',
                0,
                'vacuum')
    sim.SetRegionRectangle(
        'forest',
        'substrate',
        (0,0),
        0,
        (half_post_width, half_cell_width)) 
    sim.SetExcitationPlanewave(
        (0,0),
        s_amp,  
        p_amp 
    )
    sim.SetFrequency(excitation_frequency)
    E_field, H_field = sim.GetFields(x_coord, y_coord, z_coord)

    return E_field, H_field

def convert(arr):
    '''
    Used to convert numpy arrays to lists when saving to json.

    Parameters
    ----------
    arr : np.ndarray
        The array to be converted
    
    Returns
    -------
    arr.tolist() : list
    '''
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    raise TypeError

def meta_fields_hex(sim_params):
    '''
    Parameters
    ----------
    sim_params : dict with keys
        'n_refractive' (float): refractive index
        'post_radius' (float): width of the post
        'numG' (int): number of basis vectors
        's_amp' (float): amplitude of the s-polarized field
        'p_amp' (float): amplitude of the p-polarized field
        'wavelength' (float): wavelength of the light in vacuum
        'lattice_constant' (float): lattice constant of the hexagonal lattice
        'samples' (int): how many samples taken along the fundamental parallelogram
        'zCoord' (float): z-coordinate where the resultant fields will be computed
    
    Returns
    -------
    (E_field, H_field) : tuple of triples of complex numbers
    '''
    lattice_constant = sim_params['lattice_constant']
    post_height = sim_params['post_height']
    post_radius = sim_params['post_radius']
    samples = sim_params['samples']
    zCoord = sim_params['zCoord']
    (half_post_radius, half_lattice_constant)  = post_radius/2, lattice_constant/2
    numG = sim_params['numG']
    (s_amp, p_amp) = sim_params['s_amp'], sim_params['p_amp']
    n = sim_params['n_refractive']
    epsilon = n**2
    wavelength = sim_params['wavelength']
    excitation_frequency = 1/wavelength
    v1_arr = lattice_constant*np.array([1,0])
    v2_arr = lattice_constant*np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    fractions = np.linspace(0,1,samples+1)[:-1]
    v1 = tuple(v1_arr)
    v2 = tuple(v2_arr)
    sim = S4.New((v1, v2), numG)
    sim.AddMaterial("vacuum", 1)
    sim.AddMaterial("substrate", epsilon)
    sim.AddLayer('bottom',
                0,
                'substrate')
    sim.AddLayer('forest',
                post_height,
                'vacuum')
    sim.AddLayer('top',
                0,
                'vacuum')
    sim.SetRegionCircle(
        'forest',
        'substrate',
        (0,0),
        post_radius) 
    sim.SetExcitationPlanewave(
        (0,0),
        s_amp,  
        p_amp 
    )
    sim.SetFrequency(excitation_frequency)

    EH_fields = np.array(sim.GetFieldsOnGrid(zCoord, NumSamples=(samples, samples), Format = 'Array'))
    return EH_fields

def phase_crunch(n_refractive, min_post_width, max_post_width, 
                 num_post_widths,
                 free_wave, lattice_const, post_height,
                 numG, s_amp, p_amp):
    '''
    Parameters
    ----------
    n_refractive : float
        refractive index
    min_post_width : float
        minimum post width
    max_post_width : float
        maximum post width
    num_post_widths : int
        number of post widths to simulate
    free_wave : float
        equivalent wavelength in vacuum
    lattice_const : float
        width of the cell
    post_height : float
        height of the post
    numG : int
        number of basis vectors
    s_amp : float
        amplitude of the s-polarized field
    p_amp : float
        amplitude of the p-polarized field
    
    Returns
    -------
    post_widths, phases : tuple of np.arrays
        post widths and corresponding phases in radians

    '''
    phases = []
    post_widths = np.linspace(min_post_width, max_post_width, num_post_widths)
    info_message = [f'Simulating {num_post_widths} post geometries with diameters in the range [{min_post_width}, {max_post_width}] μm',
                    f'Using a hexagonal cell of lattice constant {lattice_const} μm and post height {post_height} μm',
                    f'Assuming a plane wave with free space wavelength of {free_wave} μm',
                    f'The posts are assumed to be made of a material with refractive index {n_refractive}',
                    f'Using {numG} basis vectors for RCWA',
                    f'Using s-polarized field amplitude of {s_amp} and p-polarized field amplitude of {p_amp}']
    rule()
    for im in info_message:
        printer(">> " + im)
    rule()
    for post_width in post_widths:
        
        sim_params = {
            'wavelength'   : free_wave,
            'lattice_constant'   : lattice_const,
            'post_height'  : post_height,
            'post_radius'  : post_width/2,
            'samples'      : 100,
            'numG'         : numG,
            's_amp'        : s_amp,
            'p_amp'        : p_amp,
            'n_refractive' : n_refractive
        }
        sim_params['zCoord'] = sim_params['post_height']
        fields = meta_fields_hex(sim_params)
        E_field        = fields[0,:,:,:]
        Ex, Ey         = E_field[:,:,0], E_field[:,:,1]
        Exabs, Eyabs = np.max(np.abs(Ex)), np.max(np.abs(Ey))
        if max(Exabs, Eyabs) == Exabs:
            phase = np.angle(np.mean(Ex))
        else:
            phase = np.angle(np.mean(Ey))
        printer("> post_width: %f μm -> phase: %f rad" % (post_width, phase))
        phases.append(phase)
    rule()
    printer('Unwrapping and removing offset in phases')
    phases = np.array(phases)
    phases = np.unwrap(phases)
    phases = phases - phases[0]
    return post_widths, phases

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_refractive',             type=float, default=2.41)
    parser.add_argument('--min_post_width',   type=float, default=0.05)
    parser.add_argument('--max_post_width',   type=float, default=0.2)
    parser.add_argument('--free_wave',        type=float, default=0.532)
    parser.add_argument('--lattice_const',       type=float, default=0.25)
    parser.add_argument('--post_height',      type=float, default=0.6)
    parser.add_argument('--num_post_widths',  type=int,   default=100)
    parser.add_argument('--numG',             type=int,   default=30)
    parser.add_argument('--s_amp',            type=float, default=0)
    parser.add_argument('--p_amp',            type=float, default=1.0)
    args = parser.parse_args()

    start_time = int(time())
    post_widths, phases = phase_crunch(
        args.n_refractive, args.min_post_width, args.max_post_width,
        args.num_post_widths, args.free_wave,
        args.lattice_const, args.post_height,
        args.numG, args.s_amp, args.p_amp)
    end_time = int(time())
    rule()
    printer("Total time: %d seconds" % (end_time - start_time))
    rule()
    created_on = int(start_time)
    results = {
        'numG': args.numG,
        's_amp': args.s_amp,
        'p_amp': args.p_amp,
        'created_on': created_on,
        'post_widths': post_widths,
        'free_wave': args.free_wave,
        'lattice_const': args.lattice_const,
        'phases': phases,
        'post_height': args.post_height,
        'n_refractive': args.n_refractive,
        'num_post_widths': args.num_post_widths}
    json_timestamp = int(time()*1e7)
    json_fname = './json_out/phase_data-%d.json' % (json_timestamp)
    results['json_fname'] = json_fname
    with open(json_fname, 'w') as f:
        printer("Saving results to %s" % json_fname)
        json.dump(results, f, default=convert, indent=4, sort_keys=True)
    rule()
    # this final print is necessary for being able to retrieve the results
    print(json_fname)
