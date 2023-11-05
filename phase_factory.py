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
    exit()
import json
import argparse
import numpy as np
from time import time

printer_width = 0

def meta_field(sim_params):
    '''
    Parameters
    ----------
    sim_params : dict with keys
        'n_refractive' (float): refractive index
        'cell_width' (float): width of the cell
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
    cell_width                  = sim_params['cell_width']
    post_width                  = sim_params['post_width']
    wavelength                  = sim_params['wavelength']
    post_height                 = sim_params['post_height']
    (s_amp, p_amp)              = sim_params['s_amp'], sim_params['p_amp']
    half_post_width             = post_width/2
    half_cell_width             = cell_width/2
    excitation_frequency        = 1/wavelength
    epsilon                     = n_refractive**2

    sim = S4.New(((cell_width,0), (0,cell_width)), numG)
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

def convert(o):
    ''''
    Used to convert numpy arrays to lists when saving to json.
    '''
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError

def phase_crunch(n_refractive, min_post_width, max_post_width, 
                 num_post_widths,
                 free_wave, cell_width, post_height,
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
    cell_width : float
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
    post_widths, scaled_phases : tuple of np.arrays
        post widths and corresponding phases in radians

    '''
    global printer_width
    phases = []
    post_widths = np.linspace(min_post_width, max_post_width, num_post_widths)
    info_message = [f'Simulating {num_post_widths} post geometries with diameters in the range [{min_post_width}, {max_post_width}] μm',
                    f'Using a hexagonal cell of lattice constant {cell_width} μm and post height {post_height} μm.',
                    f'Assuming a plane wave with free space wavelength of {free_wave} μm.',
                    f'The posts are assumed to be made of a material with refractive index {n_refractive}.',
                    f'Using {numG} basis vectors.',
                    f'Using s-polarized field amplitude of {s_amp} and p-polarized field amplitude of {p_amp}.']
    printer_width = max([len(im) for im in info_message])
    printer_width += 4
    print('-'*(printer_width))
    for im in info_message:
        print(">> ", im)
    print('-'*(printer_width))
    for post_width in post_widths:
        print("> post_width: %f μm" % post_width, end=' -> ')
        sim_params = {
        'wavelength'   : free_wave,
        'cell_width'   : cell_width,
        'post_height'  : post_height,
        'post_width'   : post_width,
        'numG'         : numG,
        's_amp'        : s_amp,
        'p_amp'        : p_amp,
        'n_refractive' : n_refractive
        }
        sim_params['coords'] = (0, 0, sim_params['post_height'])
        E_field, _ = meta_field(sim_params)
        Ex, Ey, Ez = E_field
        phase = np.arctan2(np.imag(Ex), np.real(Ex))
        print("phase: %f rad" % phase)
        phases.append(phase)
    print('-'*(printer_width))
    print('Unwrapping and removing offset...')
    phases = np.array(phases)
    phases = np.unwrap(phases)
    phases = phases - phases[0]
    scaled_phases =  phases
    return post_widths, scaled_phases

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_refractive',             type=float, default=2.41)
    parser.add_argument('--min_post_width',   type=float, default=0.05)
    parser.add_argument('--max_post_width',   type=float, default=0.2)
    parser.add_argument('--free_wave',        type=float, default=0.532)
    parser.add_argument('--cell_width',       type=float, default=0.25)
    parser.add_argument('--post_height',      type=float, default=0.6)
    parser.add_argument('--num_post_widths',  type=int,   default=100)
    parser.add_argument('--numG',             type=int,   default=30)
    parser.add_argument('--s_amp',            type=float, default=0)
    parser.add_argument('--p_amp',            type=float, default=1.0)
    parser.add_argument('--save_to_file',     type=bool,  default=True)
    args = parser.parse_args()

    start_time = int(time())
    post_widths, scaled_phases = phase_crunch(
        args.n_refractive, args.min_post_width, args.max_post_width,
        args.num_post_widths, args.free_wave,
        args.cell_width, args.post_height,
        args.numG, args.s_amp, args.p_amp)
    end_time = int(time())
    print('-'*(printer_width))
    print("Total time: %d seconds" % (end_time - start_time))
    print('-'*(printer_width))
    created_on = int(start_time)
    if args.save_to_file:
        results = {
            'numG': args.numG,
            's_amp': args.s_amp,
            'p_amp': args.p_amp,
            'created_on': created_on,
            'post_widths': post_widths,
            'free_wave': args.free_wave,
            'cell_width': args.cell_width,
            'scaled_phases': scaled_phases,
            'post_height': args.post_height,
            'n_refractive': args.n_refractive,
            'num_post_widths': args.num_post_widths}
        json_fname = 'phase_data-%d.json' % (created_on)
        with open(json_fname, 'w') as f:
            print("Saving results to %s" % json_fname)
            json.dump(results, f, default=convert, indent=4, sort_keys=True)
        print('-'*(printer_width))
    