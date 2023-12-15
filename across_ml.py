#!/users/jlizaraz/anaconda/meep/bin/python

import os
import time
import h5py
import argparse
import meep as mp
import numpy as np
import wavesight as ws
from printech import *
from scipy.interpolate import RegularGridInterpolator

data_dir        = '/users/jlizaraz/data/jlizaraz/CEM/wavesight'
exclude_dirs    = ['moovies']
save_cross_eps  = True

def metasurf_designer(metalens_design_dict, post_z_center):
    '''
    Given a dictionary with the design parameters for a metalens, this function
    will return a list of MEEP objects that represent the metalens structure.

    Parameters
    ----------
    metalens_design_dict : dict
        A dictionary with the design parameters for the metalens.
    post_z_center : float
        The z coordinate of the center of the metalens pillars.
    
    Returns
    -------
    post_forest : list
        A list of MEEP objects that represent the cylindrical pillars that make
        up the metalens.
    '''
    post_radii = metalens_design_dict['post_radii']
    post_height = metalens_design_dict['post_height']
    lattice_points = metalens_design_dict['lattice_points']
    post_forest = []
    nPosts = metalens_design_dict['nHost']
    postMedium = mp.Medium(index = nPosts)
    for lattice_point, post_radius in zip(lattice_points, post_radii):
        post_location = mp.Vector3(lattice_point[0], lattice_point[1], post_z_center)
        a_post = mp.Cylinder(radius = post_radius,
                             height = post_height,
                             center = post_location,
                             material = postMedium)
        post_forest.append(a_post)
    return post_forest

def wave_sorter(x):
    '''
    A convenience function to sort directories accoring to their
    filename.
    '''
    idx = int(x.split('-')[-1])
    return idx

def meta_duct(waveguide_id, mode_idx):
    '''
    This function will take the id for a waveguide and the index for
    one of its propagating modes, and it will take the fields that have
    been propagated to the input plane of the metalens and will produce
    the fields that are transported across it.
    '''
    # first find all the subdirs that correspond to the waveguide_id
    waveguide_dir = os.path.join(data_dir, waveguide_id)
    job_dir_contents = os.listdir(waveguide_dir)
    job_dir_contents = [a_dir for a_dir in job_dir_contents if a_dir not in exclude_dirs]
    job_dirs = [os.path.join(waveguide_dir, a_dir) for a_dir in job_dir_contents]
    expanded_params_fname = os.path.join(waveguide_dir,'config.json')
    params_dict = ws.load_from_json(expanded_params_fname)
    nHost           = params_dict['nHost']
    nBetween        = params_dict['nBetween']
    MEEP_resolution = params_dict['MEEP_resolution']
    EH3_to_ml       = params_dict['EH3_to_ml']
    ml_to_EH4       = params_dict['ml_to_EH4']
    ml_thickness    = params_dict['ml_thickness']
    parallel_MEEP   = params_dict['parallel_MEEP']
    full_ml_sim_height = params_dict['full_ml_sim_height']
    if parallel_MEEP:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank
    else:
        # if not running parallel same as if running in serial
        rank = 0

    job_dirs = [a_dir for a_dir in job_dirs if os.path.isdir(a_dir) ]
    job_dirs = list(sorted(job_dirs, key = wave_sorter))
    mode_to_dir = {jdir.split('-')[-1]: jdir for jdir in job_dirs}
    if str(mode_idx) not in mode_to_dir:
        raise Exception("Invalid mode_idx")
    job_dir = mode_to_dir[str(mode_idx)]

    propagated_h5_fname = 'EH3.h5'
    propagated_h5_fname = os.path.join(job_dir, propagated_h5_fname)
    refracted_h5_fname  = 'EH2.h5'
    refracted_h5_fname  = os.path.join(job_dir, refracted_h5_fname)

    (metadata_dict, created_on)    = ws.load_from_h5(refracted_h5_fname)
    (prop_fields_dict, created_on) = ws.load_from_h5(propagated_h5_fname)
    incident_EH_field              = prop_fields_dict.pop('EH_field')
    electric_J, magnetic_K         = ws.from_sampled_field_to_sampled_equiv_currents(incident_EH_field)

    pml_thickness = metadata_dict['pml_thickness']
    xCoords       = prop_fields_dict['xCoords']
    yCoords       = prop_fields_dict['yCoords']
    printer("retrieving predetermined thickness of the PML of %.1f um" % pml_thickness)
    field_width    = xCoords[-1] - xCoords[0]
    full_sim_width = field_width + 2 * pml_thickness

    printer("setting up the MEEP media")
    between_medium = mp.Medium(index = nBetween)
    host_medium    = mp.Medium(index = nHost)
    ml_medium      = mp.Medium(index = nHost)

    printer("padding the incident field with zeros to account for the full width of the simulation cross section")

    N_samples                = len(xCoords)
    (new_sim_width, xCoords) = ws.array_stretcher(xCoords, full_sim_width)
    (new_sim_width, yCoords) = ws.array_stretcher(yCoords, full_sim_width)
    full_sim_width           = new_sim_width
    total_pad_width          = len(xCoords) - N_samples
    electric_J               = ws.sym_pad(electric_J, total_pad_width)
    magnetic_K               = ws.sym_pad(magnetic_K, total_pad_width)

    runway_cell_thickness = params_dict['runway_cell_thickness']
    ml_cell_thickness     = params_dict['ml_cell_thickness']
    host_cell_thickness   = params_dict['host_cell_thickness']

    runway_cell_z_center = (-full_ml_sim_height/2
                            + runway_cell_thickness/2)

    ml_cell_z_center = (-full_ml_sim_height/2
                + runway_cell_thickness
                + ml_thickness/2)

    host_cell_z_center = (-full_ml_sim_height/2
                + runway_cell_thickness
                + ml_thickness
                + host_cell_thickness/2.)

    printer("setting up the simulation geometry")
    host_cell = mp.Vector3(full_sim_width, full_sim_width, host_cell_thickness)
    host_cell_center = mp.Vector3(0, 0, host_cell_z_center)

    ml_cell   = mp.Vector3(full_sim_width, full_sim_width, ml_thickness)
    ml_cell_center = mp.Vector3(0, 0, ml_cell_z_center)

    runway_cell = mp.Vector3(full_sim_width, full_sim_width, runway_cell_thickness)
    runway_cell_center = mp.Vector3(0, 0, runway_cell_z_center)

    geometry = [mp.Block(size     = host_cell,
                         center   = host_cell_center,
                         material = host_medium),
                mp.Block(size     = runway_cell,
                         center   = runway_cell_center,
                         material = between_medium)
                            ]
    printer("adding the metasurface geometry")
    metalens_desing_fname = os.path.join(waveguide_dir, 'metalens-design.h5')
    (metalens_dict, _) = ws.load_from_h5(metalens_desing_fname)
    geometry = geometry + metasurf_designer(metalens_dict, ml_cell_z_center)

    Jxf = RegularGridInterpolator((xCoords,yCoords), electric_J[0], bounds_error=False, fill_value=0.0)
    Jyf = RegularGridInterpolator((xCoords,yCoords), electric_J[1], bounds_error=False, fill_value=0.0)
    Kxf = RegularGridInterpolator((xCoords,yCoords), magnetic_K[0], bounds_error=False, fill_value=0.0)
    Kyf = RegularGridInterpolator((xCoords,yCoords), magnetic_K[1], bounds_error=False, fill_value=0.0)

    def Jx(vec):
        return complex(Jxf((vec.x, vec.y)))

    def Jy(vec):
        return complex(Jyf((vec.x, vec.y)))

    def Kx(vec):
        return complex(Kxf((vec.x, vec.y)))

    def Ky(vec):
        return complex(Kyf((vec.x, vec.y)))

    printer("setting up the simulation cell")
    sim_cell = mp.Vector3(full_sim_width, full_sim_width, full_ml_sim_height)

    printer("setting up the PML layers")
    pml_layers  = [mp.PML(pml_thickness)]

    kFree = metadata_dict['kFree']
    base_period     = 1. / kFree
    run_time        = params_dict['run_time_2']
    source_time     = run_time

    source_z = (-full_ml_sim_height/2
                + pml_thickness
                + EH3_to_ml)

    source_center = mp.Vector3(0,0, source_z)
    source_size   = mp.Vector3(full_sim_width, full_sim_width, 0)

    printer("setting up the sources for the input field")
    srcs = []
    source_fun = mp.ContinuousSource(frequency=kFree/2/np.pi,
                                        width=2*base_period,
                                        end_time=source_time)
    for pair in [((mp.Ex, mp.Ey), (Jx, Jy)),
                 ((mp.Hx, mp.Hy), (Kx, Ky))]:
        for field_component, current_fun in zip(pair[0], pair[1]):
            src = mp.Source(src = source_fun,
                            component = field_component,
                            center = source_center,
                            size = source_size,
                            amp_func = current_fun
                            )
            srcs.append(src)

    xy_mon_z = (-full_ml_sim_height/2
                + pml_thickness
                + 2*EH3_to_ml
                + ml_thickness
                + ml_to_EH4)

    printer("setting up the monitor planes")
    clear_aperture = full_sim_width - 2 * pml_thickness

    xy_monitor_plane_center = mp.Vector3(0, 0, xy_mon_z)
    xy_monitor_plane_size   = mp.Vector3(clear_aperture, clear_aperture, 0)
    xy_monitor_vol          = mp.Volume(center=xy_monitor_plane_center, size=xy_monitor_plane_size)

    xz_monitor_plane_center = mp.Vector3(0,0,0)
    xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, full_ml_sim_height)
    xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center,
                                        size=xz_monitor_plane_size)

    yz_monitor_plane_center = mp.Vector3(0,0,0)
    yz_monitor_plane_size   = mp.Vector3(0, clear_aperture, full_ml_sim_height)
    yz_monitor_vol          = mp.Volume(center=yz_monitor_plane_center,
                                        size=yz_monitor_plane_size)

    approx_runtime = ws.approx_time(sim_cell, MEEP_resolution, run_time)
    printer("simulation is estimated to take %.1f minutes" % (approx_runtime/60.))

    printer("finalizing the setup fo the simulation object")
    sim = mp.Simulation(
        cell_size  = sim_cell,
        geometry   = geometry,
        sources    = srcs,
        resolution = MEEP_resolution,
        boundary_layers      = pml_layers,
        force_complex_fields = True
    )

    start_time = time.time()
    sim.run(until=run_time)
    end_time = time.time()
    time_taken = end_time - start_time

    printer("getting final coordinates used by MEEP")
    coords = {}
    (xCoords, yCoords, zCoords, _) = sim.get_array_metadata()
    numMVx = len(xCoords) * len(yCoords) * len(zCoords)
    (xCoordsMonxy, yCoordsMonxy, zCoordsMonxy, _) = sim.get_array_metadata(xy_monitor_vol)
    (xCoordsMonxz, yCoordsMonxz, zCoordsMonxz, _) = sim.get_array_metadata(xz_monitor_vol)
    (xCoordsMonyz, yCoordsMonyz, zCoordsMonyz, _) = sim.get_array_metadata(yz_monitor_vol)
    coords['xCoords'] = xCoords
    coords['yCoords'] = yCoords
    coords['zCoords'] = zCoords
    coords['xCoordsMonxy'] = xCoordsMonxy
    coords['yCoordsMonxy'] = yCoordsMonxy
    coords['zCoordsMonxy'] = zCoordsMonxy
    coords['xCoordsMonxz'] = xCoordsMonxz
    coords['yCoordsMonxz'] = yCoordsMonxz
    coords['zCoordsMonxz'] = zCoordsMonxz
    coords['xCoordsMonyz'] = xCoordsMonyz
    coords['yCoordsMonyz'] = yCoordsMonyz
    coords['zCoordsMonyz'] = zCoordsMonyz

    # grab the transmitted field

    on_axis_eps = sim.get_array(mp.Dielectric,
                    mp.Volume(center = mp.Vector3(0,0,0),
                                size = mp.Vector3(0,0,full_ml_sim_height))
                    )
    ehfieldh5fname = os.path.join(job_dir, 'EH4.h5')
    if save_cross_eps:
        cross_eps = sim.get_array(mp.Dielectric,
                    mp.Volume(center = mp.Vector3(0,0,ml_cell_z_center),
                                size = mp.Vector3(clear_aperture,clear_aperture,0))
                    )
    monitor_vols = {'xy': xy_monitor_vol,'yz': yz_monitor_vol,'xz': xz_monitor_vol}
    h5_fname = ehfieldh5fname
    export_dict = {}
    export_dict['on_axis_eps'] = on_axis_eps
    for key, val in coords.items():
        export_dict[key] = val
    if save_cross_eps:
        export_dict['cross_eps_at_ml'] = cross_eps
    for monitor_plane in 'xz yz xy'.split(' '):
        a_monitor_fields = []
        for field_component in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]:
            the_field = sim.get_array(field_component, monitor_vols[monitor_plane])
            a_monitor_fields.append(the_field)
        a_monitor_fields = np.array(a_monitor_fields)
        for field_idx, field_component in enumerate('ex ey ez hx hy hz'.split(' ')):
            for complex_part, complex_fun in zip(['i','r'], [np.imag, np.real]):
                key = '/%s/%s.%s' % (monitor_plane, field_component, complex_part)
                export_field = complex_fun(a_monitor_fields[field_idx])
                export_dict[key] = export_field
    if rank == 0:
        ws.save_to_h5(export_dict, h5_fname, overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transport fields across the metasurface.')
    parser.add_argument('waveguide_id', type=str, help='The label for the waveguide.')
    parser.add_argument('mode_idx', type=str, help='The label for the waveguide.')
    args = parser.parse_args()
    meta_duct(args.waveguide_id, args.mode_idx)
