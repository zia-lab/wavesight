#!/usr/bin/env python3

import os
import h5py
import time
import pickle
import psutil
import argparse
import meep as mp
import numpy as np
import cmasher as cm
from math import ceil
import wavesight as ws
from matplotlib import style
# from memory_profiler import profile
from matplotlib import pyplot as plt

style.use('dark_background')

def approx_time(sim_cell, spatial_resolution, run_time, kappa=3.06e-6):
    rtime = (kappa * sim_cell.x
             * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def run_time_fun(full_sim_height, nCore):
    return 1.5 * full_sim_height * nCore

def sim_height_fun(λFree, pml_thickness):
    return (10 * λFree + 2 * pml_thickness)

# multiplies the time of the test-run for resource allocation
time_boost_factor = 1.75
# multiplies the simulation time
steady_time_boost = 1.5
# multiplies the memory of the test-run for resource allocation
memory_boost_factor = 1.25
# scales the run time, useful to debug by making the simulation end prematurely
run_time_scaler = 1.0
# whether to show the plots as simulations are running
show_plots = False
# control which plots are made
plot_field_profiles = False
plot_current_streams = False
# determines which sagittal planes are used for plotting
sag_plot_planes      = ['xz']
# minimun wall time for sim jobs
min_req_time_in_s    = 25 * 60
# Whether to save the fields to the pickle or just leave them in the .h5 files
save_fields_to_pkl = False

# required simulation time ceiled to this many seconds
time_req_rounded_to  = 60*15
# and memory ceiled to this many Gb
mem_req_rounded_to   = 1

def mode_solver(num_time_slices, mode_idx, sim_time, config_dict):
    '''
    This function takes a number of values, that are taken from 
    the CLI and a configuration dictionary that contains the basic
    geometry of the simulation region.

    There are three planes at which the fields are saved. An xy
    plane at a distance of 2 * λFree from the interface, and two
    sagittal planes at the center of the simulation volume.

    The xy monitors do not include the PML layers in the xy plane,
    but the sagittal monitors do include the PML layers in the
    z direction.

    Parameters
    ----------
    num_time_slices : int
        number of time slices for the time resolved simulation, if
        num_time_slices is 0, then the simulation is steady-state,
        in the sense that the simulation object is run without
        collecting fields at intervals.
    mode_idx : int
        index of the mode to be simulated. The specific quantities
        needed to launch this field into the simulation volume are
        taken from the fiber_sol dictionary.
    sim_time : float
        time for the simulation to run, the special value of 'auto' will
        cause the simulation time to be estimated from the height of
        the simulation volume.
    config_dict : dict
        Containing the following keys: nUpper, numModes, fiber_sol, fiber_alpha,
        slack_channel, MEEP_resolution, sample_resolution, distance_to_monitor.
    Returns
    -------
    None
        The function returns nothing but it saves to disk a pickle file
        for mode_sol, and several .h5 files in the case of a time-resolved
        simulation or a single one in the case of a steady-state simulation.
    '''
    nUpper = config_dict['nUpper']
    numModes = config_dict['numModes']
    fiber_sol = config_dict['fiber_sol']
    fiber_alpha = config_dict['fiber_alpha']
    sample_posts = config_dict['sample_posts']
    slack_channel = config_dict['slack_channel']
    MEEP_resolution = config_dict['MEEP_resolution']
    sample_resolution = config_dict['sample_resolution']
    distance_to_monitor = config_dict['distance_to_monitor']

    time_resolved = (num_time_slices > 0)
    initial_time  = time.time()
    sim_id        = ws.rando_id()
    output_dir    = '%s-%s-%s' % (data_dir, sim_id, mode_idx)
    send_to_slack = ((mode_idx % ceil(numModes/sample_posts)) == 0)
    if send_to_slack:
        slack_thread = ws.post_message_to_slack("%s - %s - %s" % (mode_idx, big_job_id, sim_id),
                                                slack_channel=slack_channel)
        thread_ts = slack_thread['ts']

    h_xy_slices_fname = 'h-field-xy-slices-' + sim_id
    e_xy_slices_fname = 'e-field-xy-slices-' + sim_id
    h_xz_slices_fname = 'h-field-xz-slices-' + sim_id
    e_xz_slices_fname = 'e-field-xz-slices-' + sim_id
    h_yz_slices_fname = 'h-field-yz-slices-' + sim_id
    e_yz_slices_fname = 'e-field-yz-slices-' + sim_id

    mode_sol = {'mode_idx': mode_idx,
                'nCore': fiber_sol['nCore'],
                'nCladding': fiber_sol['nCladding'],
                'coreRadius': fiber_sol['coreRadius'],
                'kFree': fiber_sol['kFree'],
                'sample_resolution': sample_resolution,
                'MEEP_resolution': MEEP_resolution,
                'num_time_slices': num_time_slices,
                'sim_id': sim_id
                }
    if send_to_slack:
        mode_sol['thread_ts'] = thread_ts
    
    mode_sol['time_resolved']        = time_resolved
    mode_sol['h_xy_slices_fname_h5'] = h_xy_slices_fname + '.h5'
    mode_sol['e_xy_slices_fname_h5'] = e_xy_slices_fname + '.h5'
    mode_sol['h_xz_slices_fname_h5'] = h_xz_slices_fname + '.h5'
    mode_sol['e_xz_slices_fname_h5'] = e_xz_slices_fname + '.h5'
    mode_sol['h_yz_slices_fname_h5'] = h_yz_slices_fname + '.h5'
    mode_sol['e_yz_slices_fname_h5'] = e_yz_slices_fname + '.h5'

    ehfieldh5fname = 'e-h-fields-%s.h5' % sim_id
    ehfieldh5fname = os.path.join(output_dir, ehfieldh5fname)
    mode_sol['eh_monitors_fname_h5'] = ehfieldh5fname

    coord_layout = fiber_sol['coord_layout']
    (coreRadius, sim_width, Δs, xrange,  yrange, 
        ρrange,  φrange,   Xg, Yg, ρg, φg, nxy, 
        crossMask, numSamples) = coord_layout
    eigennums  = fiber_sol['eigenbasis_nums']
    mode_idx = mode_sol['mode_idx']
    mode_params = eigennums[mode_idx]
    (modType, parity, m, kzidx, kz, γ, β) = mode_params

    mode_sol['kz'] = float(kz)
    mode_sol['m']  = m
    mode_sol['parity'] = parity
    mode_sol['modeType'] = modType
    mode_sol['γ'] = float(γ)
    mode_sol['β'] = float(β)

    nCore = mode_sol['nCore']
    nCladding = mode_sol['nCladding']
    coreRadius = mode_sol['coreRadius']
    kFree = mode_sol['kFree']

    # calculate the field functions
    print("Calculating the field functions from the analytical solution ...")
    (Efuncs, Hfuncs) = ws.fieldGenerator(coreRadius, kFree, kz, m, nCladding, nCore, modType)
    (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz) = Efuncs
    (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz) = Hfuncs
    funPairs = (((ECoreρ, ECladdingρ), (HCoreρ, HCladdingρ)),
                ((ECoreϕ, ECladdingϕ), (HCoreϕ, HCladdingϕ)),
                ((ECorez, ECladdingz), (HCorez, HCladdingz)))

    # put them together as needed to be provided to the custom current builder in MEEP
    modeFuns = {}
    for funPair in funPairs:
        (ECoreρ, ECladdingρ), (HCoreρ, HCladdingρ) = funPair
        EfunVal = ECoreρ(np.pi/np.sqrt(2))
        HfunVal = HCoreρ(np.pi/np.sqrt(2))
        if EfunVal != 0 or HfunVal != 0:
            if EfunVal == 0:
                componentName = HCoreρ.__name__.split('_')[-1].replace('Core','')
                modeFuns[componentName] =  (HCoreρ, HCladdingρ)
            else:
                componentName = ECoreρ.__name__.split('_')[-1].replace('Core','')
                modeFuns[componentName] =  (ECoreρ, ECladdingρ)

    # calculate the radial profile of the fields
    field_profiles = {}
    for componentName, (coreFun, claddingFun) in modeFuns.items():
        Ecorevals = np.vectorize(coreFun)(ρrange)
        ECladdingvals = np.vectorize(claddingFun)(ρrange)
        ρmask = ρrange < coreRadius
        Ecorevals[~ρmask] = 0
        ECladdingvals[ρmask] = 0
        E_all = Ecorevals + ECladdingvals
        field_profiles[componentName] = E_all

    mode_sol['field_profiles'] = (ρrange, field_profiles)

    print("Calculating a sample of the generating effective currents ...")
    # calculate the generating currents
    Xg, Yg, electric_J, magnetic_K = ws.field_sampler(funPairs, sim_width,
                                                sample_resolution,
                                                m, parity,
                                                coreRadius,
                                                coord_sys = 'cartesian-cartesian',
                                                equiv_currents=True)

    mode_sol['sampled_electric_J'] = electric_J
    mode_sol['sampled_magnetic_K'] = magnetic_K
    mode_sol['sampled_Xg'] = Xg
    mode_sol['sampled_Yg'] = Yg

    if plot_field_profiles:
        print("Making a plot of the field profiles ...")
        # make a plot of the field profiles
        vrange = 0
        fig, ax = plt.subplots(figsize=(6,3))
        for componentName in field_profiles:
            E_all = field_profiles[componentName]
            if (E_all.dtype) == np.complex128:
                E_all = np.imag(E_all)
                componentName = 'Im(%s)' % componentName
            ax.plot(ρrange, E_all, label=componentName)
            vrange = max(vrange, np.max(np.abs(E_all)))
        ax.plot([coreRadius]*2,[-vrange,vrange],'o--',color='w',ms=2,lw=0.5)
        plt.legend()
        ax.set_xlabel('x/μm')
        ax.set_ylabel('field')
        plt.tight_layout()
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel,
                                 'Field profiles',
                                 'field-profiles-%s.png' % sim_id,
                                 thread_ts = thread_ts)
        if show_plots:
            plt.show()
        else:
            plt.close()

    if plot_current_streams:
        print("Making a streamplot of the generating currents ...")
        streamArrayK = np.real(magnetic_K)
        streamArrayJ = np.real(electric_J)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.streamplot(Xg, Yg, streamArrayK[0],streamArrayK[1], density=1., color='b')
        ax.streamplot(Xg, Yg, streamArrayJ[0],streamArrayJ[1], density=1., color='r')
        coreBoundary = plt.Circle((0,0), coreRadius, color='w', fill=False)
        ax.add_patch(coreBoundary)
        ax.set_aspect('equal')
        ax.set_xlabel('x/μm')
        ax.set_ylabel('y/μm')
        ax.set_title('Equivalent currents.')
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel,
                                 'Current lines',
                                 'current-lines-%s.png' % sim_id, thread_ts = thread_ts)
        if show_plots:
            plt.show()
        else:
            plt.close()
        del streamArrayK
        del streamArrayJ

    (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz) = Efuncs
    (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz) = Hfuncs

    print("Calculating the functions for the necessary equivalent currents ...")
    Jx, Jy, Kx, Ky = ws.equivCurrents(Efuncs, Hfuncs, coreRadius, m, parity)

    print("Setting up the MEEP simulation ...")
    λFree                       = fiber_sol['λFree']
    kFree                       = 2*np.pi/λFree
    pml_thickness               = 2 * λFree
    cladding_width              = (sim_width/2 - coreRadius) + distance_to_monitor * np.tan(fiber_alpha)
    sim_width                   = 2*(coreRadius + cladding_width)
    mode_sol['sim_width']       = float(sim_width)
    full_sim_height             = sim_height_fun(λFree, pml_thickness)
    sim_height                  = full_sim_height - 2*pml_thickness
    mode_sol['full_sim_height'] = float(full_sim_height)
    mode_sol['pml_thickness']   = float(pml_thickness)
    mode_sol['cladding_width']  = float(cladding_width)
    mode_sol['free_wavelength'] = float(λFree)
    # from top edge of bottom pml to loc of source
    # also equal, from bottom edge of top pml to loc of monitor
    source_loc      = 2 * λFree
    # period of the fields
    base_period     = 1. / kFree

    # how long the source and simulation run
    if sim_time == 'auto':
        run_time    = run_time_scaler * run_time_fun(full_sim_height, nCore)
    else:
        run_time = sim_time
    mode_sol['run_time'] = float(run_time)
    if time_resolved:
        field_sampling_interval = run_time/num_time_slices
    else:
        field_sampling_interval = None
    source_time = run_time
    # the width of the simulation vol in the x and y directions adding the PML thickness
    full_sim_width = 2*(coreRadius + cladding_width) + 2 * pml_thickness
    # the PML layers evenly spread on each side of the simulation vol
    pml_layers  = [mp.PML(pml_thickness)]
    # the vol of the simulation
    sim_cell    = mp.Vector3(full_sim_width, full_sim_width, full_sim_height)
    # estime the required simulation time
    approx_runtime = approx_time(sim_cell, MEEP_resolution, run_time)

    # plot of the cross section
    print("Making a design draft from the fiber geometry ...")

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    axes[0].add_patch(plt.Circle((0,0), coreRadius, color='b', fill=True))
    pml_patch = ws.frame_patch((-full_sim_width/2,-full_sim_width/2),
                            full_sim_width,
                            full_sim_width,
                            pml_thickness,hatch='/')
    axes[0].add_patch(pml_patch),
    axes[0].set_xlim(-full_sim_width/2, full_sim_width/2)
    axes[0].set_ylim(-full_sim_width/2, full_sim_width/2)
    axes[0].set_xlabel('x/μm')
    axes[0].set_ylabel('y/μm')
    axes[0].set_aspect('equal')
    axes[0].set_title('xy cross section of simulation vol')

    # plot of the sagittal cross section
    axes[1].add_patch(plt.Rectangle((-full_sim_width/2,-full_sim_height/2),
                            full_sim_width, full_sim_height/2, 
                            color='g',
                            alpha=0.2,
                            fill=True))
    axes[1].add_patch(plt.Rectangle((-coreRadius,-full_sim_height/2),
                            2*coreRadius, full_sim_height/2, 
                            color='b',
                            alpha=0.8,
                            fill=True))
    # source
    axes[1].plot([-full_sim_width/2, full_sim_width/2], 
                 [-full_sim_height/2 + pml_thickness + source_loc]*2,
                 color='r')
    # monitor
    axes[1].plot([-full_sim_width/2, full_sim_width/2],
                 [full_sim_height/2 - pml_thickness - source_loc]*2,
                 color='g')
    axes[1].add_patch(plt.Rectangle([-full_sim_width/2 + pml_thickness, -full_sim_height/2 + pml_thickness], 
                            full_sim_width - pml_thickness*2, 
                            full_sim_height - pml_thickness*2,
                            color='w',
                            fill=False))
    # the PML hatching shade
    pml_patch = ws.frame_patch((-full_sim_width/2,-full_sim_height/2),
                               full_sim_width,
                               full_sim_height,
                               pml_thickness,hatch='/')
    axes[1].add_patch(pml_patch),
    axes[1].set_xlim(-full_sim_width/2, full_sim_width/2)
    axes[1].set_ylim(-full_sim_height/2,
                     full_sim_height/2)
    axes[1].set_xlabel('x/μm')
    axes[1].set_ylabel('z/μm')
    axes[1].set_title('Sagittal cross section of simulation vol')
    axes[1].set_aspect('equal')
    if send_to_slack and (mode_idx == 0):
        ws.send_fig_to_slack(fig, slack_channel, "Device layout", 'device-layout-%s.png' % sim_id, thread_ts = thread_ts)
    if show_plots:
        plt.show()
    else:
        plt.close()

    clear_aperture = sim_width
    print("Setting up the basic geometry of the FDTD simulation ...")
    cladding_medium = mp.Medium(index = nCladding)
    core_medium     = mp.Medium(index = nCore)
    upper_medium    = mp.Medium(index = nUpper)
    # set up the basic simulation geometry
    cladding_cell   = mp.Vector3(full_sim_width, full_sim_width, full_sim_height)
    cladding_center = mp.Vector3(0,0,0)
    geometry = [
        mp.Block(size    = cladding_cell,
                center   = cladding_center,
                material = cladding_medium),
        mp.Cylinder(radius   = coreRadius,
                    height   = full_sim_height/2,
                    axis     = mp.Vector3(0,0,1),
                    center   = mp.Vector3(0,0,-full_sim_height/4),
                    material = core_medium),
        mp.Block(size    = mp.Vector3(full_sim_width, full_sim_width, full_sim_height/2),
                center   = mp.Vector3(0, 0, full_sim_height/4),
                material = upper_medium),  
    ]

    print("Setting up the time-function for the sources ...")
    source_fun = mp.ContinuousSource(frequency=kFree/2/np.pi,
                                     width=2*base_period,
                                     end_time=source_time)

    print("Setting up the monitor planes ...")
    xy_monitor_plane_center = mp.Vector3(0, 0, distance_to_monitor)
    xy_monitor_plane_size   = mp.Vector3(clear_aperture, clear_aperture, 0)
    xy_monitor_vol          = mp.Volume(center=xy_monitor_plane_center, size=xy_monitor_plane_size)

    xz_monitor_plane_center = mp.Vector3(0,0,0)
    xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, full_sim_height)
    xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center,
                                        size=xz_monitor_plane_size)

    yz_monitor_plane_center = mp.Vector3(0,0,0)
    yz_monitor_plane_size   = mp.Vector3(0, clear_aperture, full_sim_height)
    yz_monitor_vol          = mp.Volume(center=yz_monitor_plane_center,
                                        size=yz_monitor_plane_size)

    print("Setting up the effective current sources for the modal fields ...")
    source_center = mp.Vector3(0,0, -full_sim_height/2 + pml_thickness + source_loc)
    source_size   = mp.Vector3(full_sim_width, full_sim_width, 0)

    # we assume that the axis of the fiber is along the z-axis
    # as such the transverse currents are the x and y components
    srcs = []
    for pair in [((mp.Ex, mp.Ey), (Jx, Jy)),
                 ((mp.Hx, mp.Hy), (Kx, Ky))]:
        for field_component, current_fun in zip(pair[0], pair[1]):
            src = mp.Source(src = source_fun,
                            component=field_component,
                            center=source_center,
                            size=source_size,
                            amp_func=current_fun
                            )
            srcs.append(src)

    print("Setting up the base simulation object ...")
    sim = mp.Simulation(
        cell_size  = sim_cell,
        geometry   = geometry,
        sources    = srcs,
        resolution = MEEP_resolution,
        boundary_layers      = pml_layers,
        force_complex_fields = True,
        filename_prefix = ''
    )
    sim.use_output_directory(output_dir)

    msg = "Simulation is estimated to take %0.2f minutes ..." % (approx_runtime/60)
    print(msg)
    mem_usage = sim.get_estimated_memory_usage()/1024/1024
    print(">> Estimated memory usage %.2f Mb ..." % mem_usage)
    if send_to_slack:
        ws.post_message_to_slack(msg, slack_channel=slack_channel, thread_ts = thread_ts) 

    xz_monitor_plane_center = mp.Vector3(0,0,0)
    xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, full_sim_height)
    xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center,
                                        size=xz_monitor_plane_size)

    start_time = time.time()
    if time_resolved:
        sim.run(
                mp.during_sources(mp.in_volume(xy_monitor_vol,
                                    mp.to_appended(h_xy_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_hfield)))),
                mp.during_sources(mp.in_volume(xy_monitor_vol,
                                    mp.to_appended(e_xy_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_efield)))),
                mp.during_sources(mp.in_volume(xz_monitor_vol,
                                    mp.to_appended(h_xz_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_hfield)))),
                mp.during_sources(mp.in_volume(xz_monitor_vol,
                                    mp.to_appended(e_xz_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_efield)))),
                mp.during_sources(mp.in_volume(yz_monitor_vol,
                                    mp.to_appended(h_yz_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_hfield)))),
                mp.during_sources(mp.in_volume(yz_monitor_vol,
                                    mp.to_appended(e_yz_slices_fname, 
                                    mp.at_every(field_sampling_interval,
                                                mp.output_efield)))),
                until=run_time)
    else:
        sim.run(until=run_time)
    end_time = time.time()
    time_taken = end_time - start_time

    coords = {}
    (xCoords, yCoords, zCoords, _) = sim.get_array_metadata()
    numMVx = len(xCoords) * len(yCoords) * len(zCoords)
    mode_sol['numMVx'] = int(numMVx/1e6)
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
    mode_sol['coords']     = coords
    mode_sol['sim_width_original'] = float(sim_width)
    sim_width = float(xCoordsMonxy[-1] - xCoordsMonxy[0])
    mode_sol['sim_width'] = sim_width
    on_axis_eps = sim.get_array(mp.Dielectric,
                    mp.Volume(
                        center=mp.Vector3(0,0,0),
                    size=mp.Vector3(0,0,full_sim_height))
                    )

    mode_sol['on_axis_eps'] = on_axis_eps 
    mode_sol['time_taken'] = time_taken
    mode_sol['field_sampling_interval'] = field_sampling_interval
    mode_sol['full_simulation_width_with_PML'] = full_sim_width
    mode_sol['numModes'] = numModes
    msg = "Simulation took %0.2f minutes to run." % (time_taken/60)
    print(msg)
    if send_to_slack:
        ws.post_message_to_slack(msg, slack_channel=slack_channel, thread_ts = thread_ts) 

    print("Getting the field data from the h5 files ...")
    if time_resolved:
        monitor_fields = {}
        for plane in ['xy','xz','yz']:
            field_arrays = []
            for idx, field_name in enumerate(['e','h']):
                field_data = {}
                h5_full_name = os.path.join(output_dir, 'fiber_platform-' + mode_sol[f'{field_name}_{plane}_slices_fname_h5'])
                with h5py.File(h5_full_name,'r') as h5_file:
                    h5_keys = list(h5_file.keys())
                    for h5_key in h5_keys:
                        datum = np.array(h5_file[h5_key][:,:])
                        datum = np.transpose(datum, (1,0,2))
                        field_data[h5_key] = datum
                field_array = np.zeros((3,)+datum.shape, dtype=np.complex_)
                field_parts  = f'{field_name}x {field_name}y {field_name}z'.split(' ')
                for idx, field_component in enumerate(field_parts):
                    field_array[idx] = 1j*np.array(field_data[field_component+'.i'])
                    field_array[idx] +=   np.array(field_data[field_component+'.r'])
                field_arrays.append(field_array)
            field_arrays = np.array(field_arrays)
            monitor_fields[plane] = field_arrays
    else:
        monitor_vols = {'xy': xy_monitor_vol,'yz': yz_monitor_vol,'xz': xz_monitor_vol}
        h5_fname = ehfieldh5fname
        with h5py.File(h5_fname,'w') as h5_file:
            monitor_fields = {}
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
                        h5_file.create_dataset(key, data = export_field)
                monitor_fields[monitor_plane] = np.array([a_monitor_fields[:3], a_monitor_fields[3:]])

    print("Calculating basic plots for the end time ...")
    for sagplane in sag_plot_planes:
        if time_resolved:
            Ey_final_sag = monitor_fields[sagplane][0,1,:,:,-1]
        else:
            Ey_final_sag = monitor_fields[sagplane][0,1,:,:]
        extent = [-sim_width/2, sim_width/2, -full_sim_height/2, full_sim_height/2]
        plotField = np.real(Ey_final_sag.T)
        fig, ax   = plt.subplots(figsize=(3, 3 * full_sim_height / sim_width))
        prange = np.max(np.abs(plotField))
        pretty_range = '$%s$' % ws.num2tex(prange, 2)
        ax.imshow(plotField, 
                cmap   = cm.watermelon, 
                origin = 'lower',
                extent = extent,
                vmin   = -prange,
                vmax   = prange,
                interpolation = 'none')
        ax.plot([-coreRadius,-coreRadius],[-full_sim_height/2,0],'r:',alpha=0.3)
        ax.plot([coreRadius,coreRadius],[-full_sim_height/2,0],'r:',alpha=0.3)
        ax.set_xlabel('%s/μm' % sagplane[0])
        ax.set_ylabel('z/μm')
        title = 'Re(Ey) | [%s]' % (pretty_range)
        ax.set_title(title)
        plt.tight_layout()
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel,
                                 'sagittal-%s-final-Ey' % sagplane,
                                 'sagittal-%s-final-Ey' % sagplane,
                                 thread_ts)
        if show_plots:
            plt.show()
        else:
            plt.close()

    # print("Picking a notable point at z=0 for sampling Ey at different times and z-values ...")
    # numZsamples = Ey_final_sag.shape[0]
    # midZsample = numZsamples // 2
    # midCut     = Ey_final_sag[midZsample, :]
    # goodIndex = np.argmax(np.real(midCut))
    # columnChange = monitor_fields['xz'][0,1,:,goodIndex]
    # plotField = np.real(columnChange)
    # extent = [0, run_time, -full_sim_height/2, full_sim_height/2]
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(plotField, 
    #         origin='lower', 
    #         extent=extent,
    #         cmap=cm.watermelon)
    # ax.set_xlabel('t/ν')
    # ax.set_ylabel('z/μm')
    # ax.set_aspect(run_time/full_sim_height)
    # ax.set_title('Re(Ey) at fixed x,y')
    # plt.tight_layout()
    # if send_to_slack:
    #     ws.send_fig_to_slack(fig, slack_channel, 'Re(Ey) at fixed x,y','Re(Ey)-at-fixed-x,y',thread_ts)
    # if show_plots:
    #     plt.show()
    # else:
    #     plt.close()

    print("Sampling the ground-truth modal profile ...")
    Xg, Yg, E_field_GT, H_field_GT = ws.field_sampler(funPairs, 
                                                clear_aperture, 
                                                MEEP_resolution, 
                                                m, 
                                                parity, 
                                                coreRadius, 
                                                coord_sys = 'cartesian-cartesian',
                                                equiv_currents=False)
    if time_resolved:
        field_array = monitor_fields['xy'][1,:,:,:,-1]
    else:
        field_array = monitor_fields['xy'][1]
 
    print("Making a comparison plot of the last measured field against mode ...")
    component_name = 'hx'
    component_index = {'hx':0, 'hy':1, 'hz':2}[component_name]
    extent    = [-clear_aperture/2, clear_aperture/2, 
                 -clear_aperture/2, clear_aperture/2]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    for idx, plotFun in enumerate([np.real, np.imag]):
        ax = axes[0,idx]
        plotField = field_array[component_index]
        plotField = plotFun(plotField)
        real_or_im = ['Re','Im'][idx]
        prange = np.max(np.abs(plotField))
        pretty_range = '$%s$' % ws.num2tex(prange, 2)
        title = '%s($H_%s$) | [%s]' % (real_or_im, component_name[-1], pretty_range)
        ax.imshow(plotField, vmin=-prange, vmax=prange, extent=extent, cmap=cm.watermelon)
        ax.set_title(title)
        ax.set_xlabel('x/μm')
        ax.set_ylabel('y/μm')
        ax.add_patch(plt.Circle((0,0), coreRadius, color='w', fill=False))
        ax = axes[1,idx]
        plotField = plotFun(H_field_GT[component_index])
        prange = np.max(np.abs(plotField))
        pretty_range = '$%s$' % ws.num2tex(prange, 2)
        title = '%s($H_%s$) | [%s]' % (real_or_im, component_name[-1], pretty_range)
        if idx == 0:
            vmin = -prange
            vmax = prange
            cmap = cm.watermelon
        else:
            vmin = 0
            vmax = prange
            cmap = cm.ember
        ax.imshow(plotField,
                  vmin=vmin,
                  vmax=vmax,
                  extent=extent,
                  cmap=cmap)
        title = 'Mode field | ' + title
        ax.set_title(title)
        ax.set_xlabel('x/μm')
        ax.set_ylabel('y/μm')
        ax.add_patch(plt.Circle((0,0), coreRadius, color='w', fill=False))
    plt.tight_layout()
    if send_to_slack:
        ws.send_fig_to_slack(fig, slack_channel, 
                             'Comparison of last measured field',
                             'comparison-of-last-measured-field',
                             thread_ts)
    if show_plots:
        plt.show()
    else:
        plt.close()
    mode_sol['approx_MEEP_mem_usage_in_MB'] = int(mem_usage)
    process = psutil.Process(os.getpid())
    mem_used_in_bytes = process.memory_info().rss
    mem_used_in_Mbytes = mem_used_in_bytes/1024/1024
    mode_sol['overall_mem_usage_in_MB'] = int(mem_used_in_Mbytes)
    summary = ws.dict_summary(mode_sol, 'SIM-'+sim_id)
    if send_to_slack:
        ws.post_message_to_slack(summary, slack_channel=slack_channel,thread_ts=thread_ts)
    if save_fields_to_pkl:
        mode_sol['monitor_field_slices'] = monitor_fields
    pkl_fname = 'sim-%s.pkl' % sim_id
    pkl_fname = os.path.join(output_dir, pkl_fname)
    print("Calculating the size of h5 data files ...")
    disk_usage_in_MB = ws.get_total_size_of_directory(output_dir)
    mode_sol['disk_usage_in_MB'] = int(disk_usage_in_MB)
    with open(pkl_fname, 'wb') as file:
        print("Saving solution to %s ..." % pkl_fname)
        pickle.dump(mode_sol, file)

    mem_used_in_Gbytes = mem_used_in_Mbytes/1024
    final_final_time   = time.time()
    time_taken_in_s    = final_final_time - initial_time
    time_req_in_s      = ws.ceil_to_multiple(time_boost_factor*time_taken_in_s,
                                          time_req_rounded_to)
    time_req_in_s      = max(time_req_in_s, min_req_time_in_s)
    mem_req_in_GB      = int(ws.ceil_to_multiple(memory_boost_factor*mem_used_in_Gbytes,
                                              mem_req_rounded_to))
    time_req_fmt       = ws.format_time(time_req_in_s)
    mem_req_fmt        = '%d' % mem_req_in_GB
    print("Took %s s to run, and spent %.1f MB of RAM." % (time_taken_in_s,mem_used_in_Mbytes ))
    # if the simulation time was run as automatic
    # then see if the steady state can be determined
    if not os.path.exists(reqs_fname):
        if sim_time == 'auto':
            run_time = mode_sol['run_time']
            E_field_xy = monitor_fields['xy'][0]
            time_slice = np.sum(np.abs(E_field_xy)**2, axis=(0, 1, 2))
            time_axis = np.linspace(0, run_time, len(time_slice))
            steady_time = ws.transient_scope(time_axis, time_slice)
            steady_time = steady_time_boost * steady_time
            steady_time = '%.2f' % steady_time
            print("Creating resource requirement file...")
            with open(reqs_fname,'w') as file:
                file.write('%s,%s,%s,%s' % (mem_req_fmt, time_req_fmt, disk_usage_in_MB, steady_time)) 
        else:
            print("Creting resource requirement file...")
            with open(reqs_fname,'w') as file:
                file.write('%s,%s,%s' % (mem_req_fmt, time_req_fmt, disk_usage_in_MB)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mode launcher.')
    parser.add_argument('--config_dict',
                        type=str,
                        help='Configuration file for modes.')
    parser.add_argument('--num_time_slices',
                        nargs='?',
                        type=int,
                        const=0,
                        help='Determines sampling interval of fields.')
    parser.add_argument('--mode_idx',
                        type=int,
                        help='The index for the launch mode.')
    parser.add_argument('--sim_time',
                        type=float,
                        nargs='?',
                        const=0,
                        help='The simulation time.')
    args = parser.parse_args()
    config_dict_fname = args.config_dict

    config_dict = pickle.load(open(config_dict_fname,'rb'))
    big_job_id  = config_dict_fname.split('config_dict-')[-1].split('.')[0]
    reqs_fname = '%s.req' % big_job_id
    data_dir = '/users/jlizaraz/data/jlizaraz/CEM/wavesight/'
    data_dir = os.path.join(data_dir, big_job_id)
    num_time_slices = args.num_time_slices
    mode_idx = args.mode_idx
    sim_time = args.sim_time
    if sim_time == 0:
        sim_time = 'auto'
    mode_solver(num_time_slices, mode_idx, sim_time, config_dict)