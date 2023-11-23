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

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

def approx_time(sim_cell, spatial_resolution, run_time, kappa=3.06e-6):
    rtime = (kappa * sim_cell.x * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)

def ceil_to_multiple(number, multiple):
    return ceil(number / multiple) * multiple

parser = argparse.ArgumentParser(description='Mode launcher.')
parser.add_argument('config_dict_fname', type=str, help='Configuration file for modes.')
parser.add_argument('num_time_slices', type=int, help='Determines sampling interval of fields.')
parser.add_argument('modeidx', type=int, help='The index for the launch mode.')
args = parser.parse_args()

# multiplies the time of the test run for resource allocation
time_boost_factor = 1.75
# multiplies the memore of the test run for resource allocation
memory_boost_factor = 1.25
# scales the run time, useful to debug by making simulation faster
run_time_scaler = 1.0 
data_dir = '/users/jlizaraz/data/jlizaraz/CEM/wavesight/'
show_plots = False
config_dict = pickle.load(open(args.config_dict_fname,'rb'))
big_job_id  = args.config_dict_fname.split('config_dict-')[-1].split('.')[0]
reqs_fname = '%s.req' % big_job_id
data_dir = os.path.join(data_dir, big_job_id)
grab_fields = True
save_fields_to_pkl = False
sample_posts = 10 # sample about this amount and post to Slack as sims progress
plot_field_profiles = False
plot_current_streams = False
# determines which sagittal planes are use for logging to Slack
sag_plot_planes = ['xz']
min_req_time_in_s = 25 * 60

# load all the config values to session
for k,v in config_dict.items():
    globals()[k] = v

def run_time_fun(fiber_height, nCore):
    return 0.75 * fiber_height * nCore
sim_height_fun = lambda λFree, pml_thickness: (10 * λFree + 2 * pml_thickness)

# @profile
def mode_solver(num_time_slices, mode_idx):
    print("MASTER", mp.am_master())
    print("RANK", rank)
    initial_time = time.time()
    sim_id        = ws.rando_id()
    output_dir    = '%s-%s-%s' % (data_dir, sim_id, mode_idx)
    send_to_slack = ((mode_idx % ceil(numModes/sample_posts)) == 0) and (mp.am_master() == 0)
    if send_to_slack:
        slack_thread = ws.post_message_to_slack("%s - %s - %s" % (mode_idx, big_job_id, sim_id), slack_channel=slack_channel)
        thread_ts = slack_thread['ts']
    send_to_slack = False
    h_xy_slices_fname = 'h-field-xy-slices-' + sim_id
    e_xy_slices_fname = 'e-field-xy-slices-' + sim_id
    h_xz_slices_fname = 'h-field-xz-slices-' + sim_id
    e_xz_slices_fname = 'e-field-xz-slices-' + sim_id
    h_yz_slices_fname = 'h-field-yz-slices-' + sim_id
    e_yz_slices_fname = 'e-field-yz-slices-' + sim_id
    MEEP_resolution = 10
    mode_sol = {'mode_idx': mode_idx,
                'nCore': fiber_sol['nCore'],
                'nCladding': fiber_sol['nCladding'],
                'coreRadius': fiber_sol['coreRadius'],
                'kFree': fiber_sol['kFree'],
                'sample_resolution': sample_resolution,
                'MEEP_resolution': MEEP_resolution,
                'num_time_slices': num_time_slices
                }
    if send_to_slack:
        mode_sol['thread_ts'] = thread_ts    
    mode_sol['h_xy_slices_fname_h5'] = h_xy_slices_fname + '.h5'
    mode_sol['e_xy_slices_fname_h5'] = e_xy_slices_fname + '.h5'
    mode_sol['h_xz_slices_fname_h5'] = h_xz_slices_fname + '.h5'
    mode_sol['e_xz_slices_fname_h5'] = e_xz_slices_fname + '.h5'
    mode_sol['h_yz_slices_fname_h5'] = h_yz_slices_fname + '.h5'
    mode_sol['e_yz_slices_fname_h5'] = e_yz_slices_fname + '.h5'

    coord_layout = fiber_sol['coord_layout']
    (coreRadius, simWidth, Δs, xrange, yrange, 
        ρrange, φrange, Xg, Yg, ρg, φg, nxy, 
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
    Xg, Yg, electric_J, magnetic_K = ws.field_sampler(funPairs, simWidth,
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
            ws.send_fig_to_slack(fig, slack_channel, "Field profiles", 'field-profiles-%s.png' % sim_id, thread_ts = thread_ts)
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
            ws.send_fig_to_slack(fig, slack_channel, "Current lines", 'current-lines-%s.png' % sim_id, thread_ts = thread_ts)
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
    λFree           = fiber_sol['λFree']
    kFree           = 2*np.pi/λFree
    pml_thickness   = 2 * λFree
    cladding_width    = (simWidth/2 - coreRadius) + distance_to_monitor * np.tan(fiber_alpha)
    simWidth          = 2*(coreRadius + cladding_width)
    mode_sol['simWidth'] = float(simWidth)
    fiber_height    = sim_height_fun(λFree, pml_thickness)
    mode_sol['fiber_height'] = float(fiber_height)
    mode_sol['pml_thickness'] = float(pml_thickness)
    mode_sol['cladding_width'] = float(cladding_width)
    mode_sol['free_wavelength'] = float(λFree)
    # from top edge of bottom pml to loc of source
    # also equal, from bottom edge of top pml to loc of monitor
    source_loc      = 2*λFree
    # period of the fields
    base_period     = 1./kFree

    # how long the source and simulation run
    run_time    = run_time_scaler * run_time_fun(fiber_height, nCore)
    mode_sol['run_time'] = float(run_time)
    field_sampling_interval = run_time/num_time_slices
    source_time = run_time
    # the width of the simulation vol in the x and y directions adding the PML thickness
    sxy         = 2*(coreRadius + cladding_width) + 2 * pml_thickness
    # the PML layers evenly spread on each side of the simulation vol
    pml_layers  = [mp.PML(pml_thickness)]
    # the vol of the simulation
    sim_cell    = mp.Vector3(sxy, sxy, fiber_height)
    # estime the required simulation time
    approx_runtime = approx_time(sim_cell, MEEP_resolution, run_time)

    # plot of the cross section
    print("Making a design draft from the fiber geometry ...")

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    axes[0].add_patch(plt.Circle((0,0), coreRadius, color='b', fill=True))
    pml_patch = ws.frame_patch((-sxy/2,-sxy/2),
                            sxy,
                            sxy,
                            pml_thickness,hatch='/')
    axes[0].add_patch(pml_patch),
    axes[0].set_xlim(-sxy/2, sxy/2)
    axes[0].set_ylim(-sxy/2, sxy/2)
    axes[0].set_xlabel('x/μm')
    axes[0].set_ylabel('y/μm')
    axes[0].set_aspect('equal')
    axes[0].set_title('xy cross section of simulation vol')

    # plot of the sagittal cross section
    axes[1].add_patch(plt.Rectangle((-sxy/2,-fiber_height/2),
                            sxy, fiber_height/2, 
                            color='g',
                            alpha=0.2,
                            fill=True))
    axes[1].add_patch(plt.Rectangle((-coreRadius,-fiber_height/2),
                            2*coreRadius, fiber_height/2, 
                            color='b',
                            alpha=0.8,
                            fill=True))
    # source
    axes[1].plot([-sxy/2, sxy/2], 
                 [-fiber_height/2 + pml_thickness + source_loc]*2,
                 color='r')
    # monitor
    axes[1].plot([-sxy/2, sxy/2],
                 [fiber_height/2 - pml_thickness - source_loc]*2,
                 color='g')
    axes[1].add_patch(plt.Rectangle([-sxy/2 + pml_thickness, -fiber_height/2 + pml_thickness], 
                            sxy - pml_thickness*2, 
                            fiber_height - pml_thickness*2,
                            color='w',
                            fill=False))
    # the PML hatching shade
    pml_patch = ws.frame_patch((-sxy/2,-fiber_height/2),
                               sxy,
                               fiber_height,
                               pml_thickness,hatch='/')
    axes[1].add_patch(pml_patch),
    axes[1].set_xlim(-sxy/2, sxy/2)
    axes[1].set_ylim(-fiber_height/2,
                     fiber_height/2)
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

    clear_aperture = simWidth
    print("Setting up the basic geometry of the FDTD simulation ...")
    cladding_medium = mp.Medium(index = nCladding)
    core_medium     = mp.Medium(index = nCore)
    between_medium    = mp.Medium(index = nBetween)
    # set up the basic simulation geometry
    cladding_cell   = mp.Vector3(sxy, sxy, fiber_height)
    cladding_center = mp.Vector3(0,0,0)
    geometry = [
        mp.Block(size    = cladding_cell,
                center   = cladding_center,
                material = cladding_medium),
        mp.Cylinder(radius   = coreRadius,
                    height   = fiber_height/2,
                    axis     = mp.Vector3(0,0,1),
                    center   = mp.Vector3(0,0,-fiber_height/4),
                    material = core_medium),
        mp.Block(size    = mp.Vector3(sxy, sxy, fiber_height/2),
                center   = mp.Vector3(0, 0, fiber_height/4),
                material = between_medium),  
    ]

    print("Setting up the time-function for the sources ...")
    source_fun = mp.ContinuousSource(frequency=kFree/2/np.pi,
                                    end_time=source_time)

    print("Setting up the monitor planes ...")
    xy_monitor_plane_center = mp.Vector3(0, 0, distance_to_monitor)
    xy_monitor_plane_size   = mp.Vector3(clear_aperture, clear_aperture, 0)
    xy_monitor_vol          = mp.Volume(center=xy_monitor_plane_center, size=xy_monitor_plane_size)

    xz_monitor_plane_center = mp.Vector3(0,0,0)
    xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, fiber_height)
    xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center,
                                        size=xz_monitor_plane_size)

    yz_monitor_plane_center = mp.Vector3(0,0,0)
    yz_monitor_plane_size   = mp.Vector3(0, clear_aperture, fiber_height)
    yz_monitor_vol          = mp.Volume(center=yz_monitor_plane_center,
                                        size=yz_monitor_plane_size)

    print("Setting up the effective current sources for the modal fields ...")
    source_center = mp.Vector3(0,0, -fiber_height/2 + pml_thickness + source_loc)
    source_size   = mp.Vector3(sxy, sxy, 0)

    # we assume that the axis of the fiber is along the z-axis
    # as such the transverse currents are the x and y components
    srcs = []
    for pair in [((mp.Ex, mp.Ey),(Jx, Jy)),
                ((mp.Hx, mp.Hy),(Kx, Ky))]:
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
        # resolution = 5,
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
    xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, fiber_height)
    xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center, size=xz_monitor_plane_size)

    start_time = time.time()

    sim.run(until=run_time)
    end_time = time.time()
    time_taken = end_time - start_time
    if mp.am_master():
        coords = {}
        (xCoords, yCoords, zCoords, _) = sim.get_array_metadata()
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
        mode_sol['simWidth_original'] = float(simWidth)
        simWidth = float(xCoordsMonxy[-1] - xCoordsMonxy[0])
        mode_sol['simWidth'] = simWidth
        on_axis_eps = sim.get_array(mp.Dielectric,
                    mp.Volume(
                        center=mp.Vector3(0,0,0),
                    size=mp.Vector3(0,0,fiber_height))
        )

        mode_sol['on_axis_eps'] = on_axis_eps 
        mode_sol['time_taken'] = time_taken
        mode_sol['field_sampling_interval'] = field_sampling_interval
        mode_sol['full_simulation_width_with_PML'] = sxy
        mode_sol['numModes'] = numModes
        msg = "Simulation took %0.2f minutes to run." % (time_taken/60)
        print(msg)
        if send_to_slack:
            ws.post_message_to_slack(msg, slack_channel=slack_channel, thread_ts = thread_ts) 

        monitor_vols = {'xy': xy_monitor_vol,'yz': yz_monitor_vol,'xz': xz_monitor_vol}
        h5_fname = 'EH_fields-%s.h5' % sim_id
        h5_fname = os.path.join(output_dir, h5_fname)
        with h5py.File(h5_fname,'w') as h5_file:
            for monitor_plane in 'xz yz xy'.split(' '):
                a_monitor_fields = []
                for field_component in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]:
                    a_monitor_fields.append(sim.get_array(field_component,monitor_vols[monitor_plane]))
                a_monitor_fields = np.array(a_monitor_fields)
                for field_idx, field_component in enumerate('ex ey ez hx hy hz'.split(' ')):
                    for complex_part, complex_fun in zip(['i','r'], [np.real, np.imag]):
                        key = '/%s/%s.%s' % (monitor_plane, field_component, complex_part)
                        export_field = complex_fun(a_monitor_fields[field_idx])
                        h5_file.create_dataset(key, data = export_field)
        
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
        time_req_in_s      = ceil_to_multiple(time_boost_factor*time_taken_in_s, 60*15)
        time_req_in_s      = max(time_req_in_s, min_req_time_in_s)
        mem_req_in_GB      = int(ceil_to_multiple(memory_boost_factor*mem_used_in_Gbytes, 1.0))
        time_req_fmt       = format_time(time_req_in_s)
        mem_req_fmt        = '%d' % mem_req_in_GB
        print("Took %s s to run, and spent %.1f MB of RAM." % (time_taken_in_s,mem_used_in_Mbytes ))
        if not os.path.exists(reqs_fname):
            print("Creting resource requirement file...")
            with open(reqs_fname,'w') as file:
                file.write('%s,%s,%s' % (mem_req_fmt, time_req_fmt, disk_usage_in_MB)) 

if __name__ == "__main__":
    mode_solver(args.num_time_slices, args.modeidx)