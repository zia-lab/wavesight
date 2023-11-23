#!/usr/bin/env python3

import sys
import meep as mp
import numpy as np
import h5py
import cmasher as cm
from matplotlib import pyplot as plt
import wavesight as ws
import time
import cmasher as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle
import argparse

def approx_time(sim_cell, spatial_resolution, run_time, kappa=2.2e-6):
    rtime = (kappa * sim_cell.x * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def main(nCladding, nCore, coreRadius, λFree):
    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nBetween': 1.,
                'λFree': λFree}
    fiber_sol = ws.multisolver(fiber_spec,
                            solve_modes = 'all',
                            drawPlots=False,
                            verbose=True)
    numModes = fiber_sol['totalModes']
    fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
    eigenbasis = fiber_sol['eigenbasis']
    a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples = fiber_sol['coord_layout']
    show_plot = False
    send_to_slack = True
    for mode_idx in range(numModes):
        sample_resolution = 10
        MEEP_resolution  = 20
        min_cycles      = 12
        sim_height_fun = lambda λFree, pml_thickness: (10 * λFree + 2 * pml_thickness)
        sim_id        = str(int(time.time()))
        grab_fields  = True # whether to import the h5 files that contain the monotired fields
        slack_channel = 'nvs_and_metalenses'
        num_time_slices = 150 # how many time samples of fields

        if send_to_slack:
            slack_thread = ws.post_message_to_slack("%s - MEEP simulation - %s" % (mode_idx, sim_id), slack_channel=slack_channel)
            thread_ts = slack_thread['ts']

        def run_time_fun(fiber_height, nCore, min_cycles, base_period):
            return 0.75 * fiber_height * nCore

        h_xy_slices_fname = 'h-field-xy-slices-' + sim_id
        e_xy_slices_fname = 'e-field-xy-slices-' + sim_id
        h_xz_slices_fname = 'h-field-xz-slices-' + sim_id
        e_xz_slices_fname = 'e-field-xz-slices-' + sim_id

        mode_sol = {'fiber_sol': fiber_sol,
                    'mode_idx': mode_idx,
                    'nCore': fiber_sol['nCore'],
                    'nCladding': fiber_sol['nCladding'],
                    'coreRadius': fiber_sol['coreRadius'],
                    'kFree': fiber_sol['kFree'],
                    'sample_resolution': sample_resolution,
                    'MEEP_resolution': MEEP_resolution,
                    'num_time_slices': num_time_slices
                    }
        
        mode_sol['h_xy_slices_fname_h5'] = h_xy_slices_fname + '.h5'
        mode_sol['e_xy_slices_fname_h5'] = e_xy_slices_fname + '.h5'
        mode_sol['h_xz_slices_fname_h5'] = h_xz_slices_fname + '.h5'
        mode_sol['e_xz_slices_fname_h5'] = e_xz_slices_fname + '.h5'

        coord_layout = fiber_sol['coord_layout']
        (coreRadius, simWidth, Δs, xrange, yrange, 
        ρrange, φrange, Xg, Yg, ρg, φg, nxy, 
        crossMask, numSamples) = coord_layout
        mode_sol['simWidth'] = simWidth
        eigenbasis = fiber_sol['eigenbasis']
        eigennums  = fiber_sol['eigenbasis_nums']
        mode_idx = mode_sol['mode_idx']
        amode = eigenbasis[mode_idx]
        mode_params = eigennums[mode_idx]
        (modType, parity, m, kzidx, kz, γ, β) = mode_params

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
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, "Field profiles", 'field-profiles-%s.png' % sim_id, thread_ts = thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()

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
        if show_plot:
            plt.show()
        else:
            plt.close()

        (ECoreρ, ECoreϕ, ECorez, ECladdingρ, ECladdingϕ, ECladdingz) = Efuncs
        (HCoreρ, HCoreϕ, HCorez, HCladdingρ, HCladdingϕ, HCladdingz) = Hfuncs

        print("Calculating the functions for the necessary equivalent currents ...")
        Jx, Jy, Kx, Ky = ws.equivCurrents(Efuncs, Hfuncs, coreRadius, m, parity)

        print("Setting up the MEEP simulation ...")
        λFree           = fiber_sol['λFree'] 
        kFree           = 2*np.pi/λFree
        pml_thickness   = 2 * coreRadius
        cladding_width  = simWidth/2 - coreRadius
        fiber_height    = sim_height_fun(λFree, pml_thickness)
        mode_sol['fiber_height'] = fiber_height
        mode_sol['pml_thickness'] = pml_thickness
        mode_sol['cladding_width'] = cladding_width
        # from top edge of bottom pml to loc of source
        # also equal, from bottom edge of top pml to loc of monitor
        source_loc      = 2*λFree
        # period of the fields
        base_period     = 1./kFree

        # how long the source and simulation run
        run_time    = run_time_fun(fiber_height, nCore, min_cycles, base_period)
        mode_sol['run_time'] = run_time
        field_sampling_interval = run_time/num_time_slices
        source_time = run_time
        # the width of the simulation vol in the x and y directions adding the PML thickness
        sxy         = simWidth + 2 * pml_thickness
        # the PML layers evenly spread on each side of the simulation vol
        pml_layers  = [mp.PML(pml_thickness)]
        # the vol of the simulation
        sim_cell    = mp.Vector3(sxy, sxy, fiber_height)
        # estime the required simulation time
        approx_runtime = approx_time(sim_cell, MEEP_resolution, run_time)

        # plot of the cross section
        print("Making a design draft from the fiber geometry ...")
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
        axes[0].add_patch(plt.Circle((0,0), coreRadius, color='w', fill=False))
        axes[0].add_patch(plt.Rectangle((-sxy/2 + pml_thickness, -sxy/2 + pml_thickness),
                                sxy - 2*pml_thickness, sxy - 2*pml_thickness, color='w', fill=False))
        axes[0].set_xlim(-sxy/2, sxy/2)
        axes[0].set_ylim(-sxy/2, sxy/2)
        axes[0].set_xlabel('x/μm')
        axes[0].set_ylabel('y/μm')
        axes[0].set_aspect('equal')
        axes[0].set_title('Cross section of fiber')

        # plot of the sagittal cross section
        axes[1].add_patch(plt.Rectangle((-coreRadius,-fiber_height/2),
                                2*coreRadius, fiber_height, color='b', fill=True))
        # source
        axes[1].plot([-sxy/2, sxy/2], [-fiber_height/2 + pml_thickness + source_loc]*2, color='r')
        # monitor
        axes[1].plot([-sxy/2, sxy/2], [fiber_height/2 - pml_thickness - source_loc]*2, color='g')
        axes[1].add_patch(plt.Rectangle([-sxy/2 + pml_thickness, -fiber_height/2 + pml_thickness], 
                                sxy - pml_thickness*2, 
                                fiber_height - pml_thickness*2, color='w', fill=False))
        axes[1].set_xlim(-sxy/2, sxy/2)
        axes[1].set_ylim(-fiber_height/2, fiber_height/2)
        axes[1].set_xlabel('x/μm')
        axes[1].set_ylabel('z/μm')
        axes[1].set_title('Sagital cross section of simulation vol')
        axes[1].set_aspect('equal')
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, "Device layout", 'device-layout-%s.png' % sim_id, thread_ts = thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()

        clear_aperture = simWidth
        print("Setting up the basic geometry of the FDTD simulation ...")
        cladding_medium = mp.Medium(index=nCladding)
        core_medium     = mp.Medium(index=nCore)
        # set up the basic simulation geometry
        # cladding_cell = sim_cell
        cladding_cell = mp.Vector3(sxy, sxy, fiber_height)
        cladding_center = mp.Vector3(0,0,0)
        geometry = [
            mp.Block(size=cladding_cell,
                    center=cladding_center,
                    material=cladding_medium),
            mp.Cylinder(radius=coreRadius,
                        axis  = mp.Vector3(0,0,1),
                        material=core_medium),
            
        ]

        print("Setting up the time-function for the sources ...")
        source_fun = mp.ContinuousSource(frequency=kFree/2/np.pi,
                                        end_time=run_time)

        print("Setting up the monitor planes ...")
        xy_monitor_plane_center = mp.Vector3(0,0,fiber_height/2 - pml_thickness - source_loc)
        xy_monitor_plane_size   = mp.Vector3(clear_aperture, clear_aperture, 0)
        xy_monitor_vol          = mp.Volume(center=xy_monitor_plane_center, size=xy_monitor_plane_size)

        xz_monitor_plane_center = mp.Vector3(0,0,0)
        xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, fiber_height)
        xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center, size=xz_monitor_plane_size)

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
            boundary_layers      = pml_layers,
            force_complex_fields = True
        )


        msg = "Simulation is estimated to take %0.2f minutes ..." % (approx_runtime/60)
        print(msg)
        if send_to_slack:
            ws.post_message_to_slack(msg, slack_channel=slack_channel, thread_ts = thread_ts) 

        xz_monitor_plane_center = mp.Vector3(0,0,0)
        xz_monitor_plane_size   = mp.Vector3(clear_aperture, 0, fiber_height)
        xz_monitor_vol          = mp.Volume(center=xz_monitor_plane_center, size=xz_monitor_plane_size)

        start_time = time.time()
        sim.run(mp.at_beginning(mp.output_epsilon),
                mp.during_sources(mp.in_volume(xy_monitor_vol,
                                mp.to_appended(h_xy_slices_fname, 
                                    mp.at_every(field_sampling_interval, mp.output_hfield)))),
                mp.during_sources(mp.in_volume(xy_monitor_vol,
                                mp.to_appended(e_xy_slices_fname, 
                                    mp.at_every(field_sampling_interval, mp.output_efield)))),
                mp.during_sources(mp.in_volume(xz_monitor_vol,
                                mp.to_appended(h_xz_slices_fname, 
                                    mp.at_every(field_sampling_interval, mp.output_hfield)))),
                mp.during_sources(mp.in_volume(xz_monitor_vol,
                                mp.to_appended(e_xz_slices_fname, 
                                    mp.at_every(field_sampling_interval, mp.output_efield)))),
                until=run_time)
        end_time = time.time()
        time_taken = end_time - start_time

        coords = {}
        (xCoords, yCoords, zCoords, _) = sim.get_array_metadata()
        (xCoordsMonxy, yCoordsMonxy, zCoordsMonxy, _) = sim.get_array_metadata(xy_monitor_vol)
        (xCoordsMonxz, yCoordsMonxz, zCoordsMonxz, _) = sim.get_array_metadata(xz_monitor_vol)
        coords['xCoords'] = xCoords
        coords['yCoords'] = yCoords
        coords['zCoords'] = zCoords
        coords['xCoordsMonxy'] = xCoordsMonxy
        coords['yCoordsMonxy'] = yCoordsMonxy
        coords['zCoordsMonxy'] = zCoordsMonxy
        coords['xCoordsMonxz'] = xCoordsMonxz
        coords['yCoordsMonxz'] = yCoordsMonxz
        coords['zCoordsMonxz'] = zCoordsMonxz
        mode_sol['coords']     = coords
        
        mode_sol['time_taken'] = time_taken
        mode_sol['field_sampling_interval'] = field_sampling_interval
        mode_sol['full_simulation_width_with_PML'] = sxy
        msg = "Simulation took %0.2f minutes to run." % (time_taken/60)
        print(msg)
        if send_to_slack:
            ws.post_message_to_slack(msg, slack_channel=slack_channel, thread_ts = thread_ts) 

        if grab_fields == True:
            print("Getting the field data from the h5 files ...")
            monitor_fields = {}
            for plane in ['xy','xz']:
                field_arrays = []
                for idx, field_name in enumerate(['e','h']):
                    field_data = {}
                    with h5py.File('fiber_launcher-'+mode_sol[f'{field_name}_{plane}_slices_fname_h5'],'r') as h5_file:
                        h5_keys = list(h5_file.keys())
                        for h5_key in h5_keys:
                            datum = np.array(h5_file[h5_key])
                            datum = np.transpose(datum,(1,0,2))
                            field_data[h5_key] = datum
                    field_array = np.zeros((3,)+datum.shape, dtype=np.complex_)
                    field_parts  = f'{field_name}x {field_name}y {field_name}z'.split(' ')
                    for idx, field_component in enumerate(field_parts):
                        field_array[idx] = 1j*np.array(field_data[field_component+'.i'])
                        field_array[idx] += np.array(field_data[field_component+'.r'])
                    field_arrays.append(field_array)
                field_arrays = np.array(field_arrays)
                monitor_fields[plane] = field_arrays
        effective_resolution = monitor_fields['xy'].shape[2]/clear_aperture
        num_time_samples =  monitor_fields['xy'].shape[-1]
        sampling_times = np.linspace(0, run_time, num_time_samples)

        print("MEEP-adjusted resolution: %.2f px/μm" % effective_resolution)

        print("Calculating basic plots for the end time ...")
        Ey_final_sag = monitor_fields['xz'][0,1,:,:,-1]
        extent = [-simWidth/2, simWidth/2, -fiber_height/2, fiber_height/2]
        plotField = np.real(Ey_final_sag)
        fig, ax = plt.subplots(figsize=(3, 3*fiber_height/simWidth))
        ax.imshow(plotField, 
                cmap=cm.watermelon, 
                origin='lower',
                extent=extent,
                interpolation='none')
        ax.plot([-coreRadius,-coreRadius],[-fiber_height/2,fiber_height/2],'r:',alpha=0.3)
        ax.plot([coreRadius,coreRadius],[-fiber_height/2,fiber_height/2],'r:',alpha=0.3)
        ax.set_xlabel('x/μm')
        ax.set_ylabel('z/μm')
        ax.set_title('Re(Ey)')
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, 'sagittal-final-Ey','sagittal-final-Ey',thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()

        print("Picking a notable point at z=0 for sampling Ey at different times and z-values ...")
        numZsamples = Ey_final_sag.shape[0]
        midZsample = numZsamples // 2
        midCut     = Ey_final_sag[midZsample, :]
        goodIndex = np.argmax(np.real(midCut))
        columnChange = monitor_fields['xz'][0,1,:,goodIndex,:]
        plotField = np.real(columnChange)
        extent = [0, run_time, -fiber_height/2, fiber_height/2]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(plotField, 
                origin='lower', 
                extent=extent,
                cmap=cm.watermelon)
        ax.set_xlabel('t/ν')
        ax.set_ylabel('z/μm')
        ax.set_aspect(run_time/fiber_height)
        ax.set_title('Re(Ey) at fixed x,y')
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, 'Re(Ey) at fixed x,y','Re(Ey)-at-fixed-x,y',thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()

        print("Sampling the ground-truth modal profile ...")
        Xg, Yg, E_field_GT, H_field_GT = ws.field_sampler(funPairs, 
                                                    clear_aperture, 
                                                    effective_resolution, 
                                                    m, 
                                                    parity, 
                                                    coreRadius, 
                                                    coord_sys = 'cartesian-cartesian',
                                                    equiv_currents=False)

        print("Calculating the correlation of the monitored field against the ground truth ...")
        field_array = monitor_fields['xy'][1]
        component_name = 'hx'
        component_index = {'hx':0, 'hy':1, 'hz':2}[component_name]
        cosims = (field_array[component_index] * np.conjugate(H_field_GT[component_index][:,:,np.newaxis]))
        cosims = np.sum(cosims, axis=0)
        cosims = np.sum(cosims, axis=0)
        cosims = np.abs(cosims)
        n0s   = (field_array[component_index] * np.conjugate(field_array[component_index]))
        n0s   = np.sum(n0s, axis=0)
        n0s   = np.sum(n0s, axis=0)
        n0s   = np.real(np.sqrt(n0s))
        n1    = np.sqrt(np.sum(H_field_GT[component_index] * np.conjugate(H_field_GT[component_index])))
        n1    = np.real(n1)
        cosims = cosims/n1
        cosims = cosims/n0s
        cosims  = np.real(cosims)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(sampling_times, cosims)
        ax.set_xlabel('t/(μm/c)')
        ax.set_ylabel('cos(GT, ME)')
        ax.set_ylim(0,1.)
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, 'cos(GT, ME)','cos(GT, ME)',thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()
        final_cos = cosims[-1]

        print("Making a comparison plot for the last measured field ...")
        time_index = -1
        extent    = [-clear_aperture/2, clear_aperture/2, -clear_aperture/2, clear_aperture/2]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        mask   = (Xg**2 + Yg**2) < coreRadius**2
        for idx, plotFun in enumerate([np.real, np.imag]):
            ax = axes[0,idx]
            plotField = field_array[component_index][:,:,time_index]
            plotField = plotFun(plotField)
            title = ['re','im'][idx]
            title = component_name + '.' + title
            prange = np.max(np.abs(plotField))
            ax.imshow(plotField, vmin=-prange, vmax=prange, extent=extent, cmap=cm.watermelon)
            ax.set_title(title)
            ax.set_xlabel('x/μm')
            ax.set_ylabel('y/μm')
            ax.add_patch(plt.Circle((0,0), coreRadius, color='w', fill=False))
            ax = axes[1,idx]
            plotField = plotFun(H_field_GT[component_index])
            prange = np.max(np.abs(plotField))
            if idx == 0:
                vmin = -prange
                vmax = prange
                cmap = cm.watermelon
            else:
                vmin = 0
                vmax = prange
                cmap = cm.ember
            ax.imshow(plotField, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap)
            title = 'GT.' + title
            ax.set_title(title)
            ax.set_xlabel('x/μm')
            ax.set_ylabel('y/μm')
            ax.add_patch(plt.Circle((0,0), coreRadius, color='w', fill=False))
        plt.tight_layout()
        if send_to_slack:
            ws.send_fig_to_slack(fig, slack_channel, 'Comparison of last measured field','comparison-of-last-measured-field',thread_ts)
        if show_plot:
            plt.show()
        else:
            plt.close()

        summary = ws.dict_summary(mode_sol, 'SIM-'+sim_id)
        if send_to_slack:
            ws.post_message_to_slack(summary, slack_channel=slack_channel,thread_ts=thread_ts)
            ws.post_message_to_slack('CS(GT, ME) = %.3f' % final_cos, slack_channel=slack_channel,thread_ts=thread_ts)
        mem_usage = sim.get_estimated_memory_usage()
        mode_sol['mem_usage_in_MB'] = mem_usage/1024/1024
        mode_sol['monitor_field_slices'] = monitor_fields
        pkl_fname = 'sim-%s.pkl' % sim_id
        with open(pkl_fname, 'wb') as file:
            pickle.dump(mode_sol, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple CLI that accepts four parameters.')
    
    parser.add_argument('nCladding', type=float, help='The refractive index of the cladding.')
    parser.add_argument('nCore', type=float, help='The refractive index of the core.')
    parser.add_argument('coreRadius', type=float, help='The radius of the core.')
    parser.add_argument('free_space_wavelength', type=float, help='The free space wavelength.')
    
    args = parser.parse_args()
    
    main(args.nCladding, args.nCore, args.coreRadius, args.free_space_wavelength)
