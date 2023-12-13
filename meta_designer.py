#!/usr/bin/env python3

import argparse
import numpy as np
import wavesight as ws
from matplotlib import pyplot as plt
from printech import *

# This is the script that will be run to generate the phase-geometry curve
# using this, the intended focal length and other parameters that determine
# the desired properties of the resulting metalens, the radii, and the positions
# of the required cylindrical posts are calculated.

def meta_designer(sim_params_fname):
    '''
    This function takes in the filename of a metalens design specification
    and produces the corresponding metalens design.
    The design is saved to disk as an .h5 file and a summary of the design,
    together with some figures are posted to Slack for archival purposes.

    Parameters
    ----------
    sim_params_fname: str
        Filename of the simulation parameters file.
    
    Returns
    -------
    metalens_design: dict
        Dictionary with the metalens design.
    '''
    phase_factory = ws.extract_cli_arguments_and_run('./phase_factory.py')
    simulation_parameters = ws.load_from_json(sim_params_fname)
    printer("defining local variables from the simulation dictionary")
    send_to_slack = simulation_parameters['send_to_slack']
    slack_channel = simulation_parameters['slack_channel']
    show_plot = simulation_parameters['show_plot']
    MIN_FEATURE_SIZE = simulation_parameters['MIN_FEATURE_SIZE']
    if send_to_slack:
        slack_thread = ws.post_message_to_slack("Drafting a new metalens design", slack_channel=slack_channel)
        thread_ts = slack_thread['ts']
    sim_id = simulation_parameters['sim_id']
    simulation_parameters['min_post_width'] = 0.05
    simulation_parameters['max_post_width'] = simulation_parameters['lattice_const']
    simulation_parameters['post_width'] = simulation_parameters['max_post_width']
    simulation_parameters = ws.expand_sim_params(simulation_parameters)
    some_phases = phase_factory(
        simulation_parameters['nHost'],
        simulation_parameters['min_post_width'],
        simulation_parameters['max_post_width'],
        simulation_parameters['λFree'],
        simulation_parameters['lattice_const'],
        simulation_parameters['post_height'],
        simulation_parameters['num_post_widths'],
        simulation_parameters['numG'],
        np.sin(simulation_parameters['linear_polarization_direction']),
        np.cos(simulation_parameters['linear_polarization_direction'])
    )
    post_phases = np.array(some_phases['phases'])
    post_widths = np.array(some_phases['post_widths'])
    simulation_parameters['phase_geom_phases'] = post_phases
    simulation_parameters['phase_geom_widths'] = post_widths
    assert (post_phases[-1] - post_phases[0]) >= 2*np.pi, "The phase-geometry curve does not span 2pi."

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(post_widths, post_phases / (2*np.pi),'o-',ms=1.)
    ax.set_xlabel('w / $\mu$m')
    ax.set_ylabel('$\phi/2\pi$')
    ax.set_xlim(post_widths[0],post_widths[-1])
    if send_to_slack:
        ws.send_fig_to_slack(fig, slack_channel, 
                             "Phase-geometry curve", 
                             'phase-geom-%s.png' % sim_id, 
                             thread_ts = thread_ts)
    if show_plot:
        plt.show()
    else:
        plt.close()

    metalens_design = simulation_parameters
    metalens_design['min_post_width'] = MIN_FEATURE_SIZE
    rule()
    printer("producing the profile function given the material properties of the lens and focal length")
    phase_prof_fun = ws.fresnel_fun_gen(metalens_design['focal_length'], metalens_design['λFree'])

    # make a plot
    x = np.linspace(-metalens_design['mlDiameter']/2,
                    metalens_design['mlDiameter']/2,
                    500)
    ϕ = phase_prof_fun(x,0.)
    plt.figure(figsize=(10,4))
    plt.plot(x, ϕ)
    plt.xlabel('r / $\mu$m')
    plt.ylabel('$\phi$ / rad')
    if send_to_slack:
        ws.send_fig_to_slack(plt.gcf(), slack_channel, 
                             "Phase profile", 
                             'phase-profile-%s.png' % sim_id, 
                             thread_ts = thread_ts)
    if show_plot:
        plt.show()
    plt.close()

    lattice_constant = metalens_design['lattice_const']
    radius = metalens_design['mlDiameter']/2
    printer("calculating the lattice points to locate the meta-atoms")
    lattice_points = ws.hex_lattice_points(lattice_constant, radius)

    printer("calculating the necessary diameters for the required phases")
    post_phases = phase_prof_fun(lattice_points[:,0], lattice_points[:,1])
    metalens_design['post_phases'] = post_phases

    printer("clipping the phase-geometry characteristic to account for the minimum post width")
    phase_geom_phases = metalens_design['phase_geom_phases']
    phase_geom_widths = metalens_design['phase_geom_widths']
    metalens_design['phase_geom_clipped_widths'] = np.linspace(metalens_design['min_post_width'], phase_geom_widths[-1], sum(phase_geom_widths > metalens_design['min_post_width']))
    metalens_design['phase_geom_clipped_phases'] = np.interp(metalens_design['phase_geom_clipped_widths'], phase_geom_widths, phase_geom_phases)
    metalens_design['phase_geom_clipped_phases'] = metalens_design['phase_geom_clipped_phases'] - metalens_design['phase_geom_clipped_phases'][0]

    post_diameters = np.interp(post_phases, metalens_design['phase_geom_clipped_phases'], metalens_design['phase_geom_clipped_widths'])
    post_radii = post_diameters/2
    metalens_design['post_radii'] = post_radii
    metalens_design['num_posts']  = len(post_radii)
    metalens_design['lattice_points'] = lattice_points
    printer(f"metalens design consists of {metalens_design['num_posts']} posts with a maximum radius of {max(post_radii):.2f} um")
    printer("making a draft of the metalens design")
    fig, ax = plt.subplots(figsize=(8,8))
    for center, rad in zip(lattice_points, post_radii):
        circle = plt.Circle(center, rad, fill=False, color='y')
        ax.add_artist(circle)
    ax.set_aspect('equal')
    ax.set_xlim(-1.1*radius, 1.1*radius)
    ax.set_ylim(-1.1*radius, 1.1*radius)
    ax.set_xlabel('x / $\mu$m')
    ax.set_ylabel('y / $\mu$m')
    plt.tight_layout()
    if send_to_slack:
        ws.send_fig_to_slack(fig, slack_channel, 
                             "Metalens design", 
                             'meta-design-%s.pdf' % sim_id, 
                             thread_ts = thread_ts,
                             format='pdf')
    if show_plot:
        plt.show()
    else:
        plt.close()

    metalens_design_fname = 'metalens-design-%s.h5' % sim_id
    printer("saving design to %s" % metalens_design_fname)
    metalens_design['metalens_design_fname'] = metalens_design_fname
    ws.save_to_h5(metalens_design, metalens_design_fname, overwrite=True)
    summary = ws.dict_summary(metalens_design, 'meta-design-'+sim_id)
    if send_to_slack:
        ws.post_message_to_slack(summary, slack_channel=slack_channel,thread_ts=thread_ts)
    return metalens_design

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Design a metalens.')
    parser.add_argument('sim_params_fname', type=str, help='Path to the simulation parameters file.')
    args = parser.parse_args()
    meta_designer(args.sim_params_fname)
