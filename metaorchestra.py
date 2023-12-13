#!/usr/bin/env python3

import time
import argparse
import numpy as np
from printech import *
from datapipes import *
from misc import rando_id
from fiber_bundle import *
from meta_designer import *
from misc import dict_summary
from sniffer import output_vigilante

#[paramExpand-Calc]
# This script also controls all the sequences of simulations and 
# calculations that are necessary.

def param_expander(sim_params):
    '''
    Takes a dictionary of simulation parameters and calculates
    the dependent parameters. It also generates a random id
    for the simulation and adds it to the dictionary.
    '''
    mlDiameter = sim_params['mlDiameter']
    nCore = sim_params['nCore']
    nCladding = sim_params['nCladding']
    nBetween = sim_params['nBetween']
    nHost = sim_params['nHost']
    parallel_MEEP = sim_params['parallel_MEEP']
    λFree = sim_params['λFree']
    coreRadius = sim_params['coreRadius']
    emDepth = sim_params['emDepth']
    emDepth_Δz = sim_params['emDepth_Δz']
    emDepth_Δxy = sim_params['emDepth_Δxy']
    post_height = sim_params['post_height']
    # using the emitter depth and the size of the core
    rule()
    printer("calculating dependent model parameters")
    rule()
    mlRadius = mlDiameter/2.
    printer("determining the size of the gap between metalens and the end of fiber")
    NA_fiber = np.sqrt(nCore**2 - nCladding**2)
    sim_params['NA_fiber'] = NA_fiber
    β_fiber  = np.arcsin(NA_fiber/nBetween)
    wg_to_ml = (mlRadius - coreRadius) / np.tan(β_fiber)
    assert wg_to_ml >= 0, "Metalens is too small for fiber."
    sim_params['wg_to_ml'] = wg_to_ml
    printer("defining the spatial resolution of the simulation")
    Δfields = λFree/10.
    sim_params['Δfields'] = Δfields
    # calculate the permitivity of the different media
    printer("calculating corresponding permitivitty for given refractive indices")
    εCore     = nCore**2
    εCladding = nCladding**2
    εFree     = nBetween**2
    εHost     = nHost**2
    sim_params['εCore'] = εCore
    sim_params['εCladding'] = εCladding
    sim_params['εFree'] = εFree
    sim_params['εHost'] = εHost
    printer("calculating the focal length for grazing coupling")
    δ = coreRadius / np.tan(β_fiber)
    focal_length = 1/(nHost/emDepth + nBetween/(wg_to_ml+δ))
    sim_params['focal_length'] = focal_length
    printer("calculating a few necessary parameters for the waveguide FDTD simulations")
    pml_thickness = 2 * λFree
    sim_params['pml_thickness'] = pml_thickness
    λBetween = λFree / nBetween
    sim_params['λBetween'] = λBetween
    wg_to_EH2 = λBetween
    sim_params['wg_to_EH2'] = wg_to_EH2
    # define the distance between the input plane to ML and EH3
    EH3_to_ml = λBetween
    sim_params['EH3_to_ml'] = EH3_to_ml
    # and use the same as the distance between EH3 and the output plane of the ML simulation
    ml_to_EH4 = EH3_to_ml
    sim_params['ml_to_EH4'] = ml_to_EH4
    # define the distance between EH3 to input plane of ML
    sim_params['zProp'] =  wg_to_ml - wg_to_EH2 - EH3_to_ml
    # sim_id
    sim_id = rando_id(1)
    sim_params['sim_id'] = sim_id
    # some parameters for the second FDTD simulations
    ml_thickness = post_height
    sim_params['ml_thickness'] = ml_thickness
    runway_cell_thickness = pml_thickness + 2 * EH3_to_ml
    ml_cell_thickness     = ml_thickness
    host_cell_thickness   = 2*ml_to_EH4 + pml_thickness
    sim_params['runway_cell_thickness'] = runway_cell_thickness
    sim_params['ml_cell_thickness'] = ml_cell_thickness
    sim_params['host_cell_thickness'] = host_cell_thickness
    full_sim_height = runway_cell_thickness + ml_cell_thickness + host_cell_thickness
    sim_params['full_sim_height'] = full_sim_height
    run_time_2 = (nHost + nBetween) * full_sim_height
    sim_params['run_time_2'] = run_time_2
    if 'zmin' not in sim_params:
        printer("since zmin was not provided it is being calculated from the depth and uncertainty of the emitter position")
        zmin = (emDepth - ml_to_EH4 - emDepth_Δz)
        zmax = (emDepth - ml_to_EH4 + emDepth_Δz)
        sim_params['zmin'] = zmin
        sim_params['zmax'] = zmax
    else:
        assert 'zmax' in sim_params, "since zmin was provided, zmax must also be provided"
        zmin = sim_params['zmin']
        zmax = sim_params['zmax']
        printer("since zmin and zmax were provided, these are overriding the values calculated from the depth and uncertainty of the emitter position")
    if 'xymin' not in sim_params:
        printer("since xymin was not provided it is being calculated from the uncertainty in the lateral position of the emitter")
        xymin = -emDepth_Δxy
        xymax =  emDepth_Δxy
        sim_params['xymin'] = xymin
        sim_params['xymax'] = xymax
    else:
        assert 'xymax' in sim_params, "since xymin was provided, xymax must also be provided"
        printer("since xymin and xymax were provided, these are overriding the inferred values from the uncertainty in the lateral position of the emitter")
    printer("calculating the array of z values for the EH5 volume")
    num_zProps = int((zmax-zmin)*sim_params['MEEP_resolution'])+1
    zProps = np.linspace(sim_params['zmin'] - sim_params['ml_to_EH4'],
                         sim_params['zmax'] - sim_params['ml_to_EH4'],
                         num_zProps
                        )
    sim_params['zProps']     = zProps
    sim_params['num_zProps'] = num_zProps
    sim_params['start_time'] = time.time()
    if parallel_MEEP:
        if 'MEEP_num_cores' not in sim_params:
            sim_params['MEEP_num_cores']  = 4
        sim_params['python_bin_MEEP'] = 'mpirun -np %d %s -m mpi4py' % (sim_params['MEEP_num_cores'], sim_params['python_bin'])
    else:
        sim_params['MEEP_num_cores'] = 1
        sim_params['python_bin_MEEP'] = sim_params['python_bin']
    return sim_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a json config file')
    parser.add_argument('configfile', help='Name of json config file')
    args = parser.parse_args()
    # read the config file
    console.print(Rule("-- Metaorchestra", style="bold red", align="left"))
    printer("reading the configuration file %s" % args.configfile)
    sim_params = load_from_json(args.configfile)
    expanded_params = param_expander(sim_params)
    table_rows = dict_summary(expanded_params, header='', bullet='', aslist=True, split=True)
    rule()
    table(['param','value'], table_rows, title='Expanded parameters')
    expanded_fname = (args.configfile).replace('.jsonc', '-expanded.json')
    printer("clalculating dependent parameters and saving to %s" % expanded_fname)
    expanded_params['expanded_fname'] = expanded_fname
    save_to_json(expanded_fname, expanded_params)
    rule()
    printer("designing the required metasurface")
    metalens_design = meta_designer(expanded_fname)
    rule()
    printer("printing the metalens design")
    rule()
    printer(metalens_design)
    rule()
    printer("executing the script fiber_bundle.py which will:", tail='')
    printer("> launch the mode calculator", tail=',')
    printer("> schedule the FDTD simulations for launching the propagating modes", tail=',')
    printer("> schedule the jobs for propagating the fields across the gap", tail=',')
    printer("> schedule the jobs for transporting the fields across the metasurface", tail=',')
    printer("> schedule the jobs for propagating the transmitted fields to the final volume")
    rule()
    waveguide_dir = fan_out(expanded_params)
    rule()
    output_vigilante(waveguide_dir)
