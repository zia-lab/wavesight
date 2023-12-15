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

def wg_sim_height_fun(λFree, pml_thickness):
    return (10 * λFree + 2 * pml_thickness)

def param_expander(config_params):
    '''
    Takes a dictionary of simulation parameters and calculates
    the dependent parameters. It also generates a random id
    for the simulation and adds it to the dictionary.
    '''
    mlDiameter    = config_params['mlDiameter']
    nCore         = config_params['nCore']
    nCladding     = config_params['nCladding']
    nBetween      = config_params['nBetween']
    nHost         = config_params['nHost']
    parallel_MEEP = config_params['parallel_MEEP']
    λFree         = config_params['λFree']
    coreRadius    = config_params['coreRadius']
    emDepth       = config_params['emDepth']
    emDepth_Δz    = config_params['emDepth_Δz']
    emDepth_Δxy   = config_params['emDepth_Δxy']
    post_height   = config_params['post_height']
    spacer_ML_sim = config_params['spacer_ML_sim']
    spacer_wg_sim = config_params['spacer_wg_sim']

    rule()
    printer("calculating dependent model parameters")
    rule()

    mlRadius = mlDiameter/2.

    printer("determining the size of the gap between metalens and the end of fiber")
    fiber_NA = np.sqrt(nCore**2 - nCladding**2)
    config_params['fiber_NA'] = fiber_NA
    fiber_β  = np.arcsin(fiber_NA/nBetween)
    config_params['fiber_β']  = fiber_β
    wg_to_ml = (mlRadius - coreRadius) / np.tan(fiber_β)
    assert wg_to_ml >= 0, "Metalens is too small for fiber."
    config_params['wg_to_ml'] = wg_to_ml
    λBetween = λFree / nBetween
    config_params['λBetween'] = λBetween

    printer("defining the spatial resolution of the simulation")
    Δfields = λFree/10.
    config_params['Δfields'] = Δfields

    # calculate the permitivity of the different media
    printer("calculating corresponding permitivitty for given refractive indices")
    εCore     = nCore**2
    εCladding = nCladding**2
    εFree     = nBetween**2
    εHost     = nHost**2
    config_params['εCore']     = εCore
    config_params['εCladding'] = εCladding
    config_params['εFree']     = εFree
    config_params['εHost']     = εHost

    # calculate the focal length
    printer("calculating the focal length for grazing coupling")
    δ = coreRadius / np.tan(fiber_β)
    focal_length = 1/(nHost/emDepth + nBetween/(wg_to_ml+δ))
    config_params['focal_length'] = focal_length

    # calculate necessary parameters for waveguide FDTD simulations
    printer("calculating a few necessary parameters for the waveguide FDTD simulations")
    # fix the thickness of the PML proportional to the wavelength inside the intermediate medium
    pml_thickness                  = λBetween
    config_params['pml_thickness'] = pml_thickness
    # fix the distance from the end face of the waveguide to the output plane EH2
    wg_to_EH2                      = λBetween
    config_params['wg_to_EH2']     = wg_to_EH2
    # fix the distance between the input plane EH1 to the end face of the waveguide
    EH1_to_wg                      = λBetween
    config_params['EH1_to_wg']     = EH1_to_wg

    # calculate a few additional goodies
    kFree                          = 2*np.pi / λFree
    config_params['kFree']         = kFree
    frequency_f                    = kFree / (2*np.pi)
    config_params['frequency_f']   = frequency_f
    base_period                    = 1. / frequency_f
    config_params['base_period']   = base_period
    unit_of_length_in_m            = config_params['unit_of_length_in_m']
    speed_of_light_in_m_per_s      = config_params['speed_of_light_in_m_per_s']
    unit_of_time_in_fs             = unit_of_length_in_m / speed_of_light_in_m_per_s * 1e15
    config_params['unit_of_time_in_fs'] = unit_of_time_in_fs

    # the width of this FDTD simulation cannot be determined here
    # since it depends on how thick the cladding will be determined to be
    # the the mode solver
    full_wg_sim_height         = (pml_thickness 
                                  + spacer_wg_sim
                                  + EH1_to_wg 
                                  + wg_to_EH2
                                  + spacer_wg_sim
                                  + pml_thickness)
    config_params['full_wg_sim_height'] = full_wg_sim_height


    # define the distance between the input plane to ML and EH3
    EH3_to_ml = λBetween
    config_params['EH3_to_ml'] = EH3_to_ml

    # and use the same as the distance between EH3 and the output plane of the ML simulation
    ml_to_EH4 = EH3_to_ml
    config_params['ml_to_EH4'] = ml_to_EH4

    # define the distance between EH3 to input plane of ML
    config_params['EH2_to_EH3'] =  wg_to_ml - wg_to_EH2 - EH3_to_ml
    # sim_id
    sim_id = rando_id(1)
    config_params['sim_id'] = sim_id

    # parameters for the FTDT simulations that see fields across the ML
    ml_thickness               = post_height
    config_params['ml_thickness'] = ml_thickness
    runway_cell_thickness      = pml_thickness + EH3_to_ml + spacer_ML_sim
    ml_cell_thickness          = ml_thickness
    host_cell_thickness        = pml_thickness + ml_to_EH4 + spacer_ML_sim
    config_params['runway_cell_thickness'] = runway_cell_thickness
    config_params['ml_cell_thickness']     = ml_cell_thickness
    config_params['host_cell_thickness']   = host_cell_thickness
    full_ml_sim_height                  = runway_cell_thickness + ml_cell_thickness + host_cell_thickness
    config_params['full_ml_sim_height']    = full_ml_sim_height

    # the run time of the ML FDTD simulations I simply set the time a wave in free space
    # would take to transverse the entire thickness of the ML simulation cell
    run_time_2                          = (nHost + nBetween) * full_ml_sim_height
    config_params['run_time_2']            = run_time_2

    if 'zmin' not in config_params:
        printer("since zmin was not provided it is being calculated from the depth and uncertainty of the emitter position")
        zmin = (emDepth - ml_to_EH4 - emDepth_Δz)
        zmax = (emDepth - ml_to_EH4 + emDepth_Δz)
        config_params['zmin'] = zmin
        config_params['zmax'] = zmax
    else:
        assert 'zmax' in config_params, "since zmin was provided, zmax must also be provided"
        zmin = config_params['zmin']
        zmax = config_params['zmax']
        printer("since zmin and zmax were provided, these are overriding the values calculated from the depth and uncertainty of the emitter position")
    if 'xymin' not in config_params:
        printer("since xymin was not provided it is being calculated from the uncertainty in the lateral position of the emitter")
        xymin = -emDepth_Δxy
        xymax =  emDepth_Δxy
        config_params['xymin'] = xymin
        config_params['xymax'] = xymax
    else:
        assert 'xymax' in config_params, "since xymin was provided, xymax must also be provided"
        printer("since xymin and xymax were provided, these are overriding the inferred values from the uncertainty in the lateral position of the emitter")
    printer("calculating the array of z values for the EH5 volume")
    num_zProps = int((zmax-zmin)*config_params['MEEP_resolution'])+1
    zProps = np.linspace(config_params['zmin'] - config_params['ml_to_EH4'],
                         config_params['zmax'] - config_params['ml_to_EH4'],
                         num_zProps
                        )
    config_params['zProps']     = zProps
    config_params['num_zProps'] = num_zProps
    config_params['start_time'] = time.time()
    if parallel_MEEP:
        if 'MEEP_num_cores' not in config_params:
            config_params['MEEP_num_cores']  = 4
        config_params['python_bin_MEEP'] = 'mpirun -np %d %s -m mpi4py' % (config_params['MEEP_num_cores'], config_params['python_bin'])
    else:
        config_params['MEEP_num_cores'] = 1
        config_params['python_bin_MEEP'] = config_params['python_bin']
    return config_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a json config file')
    parser.add_argument('configfile', help='Name of json config file')
    args = parser.parse_args()
    # read the config file
    console.print(Rule("-- Metaorchestra", style="bold red", align="left"))
    printer("reading the configuration file %s" % args.configfile)
    config_params = load_from_json(args.configfile)
    expanded_params = param_expander(config_params)
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
