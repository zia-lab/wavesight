#!/usr/bin/env python3

from datapipes import *
from misc import rando_id
import numpy as np
import argparse
from meta_designer import *
from printech import *
from misc import dict_summary
from fiber_bundle import *
from sniffer import output_vigilante

#[paramExpand-Calc]
# This script also controls all the sequences of simulations and 
# calculations that are necessary.

def param_expander(sim_params):
    mlDiameter = sim_params['mlDiameter']
    nCore = sim_params['nCore']
    nCladding = sim_params['nCladding']
    nBetween = sim_params['nBetween']
    nHost = sim_params['nHost']
    λFree = sim_params['λFree']
    coreRadius = sim_params['coreRadius']
    emDepth = sim_params['emDepth']
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
    wg_to_mon = 1.5 * λBetween
    sim_params['wg_to_mon'] = wg_to_mon
    # define the distance between the input plane to ML and EH3
    EH3_to_ml = λBetween
    sim_params['EH3_to_ml'] = EH3_to_ml
    # and use the same as the distance between EH3 and the output plane of the ML simulation
    ml_to_EH4 = EH3_to_ml
    sim_params['ml_to_EH4'] = ml_to_EH4
    # define the distance between EH3 to ML
    sim_params['zProp'] =  wg_to_ml - wg_to_mon - EH3_to_ml
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
    return sim_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a json config file')
    parser.add_argument('configfile', help='Name of json config file')
    args = parser.parse_args()
    # read the config file\
    console.print(Rule("-- Metaorchestra", style="bold red", align="left"))
    printer("reading the configuration file %s" % args.configfile)
    sim_params = load_from_json(args.configfile)
    expanded_params = param_expander(sim_params)
    table_rows = dict_summary(expanded_params, header='', bullet='', aslist=True, split=True)
    rule()
    table(['param','value'], table_rows, title='Expanded parameters')
    expanded_fname = (args.configfile).replace('.json', '-' + expanded_params['sim_id']+'.json')
    printer("calculating dependent parameters and saving to %s" % expanded_fname)
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
    waveguide_dir = fan_out(expanded_params['nCladding'], expanded_params['nCore'],
            expanded_params['coreRadius'], expanded_params['λFree'],
            expanded_params['nBetween'], expanded_params['autorun'],
            expanded_params['zProp'], expanded_params['MEEP_resolution'],
            expanded_params['req_run_mem_in_GB'],
            expanded_fname)
    rule()
    output_vigilante(waveguide_dir)
