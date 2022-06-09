#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL2_innerloop.py

Script to formulate and start CCD problem at level 2 using
linear models
"""
import numpy as np
from copy import deepcopy
import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance
from weis.aeroelasticse.CaseGen_General import case_naming
from pCrunch import LoadsAnalysis, PowerProduction
from pCrunch.io import OpenFASTOutput
from dtqpy.src.DTQPy_oloc import DTQPy_oloc
from dtqpy.src.DTQPy_static import DTQPy_static
from wisdem.glue_code.gc_WT_InitModel       import yaml2openmdao
from wisdem.glue_code.gc_PoseOptimization   import PoseOptimization as PoseOptimizationWISDEM
from weis.glue_code.glue_code               import WindPark


def checkInfeasibility(pwr):
    
    infeasible = (pwr == 0)
    inf_ind = np.where(infeasible)[0]+1
    return inf_ind

def Calc_LCOE(Turbine_Cost,AEP,MR,opex_per_kW,bos_per_kW,fcr,wlf):
    
    prob = om.Problem()
    prob.model = PlantFinance()
    prob.setup() 
    
    ncase = len(Turbine_Cost)
    lcoe = np.zeros((ncase,))
    
    for i in range(ncase):
        # Set variable inputs with intended units
        prob.set_val("machine_rating", MR, units="MW")
        prob.set_val("tcc_per_kW", Turbine_Cost[i]/MR, units="USD/MW")
        prob.set_val("turbine_number", 1)
        prob.set_val("opex_per_kW", opex_per_kW, units="USD/kW/yr")
        prob.set_val("fixed_charge_rate", fcr)
        prob.set_val("bos_per_kW", bos_per_kW[i]/MR, units="USD/MW")
        prob.set_val("wake_loss_factor", wlf)
        prob.set_val("turbine_aep", AEP[i], units="kW*h")
        
        prob.run_model()
        lcoe[i] = prob.get_val('lcoe',units = 'USD/kW/h')
        
    # prob.model.list_inputs(units=True)
    # prob.model.list_outputs(units=True)

    return lcoe

def Calc_TC(wt_init, modeling_options, analysis_options,DV):
    
    # create a copy of modeling,analysis, geometry options
    MO = deepcopy(modeling_options)
    AO = deepcopy(analysis_options)
    TM = deepcopy(wt_init)
    
    CS = DV[0,0]; BV = DV[0,1]
    
    # switch off important flags
    MO['Level1']['flag'] = False
    MO['Level2']['flag'] = False
    MO['Level3']['flag'] = False
    MO['ROSCO']['flag'] = False
    AO['recorder']['flag'] = False
    AO['driver']['optimization']['flag'] = False
    AO['driver']['design_of_experiments']['flag'] = False
    TM['costs']['turbine_number'] = 1
    
    opex_per_kW = TM['costs']['opex_per_kW']
    fcr = TM['costs']['fixed_charge_rate']
    wlf = TM['costs']['wake_loss_factor']
    
    
    # update values
    
    # column spacing
    TM["components"]["floating_platform"]["joints"][2]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][3]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][4]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][5]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][6]["location"][0] = CS
    TM["components"]["floating_platform"]["joints"][7]["location"][0] = CS
  
    # ballast volume
    TM["components"]["floating_platform"]["members"][1]["internal_structure"]["ballasts"][0]["volume"] = BV
    TM["components"]["floating_platform"]["members"][2]["internal_structure"]["ballasts"][0]["volume"] = BV
    TM["components"]["floating_platform"]["members"][3]["internal_structure"]["ballasts"][0]["volume"] = BV
    
    # pose optimization
    myopt = PoseOptimizationWISDEM(TM, MO, AO)
    
    # initialize the open mdao problem
    wt_opt = om.Problem(model=WindPark(modeling_options=MO, opt_options=AO))
    wt_opt.setup()
    
    # assign the different values for the various subsystems
    wt_opt = yaml2openmdao(wt_opt, MO, TM, AO)
    wt_opt = myopt.set_initial(wt_opt, TM)
    
    # run model
    wt_opt.run_model()
    
    # extract values
    MR = wt_opt.get_val('financese.machine_rating',units = 'MW')
    Cost_turbine_MW = wt_opt.get_val('financese.tcc_per_kW', units='USD/MW')[0]
    bos = wt_opt.get_val('financese.bos_per_kW', units='USD/MW')[0]
    Turbine_Cost = Cost_turbine_MW*MR
    bos_Cost = bos*MR
      
    return Turbine_Cost,MR,opex_per_kW,bos_Cost,fcr,wlf

def Calc_AEP(summary_stats,dlc_generator,Turbine_class):
    
    
    idx_pwrcrv = []
    U = []
    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].label == '1.1':
            idx_pwrcrv = np.append(idx_pwrcrv, i_case)
            U = np.append(U, dlc_generator.cases[i_case].URef)

    stats_pwrcrv = summary_stats.iloc[idx_pwrcrv].copy()
    
    if len(U) > 1:
        pp = PowerProduction(Turbine_class)
        pwr_curve_vars   = ["GenPwr", "RtAeroCp", "RotSpeed", "BldPitch1"]
        AEP, perf_data = pp.AEP(stats_pwrcrv, U, pwr_curve_vars)
    else:
        AEP = 0
    
    return AEP

# Wrapper for actually running dtqp with a single input, useful for running in parallel
def run_dtqp(dtqp_input):
    
    # extract calltype
    calltype = dtqp_input['calltype']
    
    # get number of linearized models
    nl = len(dtqp_input['LinearTurbine'].u_h)
    
    # if nl ==1, run DTQPy_static else run DTQPy_oloc
    if nl>1:
        T,U,X,Y = DTQPy_oloc(dtqp_input['LinearTurbine'],dtqp_input['dist'],dtqp_input['dtqp_constraints'],plot=dtqp_input['plot'])
    elif nl ==1:
        T,U,X,Y = DTQPy_static(dtqp_input['LinearTurbine'],dtqp_input['dist'],dtqp_input['dtqp_constraints'],plot=dtqp_input['plot'])
    
    if calltype == 'OPT':
        # Shorten output names from linearization output to one like level3 openfast output
        # This depends on how openfast sets up the linearization output names and may break if that is changed
        OutList     = [out_name.split()[1][:-1] for out_name in dtqp_input['LinearTurbine'].DescOutput]
    
        # Turn OutData into dict like in ROSCO_toolbox
        OutData = {}
        for i, out_chan in enumerate(OutList):
            OutData[out_chan] = Y[:,i]
    
        # Add time to OutData
        OutData['Time'] = T.flatten()
    
        output = OpenFASTOutput.from_dict(OutData, dtqp_input['case_name'],magnitude_channels=dtqp_input['magnitude_channels'])
        #output.df.to_pickle(os.path.join(dtqp_input['run_dir'],dtqp_input['case_name']+'.p'))
        
        return output
    
    elif calltype == 'PLOT':
        results = {'CaseName':dtqp_input['case_name'],
                   'time':T,
                   'states':X,
                   'controls':U,
                   'outputs':Y
                   }
        
        return results
        
        

def innerloop(LinearTurbine,level2_disturbances,nw,analysis_options,modeling_options,wt_init,DV,dlc_generator,Turbine_class,calltype,opttype=None):
    
    # Set up constraints
    dtqp_constraints = {}
    
    # reference genspeed
    fst_vt = {}
    fst_vt['DISCON_in'] = {}
    fst_vt['DISCON_in']['PC_RefSpd'] = 0.7853192931562493
    
   # Control constraints that are supported
    control_const = analysis_options['constraints']['control']
    
    # Rotor overspeed
    if control_const['rotor_overspeed']['flag']:
        desc = 'ED First time derivative of Variable speed generator DOF (internal DOF index = DOF_GeAz), rad/s'
        if desc in LinearTurbine.DescStates:
            dtqp_constraints[desc] = [0,(1 + control_const['rotor_overspeed']['max']) * fst_vt['DISCON_in']['PC_RefSpd'] ]
        else:
            raise Exception('rotor_overspeed constraint is set, but ED GenSpeed is not a state in the LinearModel')
    else:
        desc = 'ED First time derivative of Variable speed generator DOF (internal DOF index = DOF_GeAz), rad/s'
        dtqp_constraints[desc] = [0,(1 + 0.2) * fst_vt['DISCON_in']['PC_RefSpd'] ]
        
    if control_const['Max_PtfmPitch']['flag']:
        desc = 'ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad'
        if desc in LinearTurbine.DescStates:
            dtqp_constraints[desc] = [control_const['Max_PtfmPitch']['min'] * np.deg2rad(1),control_const['Max_PtfmPitch']['max'] * np.deg2rad(1)]
        else:
            raise Exception('Max_PtfmPitch constraint is set, but ED PtfmPitch is not a state in the LinearModel')
    else:
        desc = 'ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad'
        dtqp_constraints[desc] = [-6 * np.deg2rad(1),6 * np.deg2rad(1)]
    
    # Loop throught and call DTQP for each disturbance
    case_names = case_naming(len(level2_disturbances),'oloc')
    
    # magnitude channels
    magnitude_channels = {
        "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
        "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
        "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
        }
    
    plot = True

    dtqp_input_list = []
    
    la = LoadsAnalysis(
            outputs=[],
        )
    
    for i_oloc, dist in enumerate(level2_disturbances): 
        dtqp_input  = {}
        dtqp_input['LinearTurbine']         = LinearTurbine
        dtqp_input['dist']                  = dist
        dtqp_input['dtqp_constraints']      = dtqp_constraints
        dtqp_input['plot']                  = plot
        dtqp_input['case_name']             = case_names[i_oloc]
        dtqp_input['magnitude_channels']    = magnitude_channels
        dtqp_input['calltype'] = calltype

        dtqp_input_list.append(dtqp_input)
        
    if calltype == 'OPT':
        # run optimization and calculate LCOE
        
        # calculate turbine, balance of system costs
        TC,MR,opex_per_kW,bos_per_kW,fcr,wlf = Calc_TC(wt_init, modeling_options, analysis_options,DV)
        
        # evaluate inner-loop response from DTQPy
        output_list = []
        for dtqp_input in dtqp_input_list:
            output_list.append(run_dtqp(dtqp_input))
            
        # Collect outputs
        ss = {}
        et = {}
        dl = {}
        dam = {}
        ct = []
        
        # process outputs
        for output in output_list:
            _name, _ss, _et, _dl, _dam = la._process_output(output)
            ss[_name] = _ss
            et[_name] = _et
            dl[_name] = _dl
            dam[_name] = _dam
        
        # post process
        summary_stats, extreme_table, DELs, Damage = la.post_process(ss, et, dl, dam)
        
        infeasible_ind = checkInfeasibility(summary_stats.loc[:, ("GenPwr", "mean")].to_frame())
        
        # calculate AEP
        AEP = Calc_AEP(summary_stats,dlc_generator,Turbine_class)
        
        # calculate LCOE
        LCOE = Calc_LCOE(TC,AEP,MR,opex_per_kW,bos_per_kW,fcr,wlf)
        
        # extract mean wind speed
        pwr_avg = summary_stats.loc['oloc_07', ("GenPwr", "mean")]
        
        # store results in a dict
        Results = {'LCOE':LCOE,'AEP':AEP,'INF_ind':infeasible_ind,'Pavg':pwr_avg}
        
        # return results
        return Results
        
    
    elif calltype == 'PLOT':
        # initialize list
        result_list = []
        
        # run dtqp and append list
        for dtqp_input in dtqp_input_list:
            result_list.append(run_dtqp(dtqp_input))
        
        # return result    
        return result_list
    
    else:
        raise('Wrong option provided')
        
    
