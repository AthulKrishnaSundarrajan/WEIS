#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL2_outerloop.py

Script to formulate and start CCD problem at level 2 using
linear models
"""

# import general libraries
import os
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt 
from scipy.io import savemat

# import specific weis classes and functions
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.turbsim_file import TurbSimFile 
from weis.dlc_driver.dlc_generator      import DLCGenerator

from weis.glue_code.gc_LoadInputs           import WindTurbineOntologyPythonWEIS
import weis.inputs as sch


# import innerloop
from WL2_innerloop import innerloop
from pyoptsparse import NSGA2,Optimization,SLSQP,IPOPT
import nlopt

from InterpolateLinearModels import LinearModel,EvaluateModel,BuildSurrogate       
from LoadMatrices import LoadMatrices

def PyOp_options(solver):
    
    if solver == 'SLSQP':
        options = {
        "IPRINT": 1,
        "ACC": 1e-4
        }
        
    elif solver == 'NSGA':
        options ={
            "maxGen":200,
            "PopSize":100,
            }
        
    elif solver == 'IPOPT':
        options = {'max_iter':150,
                   'tol':1e-3,
                   'print_level':1,
                   'output_file': 'IPOPT_OL.out',
                'file_print_level':5,
                }
            
    return options
 
def SaveLinModel(LM_):

    # extract values
    A_ops = LM_.A_ops
    B_ops = LM_.B_ops
    C_ops = LM_.C_ops 
    D_ops = LM_.D_ops 

    x_ops = LM_.x_ops
    u_ops = LM_.u_ops 
    y_ops = LM_.y_ops 

    u_h = LM_.u_h 

    LinModel = {"A_ops":A_ops,"B_ops":B_ops,"C_ops":C_ops,"D_ops":D_ops,
                "x_ops":x_ops,"u_ops":u_ops,'y_ops':y_ops,"u_h":u_h}

    pkl_name = "LinModelRBF.pkl"

    with open(pkl_name,'wb') as handle:
        pickle.dump(LinModel,handle)
        


class NLOPT_outerloop:
    def __init__(self,LinearModel,DLCs,analysis_options,modeling_options,wt_init,dlc_generator,Turbine_class):
        
        # assign
        self.LinearModel = LinearModel 
        self.DLCs = DLCs
        self.analysis_options = analysis_options
        self.modeling_options = modeling_options 
        self.wt_init = wt_init
        self.dlc_generator  = dlc_generator 
        self.Turbine_class = Turbine_class 
    
    
    def objective(self,x,grad):
            
            # extract values
            LinearModel = self.LinearModel
            u_h = LinearModel.u_h
            nw = len(u_h)
            
            DLCs = self.DLCs
            analysis_options = self.analysis_options 
            modeling_options = self.modeling_options 
            wt_init = self.wt_init 
            dlc_generator = self.dlc_generator 
            Turbine_class = self.Turbine_class
        
            # reshape x
            dv_ = x.reshape([1,-1])
                       
            # initialize and get data
            if nw == 1:
                data_ = dv_
            elif nw >1:
                data_ = np.zeros((len(u_h),3))
                data_[:,0:2] = dv_
                data_[:,2] = u_h
            
            # call type
            calltype = 'OPT'
            
            # evaluate matrices
            LM_ = EvaluateModel(LinearModel, data_,nw)
            
            # evaluate LCOE
            LCOE = innerloop(LM_,DLCs,nw,analysis_options,modeling_options,wt_init,dv_,dlc_generator,Turbine_class,calltype,opttype = 'NSGA')
            
           
            return LCOE[0]
        
        
        
class PyOptSparse_outerloop:
    def __init__(self,LinearModel,DLCs,analysis_options,modeling_options,wt_init,dlc_generator,Turbine_class):
        
        # assign
        self.LinearModel = LinearModel 
        self.DLCs = DLCs
        self.analysis_options = analysis_options
        self.modeling_options = modeling_options 
        self.wt_init = wt_init
        self.dlc_generator  = dlc_generator 
        self.Turbine_class = Turbine_class 
        
    def OL_objective(self,xdict):
        
        # extract values
        LinearModel = self.LinearModel
        u_h = LinearModel.u_h
        nw = len(u_h)
        
        DLCs = self.DLCs
        analysis_options = self.analysis_options 
        modeling_options = self.modeling_options 
        wt_init = self.wt_init 
        dlc_generator = self.dlc_generator 
        Turbine_class = self.Turbine_class
        
        
        # extract vector of optimization variables
        CS = xdict['CS']
        BV = xdict['BV']
        
        #
        dv_ = np.array([[CS,BV]])
        print(dv_)
        # initialize and get data
        if nw == 1:
            data_ = dv_
        elif nw >1:
            data_ = np.zeros((len(u_h),3))
            data_[:,0:2] = dv_
            data_[:,2] = u_h
        
        # call type
        calltype = 'OPT'
        
        # initialize
        funcs = {}
        
        # evaluate matrices
        LM_ = EvaluateModel(LinearModel, data_,nw)
        
        # evaluate LCOE
        LCOE = innerloop(LM_,DLCs,nw,analysis_options,modeling_options,wt_init,dv_,dlc_generator,Turbine_class,calltype,opttype = 'NSGA')
        
        # add to dict
        funcs['obj'] = LCOE
        
        # fail
        fail = False
        
        # return
        return funcs,fail
        

if __name__ == "__main__":
    
    # get directory 
    weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    # read modeling and analysis options
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # modeling options
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    #modeling_options = sch.load_modeling_yaml(fname_modeling_options)
    
    # geometry options
    fname_wt_input   = mydir + os.sep + "IEA-15-floating.yaml"
    #wt_init          = sch.load_geometry_yaml(fname_wt_input)
    
    # analysis options
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    #analysis_options            = sch.load_analysis_yaml(fname_analysis_options)
    
    wt_ontology = WindTurbineOntologyPythonWEIS(fname_wt_input, fname_modeling_options, fname_analysis_options)
    wt_init,modeling_options,analysis_options = wt_ontology.get_input_data()
    
    # load and generate dlc
    Turbine_class = wt_init["assembly"]["turbine_class"]
    ws_cut_in               = wt_init['control']['supervisory']['Vin']
    ws_cut_out              = wt_init['control']['supervisory']['Vout']
    ws_rated                = 11.2
    wind_speed_class        = wt_init['assembly']['turbine_class']
    wind_turbulence_class   = wt_init['assembly']['turbulence_class']
    
    # Extract user defined list of cases
    DLCs = modeling_options['DLC_driver']['DLCs']
    
    # Initialize the generator
    fix_wind_seeds = modeling_options['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modeling_options['DLC_driver']['fix_wave_seeds']
    metocean = modeling_options['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(ws_cut_in, ws_cut_out, ws_rated, wind_speed_class, wind_turbulence_class, fix_wind_seeds, fix_wave_seeds, metocean)

    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)


    # generate wind files
    FAST_namingOut = 'oloc'
    wind_directory = 'outputs/oloc/wind'
    if not os.path.exists(wind_directory):
        os.makedirs(wind_directory)
    rotorD = wt_init['assembly']['rotor_diameter']
    hub_height = wt_init['assembly']['hub_height']

    # from various parts of openmdao_openfast:
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases
    
    plotflag = True
    level2_disturbance = []
    
    # plot
    if plotflag:
        
        fig1,ax1 = plt.subplots(1,1)
        ax1.set_ylabel('Time [s]',fontsize = 16)
        ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)

    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)

        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(WindFile_name[i_case])
        ts_file.compute_rot_avg(rotorD/2)
        u_h         = ts_file['rot_avg'][0,:]
        
        off = 3 - min(u_h)
        ind = u_h < 3
        
        # if ind.any():
        #     u_h[ind] = u_h[ind]+off
        
        off = max(u_h) - 25
        ind = u_h > 25;
        
        # remove any windspeeds > 25 m/s
        # if ind.any():
        #     u_h[ind] = u_h[ind] - off
        
        # save DLC
        tt = ts_file['t']
        level2_disturbance.append({'Time':tt, 'Wind': u_h})
        print(np.min(u_h)); print(np.max(u_h))
        # plot
        if plotflag:
            ax1.plot(tt,u_h)
            
    # load matrices and DV from the output folder
    A,B,C,D,x,u,y,u_h,DV,DescStates,DescCntrlInpt,DescOutput = LoadMatrices('jmd_full')
    cs = np.unique(DV[:,0]); bv = np.unique(DV[:,1])
   
    # get shape
    ns,nx,nu,nw = np.shape(B); ny = len(DescOutput)
    
    # number of linmat
    nw = len(u_h);
    
    # set data
    if nw == 1:
        data = (cs,bv)
    elif nw >1 :
        data  = (cs,bv,u_h)
    
    # construct surrogate model
    Asm,sp_A = BuildSurrogate(A,data)
    Bsm,sp_B = BuildSurrogate(B,data)
    Csm,sp_C = BuildSurrogate(C,data)
    Dsm,sp_D = BuildSurrogate(D,data)
    
    xsm,sp_X = BuildSurrogate(x,data) 
    usm,sp_U = BuildSurrogate(u,data)
    ysm,sp_Y = BuildSurrogate(y,data)
    
    # add linear models
    LM_sm = LinearModel(Asm,Bsm,Csm,Dsm,xsm,usm,ysm,DescStates,DescCntrlInpt,DescOutput,nx,nu,ny,u_h)
    sp_info = {"sp_A":sp_A,"sp_B":sp_B,"sp_C":sp_C,"sp_D":sp_D,"sp_X":sp_X,"sp_U":sp_U,"sp_Y":sp_Y}
   
    debug_flag = True
    
    if debug_flag:
        
        if nw == 1:
            DV = np.array([[51.75,500]])
            
        elif nw >1:
            DV = np.zeros((nw,3))
            DV[:,0:2] = np.array([[49.333,450]]) 
            DV[:,2] = u_h
            
                       
        LM_ = EvaluateModel(LM_sm,sp_info,DV,nw,nx,nu,ny)
        
        calltype = 'OPT'
        t0 = time.time()
        Results = innerloop(LM_, level2_disturbance, nw, analysis_options, modeling_options, wt_init, DV, dlc_generator, Turbine_class, calltype)
        tf = time.time()
        t_opt = (tf-t0)/60
        
    
    analysis_ = ''
    
    if analysis_ == 'NLOPT':
        
        prob = NLOPT_outerloop(LM_sm,level2_disturbance,analysis_options,modeling_options,wt_init,dlc_generator,Turbine_class)
        
        # set up optimizer
        opt = nlopt.opt(nlopt.LN_COBYLA,2)
        
        # set bounds
        opt.set_lower_bounds([36,350])
        opt.set_upper_bounds([66,650])
        
        # set objective
        opt.set_min_objective(prob.objective)
        
        # set tolerence
        t0 = time.time()
        opt.set_xtol_rel(1e-3)
        x = opt.optimize([40.75,400])
        minf = opt.last_optimum_value()
        tf = time.time()
        
        t_opt = (tf-t0)/60
        
        calltype = 'PLOT'
        DV = x.reshape([1,-1])
        EvaluateModel(LM_sm,sp_info,DV,nw,nx,nu,ny)
        Results = innerloop(LM_, level2_disturbance, nw, analysis_options, modeling_options, wt_init, DV, dlc_generator, Turbine_class, calltype)
        
      
    elif analysis_ =='pyOPTSparse':
        
        method_ = 'IPOPT'
        
        # initialize problem
        prob = PyOptSparse_outerloop(LM_sm,level2_disturbance,analysis_options,modeling_options,wt_init,dlc_generator,Turbine_class)
        
        # instantiate problem
        optProb = Optimization("Level2_CCD",prob.OL_objective)
        
        # add objective function
        optProb.addObj("obj")
        optProb.addVar('CS', 'c', value = 36, lower=36, upper=66)
        optProb.addVar('BV', 'c', value = 350, lower=350, upper=650)
        
        print("Evaluating level 2 CCD results")
        # sol
        options = PyOp_options(method_)
        t0 = time.time()
        
        if method_ == 'SLSQP':
        
            opt = SLSQP(options=options)
            sol = opt(optProb,sens = 'FD',sensStep=1e-3)
            
        elif method_ == 'NSGA':
            opt = NSGA2(options = options)
            sol = opt(optProb)
            
        elif method_ == 'IPOPT':
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--opt",help = "optimizer",type = str, default = "IPOPT")
            args = parser.parse_args()
            
            opt = IPOPT(args,options = options)
            sol = opt(optProb,sens = 'FD')
            
        
        tf = time.time()
        
        t_opt = (tf-t0)/60
        
    elif analysis_ == 'DOE':
        
        # number of samples
        ns = 1
        
        # create samples
        CS_b = np.linspace(cs[0],cs[-1],ns)
        BV_b = np.linspace(bv[0],bv[-1],ns)
        
        # meshgrid
        CS,BV = np.meshgrid(CS_b,BV_b)
        
        DV_ = np.array([CS,BV])
        DV_ = np.transpose(DV_,[1,2,0])
        
        # initialize
        LCOE = np.zeros((ns,ns))
        AEP = np.zeros((ns,ns))
        PP = np.zeros((ns,ns))
        Pavg = np.zeros((ns,ns))
        infeasible = []
        iter_ = 0
        
        # start timer
        t0 = time.time()
        for i in range(ns):
            for j in range(ns):
                iter_+=1
                print('')
                print(iter_)
                print('')
                # extract design variables
                dv_ = np.zeros((nw,3))
                dv_[:,0:2] = DV_[i,j,:]
                dv_[:,2] = u_h
                
                
                # get linear model
                LM_ = EvaluateModel(LM_sm,sp_info,dv_,nw,nx,nu,ny)
                calltype = 'OPT'
                
                # results
                PP[i,j] = np.max(np.rad2deg(LM_.x_ops[0,:]))
                
                # Results
                Results = innerloop(LM_, level2_disturbance, nw, analysis_options, modeling_options, wt_init, dv_, dlc_generator, Turbine_class, calltype)
                
                # LCOE
                LCOE[i,j] = Results['LCOE']
                
                # AEP
                AEP[i,j] = Results['AEP']
                
                # avg power
                Pavg[i,j] = Results['Pavg']
                
                # infeasible 
                infeasible.append(Results['INF_ind'])
         
        # reshape
        infeasible = np.reshape(infeasible,(ns,ns),order = 'F')        
        
        # end timer
        tf = time.time()
        
        # calculate time
        t_opt = (tf-t0)/60
        
        # combine results
        Results = {'CS':CS,
                   'BV':BV,
                   'PP':PP,
                   'LCOE':LCOE,
                   'AEP':AEP,
                   'Pavg':Pavg,
                   'infeasible':infeasible}
       
        # pickle name
        mat_name = 'doe_10_tleq_4.mat'
        
        #savemat(mat_name,Results)
        
    elif analysis_ == 'PLOT':
        
        # set design
        DV = np.zeros((nw,3))
        DV[:,0:2] = np.array([[66,650]]) 
        DV[:,2] = u_h
        
                   
        LM_ = EvaluateModel(LM_sm,sp_info,DV,nw,nx,nu,ny)
        
        calltype = 'PLOT'
        t0 = time.time()
        Results = innerloop(LM_, level2_disturbance, nw, analysis_options, modeling_options, wt_init, DV, dlc_generator, Turbine_class, calltype)
        tf = time.time()
        t_opt = (tf-t0)/60
        
        
        res_name = ['oloc_'+str(i) for i in range(len(Results))]
        
        res_dict = dict(zip(res_name,Results))
       
        name_ = 'results_single6.mat' 
        #savemat(name_,res_dict)
            
        
        
            
        
            
        
                
                
        
        
        
        
        
        
    
    
    
    
    
    
    