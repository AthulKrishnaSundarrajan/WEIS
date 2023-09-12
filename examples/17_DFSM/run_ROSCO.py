import matplotlib.pyplot as plt 
import numpy as np
import os, platform
# ROSCO toolbox modules 
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import control_interface as ROSCO_ci
from ROSCO_toolbox.utilities import write_DISCON
from ROSCO_toolbox.inputs.validation import load_rosco_yaml

from scipy.interpolate import CubicSpline
import time as timer

# DFSM modules
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.evaluate_dfsm import evaluate_dfsm
from scipy.integrate import solve_ivp

def run_sim_ROSCO(t,x,DFSM,param):
    
    turbine_state = {}
    dt = t - param['time'][-1]
    
    if dt == 0:
        dt = 1e-4

    # extract data from param dict
    w = param['w_fun'](t)
    rpm2RadSec = 2.0*(np.pi)/60.0
    
    # populate turbine state dictionary
    if t == tf:
        turbine_state['iStatus'] = -1
    else:
        turbine_state['iStatus'] = 1
        
    # first step
    if  t == param['t0']:
        turbine_state['bld_pitch'] = np.deg2rad(param['bp_init'])
        turbine_state['gen_torque'] = param['gen_torque'][-1]*1000
        
    else:
        turbine_state['bld_pitch'] = np.deg2rad(param['blade_pitch'][-1])
        turbine_state['gen_torque'] = param['gen_torque'][-1]*1000
    
        
    turbine_state['t'] = t
    turbine_state['dt'] = dt
    turbine_state['ws'] = w
    turbine_state['gen_speed'] = x[0]*rpm2RadSec
    turbine_state['gen_eff'] = param['VS_GenEff']/100
    turbine_state['rot_speed'] = x[0]*rpm2RadSec/param['WE_GearboxRatio']
    turbine_state['Yaw_fromNorth'] = 0
    turbine_state['Y_MeasErr'] = 0
    #turbine_state['FA_Acc'] = x[2]
    
    # call ROSCO to get control values
    gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)
    
    gen_torque = gen_torque/1000
    bld_pitch = np.rad2deg(bld_pitch)
    
    u = np.array([w,gen_torque,bld_pitch])
    
    # update param list
    param['gen_torque'].append(gen_torque)
    param['blade_pitch'].append(bld_pitch)
    param['time'].append(t)
    
    # combine
    inputs = np.hstack([u,x])
    
    # evaluate dfsm
    dx = evaluate_dfsm(DFSM,inputs,'deriv')
    
    return dx
    

if __name__ == '__main__':
    
    # path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # parameter file
    parameter_filename = os.path.join(this_dir,'RM1_MHK.yaml')
    
    # tune directory
    
    # load
    inps = load_rosco_yaml(parameter_filename)
    path_params         = inps['path_params']
    turbine_params      = inps['turbine_params']
    controller_params   = inps['controller_params']
    
    # path to DISCON library
    lib_name = os.path.join('/home/athulsun/WEIS-AKS/local/lib','libdiscon.so')
    
    # Load turbine data from openfast model
    turbine = ROSCO_turbine.Turbine(turbine_params)
    
    cp_filename = os.path.join(this_dir,path_params['FAST_directory'],path_params['rotor_performance_filename'])
    
    turbine.load_from_fast(
        path_params['FAST_InputFile'],
        os.path.join(this_dir,path_params['FAST_directory']),
        rot_source='txt',txt_filename=os.path.join(this_dir,path_params['FAST_directory'],path_params['rotor_performance_filename'])
        )

    
    # Tune controller 
    controller      = ROSCO_controller.Controller(controller_params)
    controller.tune_controller(turbine)
    
    # Write parameter input file
    param_filename = os.path.join(this_dir,'DISCON.IN')
    write_DISCON(
      turbine,controller,
      param_file=param_filename, 
      txt_filename=cp_filename
      )
    

    # datapath
    datapath = this_dir + os.sep + 'outputs' + os.sep + 'DFSM_MHK_CT4' + os.sep + 'openfast_runs/rank_0'
    
    # get the entire path
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['GenSpeed']
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1']
    reqd_outputs = [] #,'TwrBsMyt'
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1]),
                  'control_scaling_factor': np.array([1,1,1]),
                  'output_scaling_factor': np.array([10,1000])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False],
                   'states_filter_type': [[]],
                   'states_filter_tf': [[],[],[1]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [1,0,0],
                   'outputs_filter_flag': [False,False]
                   }
    
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=5,add_dx2 = True)
    
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim
    
    # plot data
    plot_inputs(sim_detail,4,'separate')
    
    # split of training-testing data
    n_samples = [200]
    
    for isample in n_samples:
        dfsm_model = DFSM(sim_detail,n_samples = isample,L_type = 'LTI',N_type = 'GPR')
        dfsm_model.construct_surrogate()
        
        test_data = dfsm_model.test_data[0]
        
        bp_init = test_data['controls'][0,2]
        bp_of = test_data['controls'][:,2]
        bp_mean = np.mean(test_data['controls'][:,2])
        
        nt = 1000
        wind_speed = test_data['controls'][:,0] # np.ones((nt,))*2.5 #test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 700
        dt = 0.0; t1 = t0 + dt
        time = np.linspace(t0,tf,nt)
        
        w_pp = CubicSpline(time, wind_speed)
        w_fun = lambda t: w_pp(t)
        w0 = w_fun(0)
        bp0 = bp_mean
        gt0 = 8.3033588
        
        tspan = [t1,tf]
        
        x0 = np.mean(test_data['states'],0)
        states_ = test_data['states']
        
        
        # Load controller library
        #controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test')
        
        # hardcoded for now
        param = {'VS_GenEff':94.40000000000,
                 'WE_GearboxRatio':53.0,
                 'VS_RtPwr':500,
                 'time':[t0],
                 'dt':[dt],
                 'blade_pitch':[bp0],
                 'gen_torque':[gt0],
                 't0':t1,
                 'tf':tf,
                 'bp_init':bp_mean,
                 'w_fun':w_fun,
                 't':[],
                 'controller_interface':ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename)}
        
        # solver method and options
        solve_options = {'method':'RK45','rtol':1e-7,'atol':1e-7}
        
        t1 = timer.time()
        sol =  solve_ivp(run_sim_ROSCO,tspan,x0,method=solve_options['method'],args = (dfsm_model,param),rtol = solve_options['rtol'],atol = solve_options['atol'])
        t2 = timer.time()
        
        time = sol.t 
        states = sol.y
        states = states.T
        
        # kill controller
        param['controller_interface'].kill_discon()
        
        tspan = [0,tf]
        blade_pitch = np.array(param['blade_pitch'])
        gen_torque = np.array(param['gen_torque'])
        
        fig,ax = plt.subplots(4,1)
        
        ind = 0
        
        ax[ind].plot(time,w_fun(time))
        ax[ind].set_title('Flow Speed [m/s]')
        ax[ind].set_xlim(tspan)
        ind+=1
        
        ax[ind].plot(param['time'],gen_torque)
        ax[ind].set_title('GenTq [kNm]')
        ax[ind].set_xlim(tspan)
        ind+=1
        
        ax[ind].plot(param['time'],blade_pitch)
        #ax[ind].plot(time,bp_of)
        ax[ind].set_title('BldPitch [deg]')
        ax[ind].set_xlim(tspan)
        ax[ind].set_ylim([8,10.5])
        ind+=1
        
        ax[ind].plot(time,states[:,0])
        ax[ind].set_title('GenSpeed [rpm]')
        ax[ind].set_xlim(tspan)
        ax[ind].set_ylim([590,640])
        ind+=1
        
        # ax[ind].plot(time,states[:,0])
        # ax[ind].set_title('PtfmPitch [deg]')
        # ax[ind].set_xlim(tspan)
        # ind+=1
        
        fig.subplots_adjust(hspace = 0.85)
        
        fig,ax = plt.subplots(1)
        ax.plot(np.linspace(t0,tf,nt),bp_of,label = 'OpenFAST')
        ax.plot(param['time'],blade_pitch,label = 'DFSM')
        ax.set_title('BldPitch [deg]')
        ax.set_xlim(tspan)
        ax.set_ylim([8,10.5])
        ax.legend(ncol = 2)
        
        fig,ax = plt.subplots(1)
        ax.plot(np.linspace(t0,tf,nt),states_[:,0],label = 'OpenFAST')
        ax.plot(time,states[:,0],label = 'DFSM')
        ax.set_title('GenSpeed [rpm]')
        ax.set_xlim(tspan)
        ax.set_ylim([580,630])
        ax.legend(ncol = 2)
        
        
        
        
        
        
        
        
        


