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
from scipy.interpolate import CubicSpline,interp1d
from scipy.interpolate import CubicSpline
import time as timer

# DFSM modules
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension,calculate_time
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
from dfsm.evaluate_dfsm import evaluate_dfsm
from scipy.integrate import solve_ivp


def run_sim_ROSCO(t,x,DFSM,param):
    
    #print(DFSM.gen_speed_ind)
    
    turbine_state = {}
    dt = t - param['time'][-1]
    
    if dt == 0:
        dt = 1e-4

    # extract data from param dict
    w = param['w_fun'](t)
    rpm2RadSec = 2.0*(np.pi)/60.0
    gen_speed_scaling = param['gen_speed_scaling']
    #x[DFSM.gen_speed_ind] = x[DFSM.gen_speed_ind]*gen_speed_scaling
    

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
    turbine_state['gen_speed'] = x[DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling
    turbine_state['gen_eff'] = param['VS_GenEff']/100
    turbine_state['rot_speed'] = x[DFSM.gen_speed_ind]*rpm2RadSec*gen_speed_scaling/param['WE_GearboxRatio']
    turbine_state['Yaw_fromNorth'] = 0
    turbine_state['Y_MeasErr'] = 0
    
    if not(DFSM.FA_Acc_ind == None):
        turbine_state['FA_Acc'] = x[DFSM.FA_Acc_ind]
        
    if not(DFSM.NacIMU_FA_Acc_ind == None):
        turbine_state['NacIMU_FA_Acc'] = x[DFSM.NacIMU_FA_Acc_ind]
    
    # call ROSCO to get control values
    gen_torque, bld_pitch, nac_yawrate = param['controller_interface'].call_controller(turbine_state)
    
    gen_torque = gen_torque/1000
    bld_pitch = np.rad2deg(bld_pitch)
    
    if 'wave_fun' in param:
        wv = param['wave_fun'](t)
        u = np.array([w,gen_torque,bld_pitch,wv])
    else:
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
    region = 'TR'
    datapath = this_dir + os.sep + 'outputs' + os.sep + 'MHK_'+region+'_10' #+ os.sep + 'openfast_runs/rank_0'
    
    # get the entire path
    outfiles = [os.path.join(datapath,f) for f in os.listdir(datapath) if valid_extension(f)]
    outfiles = sorted(outfiles)
    
    # required states
    reqd_states = ['PtfmPitch','GenSpeed','YawBrTAxp']
    
    state_props = {'units' : ['[deg]','[rpm]','[m/s2]'],
    'key_freq_name' : [['ptfm'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095],[0.095,0.39],[0.095,0.39]]}
    
    reqd_controls = ['RtVAvgxh','GenTq','BldPitch1','Wave1Elev']
    control_units = ['[m/s]','[kNm]','[deg]','[m]']
    
    
    reqd_outputs = ['TwrBsFxt', 'GenPwr'] #['YawBrTAxp', 'NcIMURAys', 'GenPwr'] #,'TwrBsMyt'
    
    output_props = {'units' : ['[kN]','[kNm]','[kW]'],
    'key_freq_name' : [['ptfm','2P'],['ptfm','2P'],['ptfm','2P']],
    'key_freq_val' : [[0.095,0.39],[0.095,0.39],[0.095,0.39]]}
    
    # scaling parameters
    scale_args = {'state_scaling_factor': np.array([1,100,1]),
                  'control_scaling_factor': np.array([1,1,1,1]),
                  'output_scaling_factor': np.array([1,1])
                  }
    
    # filter parameters
    filter_args = {'states_filter_flag': [False,False,False],
                   'states_filter_type': [[],[],[]],
                   'states_filter_tf': [[],[0.5],[]],
                   'controls_filter_flag': [False,False,False],
                   'controls_filter_tf': [0,0,0],
                   'outputs_filter_flag': []
                   }
    
    file_name = this_dir + os.sep + 'linear-models.mat'
    
    # instantiate class
    sim_detail = SimulationDetails(outfiles, reqd_states,reqd_controls,reqd_outputs,scale_args,filter_args,tmin=00
                                   ,add_dx2 = True,linear_model_file = file_name,region = region)
    
    
    
    save_path = 'plots_ROSCO'
    # load and process data
    sim_detail.load_openfast_sim()
    
    # extract data
    FAST_sim = sim_detail.FAST_sim
    
    # plot data
    #plot_inputs(sim_detail,4,'separate')
    
    # split of training-testing data
    n_samples = 100
    test_inds = [0,1]
    
    # construct surrogate model
    dfsm_model = DFSM(sim_detail,n_samples = n_samples,L_type = 'LTI',N_type = None, train_split = 0.4)
    dfsm_model.construct_surrogate()
    dfsm_model.simulation_time = []
        
    for ind in test_inds:
            
        test_data = dfsm_model.test_data[ind]
        
        bp_init = test_data['controls'][0,2]
        bp_of = test_data['controls'][:,2]
        gt_of = test_data['controls'][:,1]
        bp_mean = np.mean(test_data['controls'][:,2])
        
        nt = 1000
        wind_speed = test_data['controls'][:,0] # np.ones((nt,))*2.5 #test_data['controls'][:,0] #
        nt = len(wind_speed)
        t0 = 0;tf = 700
        dt = 0.0; t1 = t0 + dt
        time_of = np.linspace(t0,tf,nt)
        
        w_pp = CubicSpline(time_of, wind_speed)
        w_fun = lambda t: w_pp(t)
        w0 = w_fun(0)
        bp0 = bp_mean
        gt0 = 8.3033588
        
        wave_elev = test_data['controls'][:,3]
        wave_pp = CubicSpline(time_of,wave_elev)
        wave_fun = lambda t:wave_pp(t)
        
        tspan = [t1,tf]
        
        x0 = test_data['states'][0,:]
        states_ = test_data['states']
        
        
        # Load controller library
        #controller_interface = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename,sim_name='sim_test')
        
        if True: 
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
                     'gen_speed_scaling':100,
                     'controller_interface':ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename),
                     'wave_fun':wave_fun}
            
            # solver method and options
            solve_options = {'method':'RK45','rtol':1e-7,'atol':1e-7}
            
            t1 = timer.time()
            sol =  solve_ivp(run_sim_ROSCO,tspan,x0,method=solve_options['method'],args = (dfsm_model,param),rtol = solve_options['rtol'],atol = solve_options['atol'])
            t2 = timer.time()
            dfsm_model.simulation_time.append(t2-t1)
            
            time = sol.t 
            states = sol.y
            states = states.T
            
            # kill controller
            param['controller_interface'].kill_discon()
            
            tspan = [0,tf]
            time_sim = param['time']
            blade_pitch = np.array(param['blade_pitch'])
            gen_torque = np.array(param['gen_torque'])
            
            # interpolate controls
            blade_pitch = interp1d(time_sim,blade_pitch)(time)
            gen_torque = interp1d(time_sim,gen_torque)(time)
            
            # interpolate time
            wind_of = w_fun(time)
            blade_pitch_of = interp1d(time_of,bp_of)(time)
            gen_torque_of = interp1d(time_of,gt_of)(time)
            
            wave_elev = wave_fun(time)
            
            controls_of = np.array([wind_of,gen_torque_of,blade_pitch_of,wave_elev]).T
            controls_dfsm = np.array([wind_of,gen_torque,blade_pitch,wave_elev]).T
            
            inputs_dfsm = np.hstack([controls_dfsm,states])
            states_of = CubicSpline(time_of,states_)(time)
            
            X_list = [{'time':time,'names':test_data['state_names'],'n':test_data['n_states'],'OpenFAST':states_of,
                      'DFSM':states,'units':['[rpm]','[rpm/s]'],'key_freq_name':[['2P']],'key_freq_val':[[0.39]]}]
            
            U_list = [{'time': time, 'names': test_data['control_names'],'n': test_data['n_controls']
                      ,'OpenFAST':controls_of,'DFSM':controls_dfsm,'units': ['[m/s]','[kNm]','[deg]']}]
            
            fun_type = 'outputs'
            
            if dfsm_model.n_outputs > 0:
                outputs_of = test_data['outputs']
                
                outputs_of = CubicSpline(time_of,outputs_of)(time)
                
                
                outputs_dfsm = evaluate_dfsm(dfsm_model,inputs_dfsm,fun_type)
                    
                Y_list = [{'time':time,'names':test_data['output_names'],'n':test_data['n_outputs'],
                          'OpenFAST':outputs_of,'DFSM':outputs_dfsm,'units':['[kW]','[kN]'],
                          'key_freq_name':[['2P'],['2P']],'key_freq_val':[[0.39],[0.39]]}]
                
            #plot_dfsm_results(U_list,X_list,[],Y_list,control_flag= False,simulation_flag = True,outputs_flag = True,save_flag = True,save_path = save_path)    
            
            #dfsm_time = calculate_time(dfsm_model)
            
            # fig.subplots_adjust(hspace = 0.85)
            save_flag = False;plot_path = 'plots_ROSCO'
            
            fig,ax = plt.subplots(1)
            ax.plot(time,blade_pitch,label = 'DFSM')
            ax.plot(time_of,bp_of,label = 'OpenFAST')
            ax.set_title('BldPitch [deg]')
            ax.set_xlim(tspan)
            #ax.set_ylim([8,10.5])
            ax.legend(ncol = 2)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'BldPitch' + '_comp.svg')
            
            fig,ax = plt.subplots(1)
            ax.plot(time,states[:,1],label = 'DFSM')
            ax.plot(time_of,states_[:,1],label = 'OpenFAST')
            ax.set_title('GenSpeed [rpm]')
            ax.set_xlim(tspan)
            #ax.set_ylim([580,630])
            ax.legend(ncol = 2)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'GenSpeed' + '_comp.svg')
                
            fig,ax = plt.subplots(1)
            ax.plot(time,states[:,0],label = 'DFSM')
            ax.plot(time_of,states_[:,0],label = 'OpenFAST')
            
            ax.set_title('PtfmPitch [deg]')
            ax.set_xlim(tspan)
            #ax.set_ylim([580,630])
            ax.legend(ncol = 2)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'GenSpeed' + '_comp.svg')
                
                
            fig,ax = plt.subplots(1)
            ax.plot(time,states[:,2],label = 'DFSM')
            ax.plot(time_of,states_[:,2],label = 'OpenFAST')
            
            ax.set_title('YawBrTAxp [m/s2]')
            ax.set_xlim(tspan)
            #ax.set_ylim([580,630])
            ax.legend(ncol = 2)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'GenSpeed' + '_comp.svg')
            
            if dfsm_model.n_outputs > 0:
                fig,ax = plt.subplots(1)
                ax.plot(time,outputs_dfsm[:,1],label = 'DFSM')
                ax.plot(time,outputs_of[:,1],label = 'OpenFAST')
                
                ax.set_title('GenPwr [kW]')
                ax.set_xlim(tspan)
                #ax.set_ylim([480,530])
                ax.legend(ncol = 2)
                ax.set_xlabel('Time [s]')
                
                if save_flag:
                    if not os.path.exists(plot_path):
                            os.makedirs(plot_path)
                        
                    fig.savefig(plot_path +os.sep+ 'GenPwr' + '_comp.svg')
                
                
            fig,ax = plt.subplots(1)
            ax.plot(time,gen_torque,label = 'DFSM')
            ax.plot(time_of,gt_of,label = 'OpenFAST')
            
            ax.set_title('GenTq [kNm]')
            ax.set_xlim(tspan)
            #ax.set_ylim([8,8.5])
            ax.legend(ncol = 2)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'GenTq' + '_comp.svg')
            
            fig,ax = plt.subplots(1)
            ax.plot(time_of,w_fun(time_of))
            #ax.plot(time,gen_torque,label = 'DFSM')
            ax.set_title('Current Speed [m/s]')
            ax.set_xlim(tspan)
            ax.set_xlabel('Time [s]')
            
            if save_flag:
                if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    
                fig.savefig(plot_path +os.sep+ 'FlowSpeed' + '_comp.svg')
            
        
        
        
        
        
        
        
        
        


