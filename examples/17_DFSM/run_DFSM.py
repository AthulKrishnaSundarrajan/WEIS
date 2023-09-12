import os
import numpy as np
from dfsm.simulation_details import SimulationDetails
from dfsm.dfsm_plotting_scripts import plot_inputs,plot_dfsm_results
from dfsm.dfsm_utilities import valid_extension
from dfsm.test_dfsm import test_dfsm
from dfsm.construct_dfsm import DFSM
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # get path to current directory
    mydir = os.path.dirname(os.path.realpath(__file__))
    
    # datapath
    datapath = mydir + os.sep + 'outputs' + os.sep + 'DFSM_MHK_CT4' + os.sep + 'openfast_runs/rank_0'
    
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
        
        inputs_sampled = dfsm_model.inputs_sampled
        dx_sampled = dfsm_model.dx_sampled
        model_inputs = dfsm_model.model_inputs
        state_derivatives = dfsm_model.state_derivatives
        AB = dfsm_model.AB
        AB = AB.T
        A = AB[:,3:]
        
        # nonlin_deriv = dfsm_model.nonlin_deriv
        
        
        test_data = dfsm_model.test_data
        
        test_ind = [0]
        simulation_flag = True 
        outputs_flag = (len(reqd_outputs) > 0)
        plot_flag = True
        
        dfsm,U_list,X_list,dx_list,Y_list = test_dfsm(dfsm_model,test_data,test_ind,simulation_flag,plot_flag)
        
        plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag)
        
        # dfsm,U_list,X_list,dx_list,Y_list = test_dfsm(dfsm_model,FAST_sim,0,simulation_flag,plot_flag)
        
        # plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag)
        
    # fig,ax = plt.subplots(1,1)
    
    # ax.plot(state_derivatives[:,0],state_derivatives[:,1],'.',color = 'b',markersize = 0.5,label = 'data')
    # ax.plot(dx_sampled[:,0],dx_sampled[:,1],'.',color = 'r',markersize = 3,label = 'samples')
    # ax.legend()
    # ax.set_xlabel('dx1')
    # ax.set_ylabel('dx3')
    
    # fig,ax = plt.subplots(1,1)
    
    # ax.plot(state_derivatives[:,1],state_derivatives[:,3],'.',color = 'b',markersize = 0.5,label = 'data')
    # ax.plot(dx_sampled[:,1],dx_sampled[:,3],'.',color = 'r',markersize = 3,label = 'samples')
    # ax.legend()
    # ax.set_xlabel('dx2')
    # ax.set_ylabel('dx4')
    
    
    
    
    
    

