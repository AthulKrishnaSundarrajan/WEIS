
import weis.inputs as sch
import os
from weis.aeroelasticse.turbsim_util    import generate_wind_files
from weis.aeroelasticse.turbsim_file    import TurbSimFile
from weis.dlc_driver.dlc_generator      import DLCGenerator
from weis.control.LinearModel           import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse.CaseGen_General import case_naming
from weis.control.dtqp_wrapper          import dtqp_wrapper
import pickle
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
import matplotlib.pyplot as plt
from dtqpy.src.DTQPy_oloc import DTQPy_oloc
import numpy as np
from os.path import exists

class dict2class(object):
    
    def __init__(self,my_dict):
        
        for key in my_dict:
            setattr(self,key,my_dict[key])
            
        self.A_ops = self.A
        self.B_ops = self.B
        self.C_ops = self.C
        self.D_ops = self.D
        
        if isinstance(self.u_h,list):
            self.u_h = np.array(self.u_h)
            
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

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if __name__ == '__main__':

    # read WEIS options:
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    modeling_options = sch.load_modeling_yaml(fname_modeling_options)

    fname_wt_input   = mydir + os.sep + "IEA-15-floating.yaml"
    wt_init          = sch.load_geometry_yaml(fname_wt_input)
    
    Turbine_class = wt_init["assembly"]["turbine_class"]
    
    fname_analysis_options      = mydir + os.sep + "analysis_options.yaml"
    analysis_options            = sch.load_analysis_yaml(fname_analysis_options)

    # Wind turbine inputs 
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

    level2_disturbance = []
    fig1,ax1 = plt.subplots(1,1)
    ax1.set_ylabel('Wind Speed [m/s]',fontsize = 16)
    ax1.set_xlabel('Time [s]',fontsize = 16)
    ax1.set_xlim([0,800])
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    labels = ['DLC 1','DLC 2','DLC 3','DLC 4','DLC 5','DLC 6']
    colors = ['tab:blue','tab:purple','tab:orange','tab:brown','tab:cyan','tab:green']

    for i_case in range(dlc_generator.n_cases):
        dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
            dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, i_case)

        # Compute rotor average wind speed as level2_disturbances
        ts_file     = TurbSimFile(WindFile_name[i_case])
        ts_file.compute_rot_avg(rotorD/2)
        u_h         = ts_file['rot_avg'][0,:]
        
        
        off = max(u_h) - 25
        ind = u_h > 25;
        
        # remove any windspeeds > 25 m/s
        if ind.any():
            u_h[ind] = u_h[ind] - off
        
        print(np.max(u_h));print(np.min(u_h));
        tt = ts_file['t']
        level2_disturbance.append({'Time':tt, 'Wind': u_h})
        ax1.plot(tt,u_h,linewidth = 2,markersize = 10,label = labels[i_case],color = colors[i_case])
    
    #fig1.legend(bbox_to_anchor=(1,1), loc='upper center',ncol = 1,fontsize = 12) 
    fig1.savefig('wind.svg',format = 'svg',dpi = 1200)
    # breakpoint()
    # # Linear Model
    # analysis_dir = mydir + os.sep + "outputs" + os.sep + "q8_review" 
    # pkl_file = analysis_dir + os.sep + "ABCD_matrices.pkl" 
    # #pkl_file = mydir + os.sep + 'ABCD_matrices.pkl'
    
    # with open(pkl_file,"rb") as handle:
    #     ABCD_list = pickle.load(handle)

    
    # fst_vt = {}
    # fst_vt['DISCON_in'] = {}
    # fst_vt['DISCON_in']['PC_RefSpd'] = 0.7853192931562493

    # la = LoadsAnalysis(
    #         outputs=[],
    #     )

    # magnitude_channels = {
    #     "RootMc1": ["RootMxc1", "RootMyc1", "RootMzc1"],
    #     "RootMc2": ["RootMxc2", "RootMyc2", "RootMzc2"],
    #     "RootMc3": ["RootMxc3", "RootMyc3", "RootMzc3"],
    #     }

    # run_directory = modeling_options['General']['openfast_configuration']['OF_run_dir']
    
    #     # penalty = -np.logspace(0,9,20)
    #     # penalty[0] = 0
    #     # n_p = len(penalty)
    #     # pwr = np.zeros((n_p,))
    #     # pitch_std = np.zeros((n_p,))
        
    # n = -2

    # ABCD = ABCD_list[n]
    
    # LinearTurbine = dict2class(ABCD)
        
    # dtqp_constraints = {}
    
    # # Control constraints that are supported
    # control_const = analysis_options['constraints']['control']

    # # Rotor overspeed
    # if control_const['rotor_overspeed']['flag']:
    #     desc = 'ED First time derivative of Variable speed generator DOF (internal DOF index = DOF_GeAz), rad/s'
    #     if desc in LinearTurbine.DescStates:
    #         dtqp_constraints[desc] = [0,(1 + control_const['rotor_overspeed']['max']) * fst_vt['DISCON_in']['PC_RefSpd'] ]
    #     else:
    #         raise Exception('rotor_overspeed constraint is set, but ED GenSpeed is not a state in the LinearModel')
    # else:
    #     desc = 'ED First time derivative of Variable speed generator DOF (internal DOF index = DOF_GeAz), rad/s'
    #     dtqp_constraints[desc] = [0,(1 + 0.0) * fst_vt['DISCON_in']['PC_RefSpd'] ]
        
    # if control_const['Max_PtfmPitch']['flag']:
    #     desc = 'ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad'
    #     if desc in LinearTurbine.DescStates:
    #         dtqp_constraints[desc] = [-np.inf,control_const['Max_PtfmPitch']['max'] * np.deg2rad(1)]
    #     else:
    #         raise Exception('Max_PtfmPitch constraint is set, but ED PtfmPitch is not a state in the LinearModel')
    # else:
    #     desc = 'ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad'
    #     dtqp_constraints[desc] = [-np.inf,6* np.deg2rad(1)]
        
    # plot = True
    
    # dtqp_options = modeling_options['Level2']['DTQP']
    
         
    
    # T,U,X,Y = DTQPy_oloc(LinearTurbine, level2_disturbance[2], dtqp_constraints, dtqp_options)
    
    # iPtfmPitch = LinearTurbine.DescStates.index('ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad')
    # iGenPwr = LinearTurbine.DescOutput.index('SrvD GenPwr, (kW)')
    
    # Time = T.squeeze()
    # PtfmPitch = np.squeeze(np.rad2deg(X[:,iPtfmPitch]))
    # GenSpeed = np.squeeze(X[:,4])
    # GenTq = np.squeeze(U[:,1])
    # GenPwr = np.squeeze(Y[:,iGenPwr])

    # pwr = np.mean(GenPwr)
    
    # #df = summary_stats.loc[:, ("PtfmPitch", "std")].to_frame()
    # pitch_std = np.std(PtfmPitch)
    
    # #AEP[n] = Calc_AEP(summary_stats,dlc_generator,Turbine_class)
    
    # print('here')
    # breakpoint()
    # pklname = "q8_review.pkl"
    
    # if not exists(pklname):
    #     Results = []
        
    #     with open(pklname,'wb') as handle:
    #         pickle.dump(Results,handle)
        
        
    # with open(pklname,'rb') as handle:
    #     Results = pickle.load(handle)
    
    # Res_dict = {'id':'rated',"Time":Time,"States":X,"Controls":U,"Outputs":Y,"StateNames":LinearTurbine.DescStates,"ControlNames":LinearTurbine.DescCntrlInpt,"Outputnames":LinearTurbine.DescOutput}
    
    # Results.append(Res_dict)

    # with open(pklname,'wb') as handle:
    #     pickle.dump(Results,handle)
        
    
    
    # # penalty[0] = -1
    # # pen_log = np.log10(-penalty)
    
    
    
    # # plt.plot(-penalty, pwr,'*-')
    # # plt.plot(-penalty,pitch_std,'s-')
    # # plt.xlabel("PtfmPitch penalty")
    # # plt.gca().set_xlim([1,10**9])
    # # plt.ylabel("Power [kW]")
    # # plt.gca().set_xscale("log")
    # # plt.grid()
    # #plt.gca().invert_xaxis()
    
    # # fig, ax = plt.subplots(1,1)
    # # ax.plot(n_dlc,AEP,'*-')
        
    # # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')   
    # # ax.set_xlabel('n_DLCs')
    # # ax.set_ylabel('AEP [kW]')

    
