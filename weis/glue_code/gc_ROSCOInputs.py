def assign_ROSCO_values(wt_opt, modeling_options, control):
    # ROSCO tuning parameters
    wt_opt['tune_rosco_ivc.PC_omega']      = control['pitch']['PC_omega']
    wt_opt['tune_rosco_ivc.PC_zeta']       = control['pitch']['PC_zeta']
    wt_opt['tune_rosco_ivc.VS_omega']      = control['torque']['VS_omega']
    wt_opt['tune_rosco_ivc.VS_zeta']       = control['torque']['VS_zeta']
    if modeling_options['servose']['Flp_Mode'] > 0:
        wt_opt['tune_rosco_ivc.Flp_omega']      = control['Flp_omega']
        wt_opt['tune_rosco_ivc.Flp_zeta']       = control['Flp_zeta']
    # # other optional parameters
    wt_opt['tune_rosco_ivc.max_pitch']     = control['pitch']['max_pitch']
    wt_opt['tune_rosco_ivc.min_pitch']     = control['pitch']['min_pitch']
    wt_opt['tune_rosco_ivc.vs_minspd']     = control['torque']['VS_minspd']
    wt_opt['tune_rosco_ivc.ss_vsgain']     = control['setpoint_smooth']['ss_vsgain']
    wt_opt['tune_rosco_ivc.ss_pcgain']     = control['setpoint_smooth']['ss_pcgain']
    wt_opt['tune_rosco_ivc.ps_percent']    = control['pitch']['ps_percent']
    # Check for proper Flp_Mode, print warning
    if modeling_options['airfoils']['n_tab'] > 1 and modeling_options['servose']['Flp_Mode'] == 0:
            print('WARNING: servose.Flp_Mode should be >= 1 for aerodynamic control.')
    if modeling_options['airfoils']['n_tab'] == 1 and modeling_options['servose']['Flp_Mode'] > 0:
            print('WARNING: servose.Flp_Mode should be = 0 for no aerodynamic control.')

    return wt_opt
