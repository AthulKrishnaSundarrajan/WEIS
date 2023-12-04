import numpy as np


def evaluate_dfsm(DFSM,inputs,fun_type = 'deriv'):
    
    # get number of points
    n_points = len(np.shape(inputs))
    
    if n_points == 1:
        inputs = inputs.reshape(1,-1)
        nt = 1
    else:
        nt = np.shape(inputs)[0]
    
    # based on the function type extract the info
    if fun_type == 'deriv':
        
        # state derivative function
        lin = DFSM.AB
        nonlin = DFSM.nonlin_deriv
        no = DFSM.n_deriv
        error_ind = DFSM.error_ind_deriv
        
    elif fun_type == 'outputs':
        
        # output function
        lin = DFSM.CD
        nonlin = DFSM.nonlin_outputs
        no = DFSM.n_outputs
        error_ind = DFSM.error_ind_outputs
        
    if DFSM.L_type == None:
        
        dx_lin = np.zeros((nt,no))
        
    else:
        
        # if LTI model
        if DFSM.L_type == 'LTI':
            dx_lin = np.dot(inputs,lin)
            
    dx_nonlin = np.zeros((nt,no))
    
    if not(DFSM.N_type == None):
        
        # predict 
        nonlin_prediction = nonlin.predict(inputs)
        
        
        if len(np.shape(nonlin_prediction)) == 1:
            nonlin_prediction = nonlin_prediction.reshape(-1,1)
        
        dx_nonlin[:,error_ind] = dx_nonlin[:,error_ind] + nonlin_prediction
        #print(nonlin_prediction)
        
    dx = dx_lin + dx_nonlin
    
    if n_points == 1:
        dx = np.squeeze(dx)
        
    return dx
            
        
        
    
            
    
    
    

