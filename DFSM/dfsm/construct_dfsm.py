import os
import numpy as np

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from numpy.linalg import lstsq,qr,inv,norm
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBFsk
from sklearn.gaussian_process.kernels import ExpSineSquared
import time as timer
from sklearn.cluster import KMeans

from dfsm.dfsm_sample_data import sample_data
from sklearn.neural_network import MLPRegressor


class DFSM:
    
    def __init__(self,SimulationDetails,L_type = 'LTI',N_type = 'GPR',n_samples = 300,sampling_method = 'KM',train_split = 0.8):
        
        self.L_type = L_type
        self.N_type = N_type 
        self.n_samples = n_samples
        self.sampling_method = sampling_method 
        self.train_split = train_split
        
        FAST_sim = SimulationDetails.FAST_sim
        
        self.n_model_inputs = SimulationDetails.n_model_inputs
        self.n_deriv = SimulationDetails.n_deriv
        self.n_outputs = SimulationDetails.n_outputs
        
        n_sim = SimulationDetails.n_sim
        
        train_index = int(np.floor(train_split*n_sim))
        self.train_data = FAST_sim[0:train_index]
        
        self.test_data = FAST_sim[train_index:]
        
    def construct_nonlinear(self,inputs,outputs,N_type,error_ind,n_inputs,n_outputs,ftype = 'deriv'):
        
        if N_type == 'GPR':
            
            # train a gaussian process model
            kernel = 1*RBFsk(length_scale = [1]*n_inputs,length_scale_bounds=(1e-5, 1e5))
            sm = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 5,random_state = 34534)
            sm.fit(inputs,outputs[:,error_ind])
            
        elif N_type == 'NN':
            
            # train a neural network to predict the errors
            sm = MLPRegressor(hidden_layer_sizes = (50,10),max_iter = 300,activation = 'tanh',solver = 'adam',verbose = True,tol = 1e-5)
            sm.fit(inputs,outputs[:,error_ind])
            
        if ftype == 'deriv':
            
            self.nonlin_deriv = sm
            
        elif ftype == 'outputs':
            
            self.nonlin_outputs = sm
            
       
    def construct_surrogate(self):
        
        # extract samples
        t1 = timer.time()
        inputs_sampled,dx_sampled,outputs_sampled,model_inputs,state_derivatives,outputs = sample_data(self.train_data,self.sampling_method,self.n_samples,grouping = 'together')
        t2 = timer.time()
        
        self.sampling_time = t2 - t1
        
        # depending on the type of L and N construct the surrogate model
        if self.L_type == None:
            
            # set the AB and CD matrices as empty
            self.AB = []
            self.CD = []
            self.lin_construct = 0
            
            self.error_ind_deriv = np.full(self.n_deriv,True)
            
            if self.n_outputs > 0:
                self.error_ind_outputs = np.full(self.n_outputs,True)
            
            t1 = timer.time()
            self.construct_nonlinear(inputs_sampled,dx_sampled,self.N_type,self.error_ind_deriv,self.n_model_inputs,self.n_deriv,'deriv')
            
            if self.n_outputs > 0:
                self.construct_nonlinear(inputs_sampled,outputs_sampled,self.N_type,self.error_ind_outputs,self.n_model_inputs,self.n_outputs,'outputs')
            
            t2 = timer.time()
            
            self.nonlin_construct_time = t2-t1
            self.inputs_sampled = inputs_sampled
            self.dx_sampled = dx_sampled
            
        else:
            
            if self.L_type == 'LTI':
                    
                    
                # start timer
                t1 = timer.time()
                
                AB = lstsq(model_inputs,state_derivatives,rcond = -1)
                self.AB = AB[0]

                if self.n_outputs > 0:
                    CD = lstsq(model_inputs,outputs,rcond = -1)
                    
                    self.CD = CD[0]
                    
                else:
                    
                    self.CD = []
                
                # end timer
                t2 = timer.time()
                
                # training time
                self.linear_construct_time = t2-t1
                self.inputs_sampled = inputs_sampled
                self.dx_sampled = dx_sampled
                self.model_inputs = model_inputs
                self.state_derivatives = state_derivatives
                
                
                
                dx_error = dx_sampled - np.dot(inputs_sampled,self.AB)
                
                error_mean = np.mean(dx_error,0)

                error_ind_deriv = np.array((np.abs(error_mean) > 1e-5))
                
                self.error_ind_deriv = error_ind_deriv
                
                if self.n_outputs > 0:
                    outputs_error = outputs_sampled - np.dot(inputs_sampled,self.CD)
                    
                    
                    self.error_ind_outputs = (np.abs(np.mean(outputs_error,0)) > 1e-5)
                                                    
            
            if self.N_type == None:
                
                self.nonlin_deriv = None
                self.nonlin_outputs = None
                
            else:
                
                t1 = timer.time()

                self.construct_nonlinear(inputs_sampled,dx_error,self.N_type,self.error_ind_deriv,self.n_model_inputs,self.n_deriv,'deriv')
                
                if self.n_outputs > 0:
                    self.construct_nonlinear(inputs_sampled,outputs_error,self.N_type,self.error_ind_outputs,self.n_model_inputs,self.n_outputs,'outputs')
                
                t2 = timer.time()
                
                self.nonlin_construct_time = t2-t1
                
        
