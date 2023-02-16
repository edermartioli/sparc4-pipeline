# -*- coding: utf-8 -*-
"""
Created on Thu Jun 2 2022
@author: Eder Martioli
Laboratório Nacional de Astrofísica - LNA
"""

import numpy as np
import batman

def batman_model(time, per, t0, a, inc, rp, u0, u1=0., ecc=0., w=90.) :
    
    """
        Function for computing transit models for the set of 8 free paramters
        x - time array
        """
    params = batman.TransitParams()
    
    params.per = per
    params.t0 = t0
    params.inc = inc
    params.a = a
    params.ecc = ecc
    params.w = w
    params.rp = rp
    params.u = [u0,u1]
    params.limb_dark = "quadratic"       #limb darkening model
    
    m = batman.TransitModel(params, time)    #initializes model
    
    flux_m = m.light_curve(params)          #calculates light curve
    
    return np.array(flux_m)
