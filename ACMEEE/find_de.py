import iDEA as idea
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime
import ACMEEE.save_outputs

#function to determine if a density contains two electons, each in their first state.
#INPUTS:    system of state being checked
#           state being checked
#           sensitivity of excitation finder
#RETURNS:   True or False if the inputs indicate a double excitation          
def isdoubleexcitation (system,state,sensitivity):
    xgrid = system.x
    #compute density
    density = idea.observables.density(system,state=state)
    #find peaks in the density which are more than half the height of the maximum.
    density_peaks = sp.signal.find_peaks(density, height = 0.01)

    #if there are 4 peaks then continue, otherwise return false
    if len(density_peaks[0]) != 4:
        return False
    else:
        #if the peaks are grouped into sets of 2 then return true, otherwise return false
        if abs(xgrid[density_peaks[0][1]]-xgrid[density_peaks[0][0]]) < sensitivity and abs(xgrid[density_peaks[0][3]]-xgrid[density_peaks[0][2]]) < sensitivity:
            return True
        else:
            return False

#iterates through excitations to find the doubly excited state.
#INPUTS:    system being chcked
#           sensitivity of excitation finder
#           limit: excitation where excitation finder will error.
#RETURNS:   excitation number of doubly excited state for this system.
def finddoubleexcitation(system,sensitivity,limit,startpoint,outputpath):
    #solve
    print(f"{datetime.datetime.now()}: Finding double excitation",flush=True)
    found = 0
    i = startpoint
    while found == 0:
        teststate = idea.methods.interacting.solve(system, k=i,stopprint=True)
        save_density(teststate,i,system,outputpath)
        ifinnerprod = isdoubleexcitation(system,teststate,sensitivity)
        if i > limit:
            found = 2
        elif ifinnerprod == True:
            found = 1
        elif ifinnerprod == False:
            print(f"{datetime.datetime.now()}: Searched k={i}, continuing...",flush=True)
            i = i + 1
        
    if found == 1:  
        print(f"{datetime.datetime.now()}: Double excitation found in the {i}th excited state.",flush=True)
        return i
    elif found == 2:
        raise Exception(f"{datetime.datetime.now()}: No double excitations found up to the {limit}th excited state",flush=True)
