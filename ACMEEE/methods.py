import iDEA as idea
import numpy as np
import gc
import datetime
import sys
import os

import ACMEEE as acme

def assemble(
        xgrid: np.array,
        potential_name: str,
        initial_distance: float,
        distance_step: float,
        electronconfig: str,
        outputpath: str,
        debugging: bool = False,
        find_startpoint: float = None,
        trackedstate: int = None,
        sensitivity: float = 5,
        limit: int = 50,
        abovetracked: int = 5,
        innerprod_tolerence: float = 0.1,
        maxdivisions: int = 30,
        orbital_max_excitation: int = 20) -> tuple[int,int,int,int]:
    """
    Performs the adiabatic movement

    Parameters:
    xgrid: np.array, grid of x values in 1D space
    potential_name: str, the name of the external potential that is to be used
    intial_distance: float, initial distance of potential wells from x=0
    distance step: float, the largest distance step during the adiabatic movement
    electronconfig: str, the spin configuration of electrons e.g 'uu'
    outputpath: str, filepath for output folders
    debugging: bool, True will give extra outputs and prints
    find_startpoint: int, the excitation number that the double excitation finding algorithm will begin at. If None, user selected trackedstate parameter will be used
    trackedstate: int, the excitation number of the state that is to be followed throughout the movement. If None, finding algorithm should be used
    sensititivty: float, the sensitivity parameter of the double excitation finding algorithm
    limit: int, maximum excitation number that the double excitation finder will check
    abovetracked: int, number of excitations above the tracked state that will be generated in order to track it
    innerprod_tolerence: float, inner product tolerance used for accepting states in the adiabatic movement. e.g. tol=0.1 will accept states that has an inner product >0.9
    maxdivisions: int, maximum number of step divisions before program gives up
    orbital_max_excitations: highest orbital excitation used in the CI expansions

    Returns:
    trackedstate: int, excitation number of the tracked state at end of movemement
    num_accepted: int, number of steps that were accepted during the movement
    num_rejected: int, number of steps that were rejected during the movement
    num_total: int, total number of steps during the movement
    """

    outputpath = acme.utilities.clearoutputs(outputpath)

    #initialise outputs
    num_total = 1
    num_rejected = 0
    num_accepted = 0

    #get initial system for finding double excitation
    v_int = idea.interactions.softened_interaction(xgrid)
    initial_potential = acme.potential.potential(xgrid,initial_distance,potential_name)
    initial_system = idea.system.System(xgrid,initial_potential,v_int,electrons=electronconfig)

    #if no initial excitation specified, find it for the initial distance.
    if trackedstate == -1:
        trackedstate = acme.find_de.finddoubleexcitation(initial_system,sensitivity,limit,find_startpoint,outputpath)

    maxexcitation_gen = trackedstate + abovetracked
    
    #Generate state at first distance
    print(f"{datetime.datetime.now()}: Generating initial state at distance {initial_distance}, DE={trackedstate}",flush=True)
    distance_old = initial_distance
    distance_new  = initial_distance - distance_step
    system_old = idea.system.System(xgrid,acme.potential.potential(xgrid,distance_old,potential_name),v_int,electrons=electronconfig)
    with acme.utilities.HiddenPrints():
        state_old = idea.methods.interacting.solve(system_old, k=-1, level=idea.methods.interacting._estimate_level(system_old,maxexcitation_gen))
    state_id = 1
    acme.save_outputs.save_observables(state_old,system_old,trackedstate,distance_old,distance_old,outputpath,state_id,0)
    state_temp = state_old
    state_temp.full = state_temp.fulls[:,:,:,:,trackedstate]
    hf = acme.orbitals.orbitals(state_temp,system_old,state_id,distance_old,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
    print(f"     Hartree-Fock completeness: {hf*100}%")
    sys.stdout.flush()    


    n = 1

    #Begin moving closer
    print(f"{datetime.datetime.now()}: Starting Movement",flush=True)
    sys.stdout.flush()
    while distance_old != 0:
        
        #generate new state
        num_total = num_total + 1
        print(f"{datetime.datetime.now()}: Generating state at distance {distance_new}",flush=True)
        sys.stdout.flush()
        maxexcitation_gen = trackedstate + abovetracked
        with acme.utilities.HiddenPrints():
            system_new = idea.system.System(xgrid,acme.potential.potential(xgrid,distance_new,potential_name),v_int,electrons=electronconfig)
        state_new = idea.methods.interacting.solve(system_new, k=-1, level=idea.methods.interacting._estimate_level(system_old,maxexcitation_gen))

        #compute inner product grid for these states
        innergrid_old_new = acme.innerprod.innerprodgrid(state_old,state_new,system_old,system_new,maxexcitation_gen)
        if debugging == True:
            acme.save_outputs.save_innerprodgrid(innergrid_old_new,distance_old,distance_new,outputpath)

        #get value and index of highest inner product for old state double excitation
        de_innerprod_value = np.max(innergrid_old_new[trackedstate])
        de_innerprod_index = np.argmax(innergrid_old_new[trackedstate])

        #check if there is an inner product above 1-tolerance
        if (de_innerprod_value > (1-innerprod_tolerence)):
            #state found
            num_accepted = num_accepted + 1
            #is the current distance a multiple of the step distance?
            if (round(distance_new/distance_step,2)).is_integer():
                print(f"{datetime.datetime.now()}: Double excitation state found at distance {distance_new}, Innerproduct {de_innerprod_value}, DE={de_innerprod_index}",flush=True)
                sys.stdout.flush()
                state_id = state_id + 1
                trackedstate = de_innerprod_index
                acme.save_outputs.save_observables(state_new,system_new,trackedstate,distance_new,distance_old,outputpath,state_id,innergrid_old_new)
                state_temp = state_new
                state_temp.full = state_temp.fulls[:,:,:,:,trackedstate]
                hf = acme.orbitals.orbitals(state_temp,system_new,state_id,distance_new,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
                print(f"     Hartree-Fock completeness: {hf*100}%")
                sys.stdout.flush()
                system_old = system_new
                state_old = state_new
                del state_new
                gc.collect()
                distance_old = distance_new
                distance_new = distance_new - distance_step
                n = 1
                
            else:
                print(f"{datetime.datetime.now()}: Double excitation state found at distance {distance_new}, Innerproduct {de_innerprod_value}, DE={de_innerprod_index}",flush=True)
                sys.stdout.flush()
                state_id = state_id + 1
                trackedstate = de_innerprod_index
                acme.save_outputs.save_observables(state_new,system_new,trackedstate,distance_new,distance_old,outputpath,state_id,innergrid_old_new)
                state_temp = state_new
                state_temp.full = state_temp.fulls[:,:,:,:,trackedstate]
                hf = acme.orbitals.orbitals(state_temp,system_new,state_id,distance_new,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
                print(f"     Hartree-Fock completeness: {hf*100}%")
                sys.stdout.flush()
                system_old = system_new
                state_old = state_new
                del state_new
                gc.collect()
                distance_old = distance_new
                distance_new = distance_new - (distance_step/(2**(n-1)))
                
                
        else:
            #state not found, check half distance
            num_rejected = num_rejected + 1
            print(f"{datetime.datetime.now()}: Double excitation state not found at distance {distance_new}, Innerproduct {de_innerprod_value}",flush=True)
            sys.stdout.flush()
            del state_new
            gc.collect()
            distance_new = distance_old-(distance_step/(2**n))
            if n >= maxdivisions:
                raise Exception("Max number of deivisons reached. Stopping")
            n = n + 1

    idea.state.save_many_body_state(state_old.fulls[:,:,:,:,trackedstate],f"{outputpath}/doublestate.state")
    idea.system.save_system(system_old,f"{outputpath}/doublestate.system")

    acme.save_outputs.gif_wavefunctions(outputpath)
    acme.save_outputs.gif_densities(outputpath)
    acme.save_outputs.gif_innerproducts(outputpath)
    acme.save_outputs.energy_graph(outputpath)

    return trackedstate, num_accepted, num_rejected, num_total
    
