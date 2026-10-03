import iDEA as idea
import numpy as np
import gc
from innerprodgrid import innerprodgrid
from find import finddoubleexcitation
from save_observables import save_observables
from potential import potential
from save_observables import save_innerprodgrid
from orbitals import orbitals
import datetime
import sys
import os




#Assemble a double excitation state
#========================================
#INPUTS:
# potential, xgrid
#========================================
# GENERAL PLAN:
#
# find initial excitation
# generate state at starting distance
# generate state at next distance
# does an inner product exist for the interested state?
# if yes continue, if no generate state halfway between this state and last state

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def assemble(xgrid,potential_name,debugging,find_startpoint,doubleexcitation,initial_distance,sensitivity,limit,abovedouble,innerprod_tolerence,distance_step,maxdivisions,electronconfig,outputpath,non_interacting,hartree_fock,natural,orbital_max_excitation,naturaltol):

    #initialise outputs
    num_total = 1
    num_rejected = 0
    num_accepted = 0

    #get initial system for finding double excitation
    v_int = idea.interactions.softened_interaction(xgrid)
    initial_potential = potential(xgrid,initial_distance,potential_name)
    initial_system = idea.system.System(xgrid,initial_potential,v_int,electrons=electronconfig)

    #if no initial excitation specified, find it for the initial distance.
    if doubleexcitation == -1:
        doubleexcitation = finddoubleexcitation(initial_system,sensitivity,limit,find_startpoint,outputpath)

    maxexcitation_gen = doubleexcitation + abovedouble
    
    #Generate state at first distance
    print(f"{datetime.datetime.now()}: Generating initial state at distance {initial_distance}, DE={doubleexcitation}",flush=True)
    distance_old = initial_distance
    distance_new  = initial_distance - distance_step
    system_old = idea.system.System(xgrid,potential(xgrid,distance_old,potential_name),v_int,electrons=electronconfig)
    with HiddenPrints():
        state_old = idea.methods.interacting.solve(system_old, k=-1, level=idea.methods.interacting._estimate_level(system_old,maxexcitation_gen))
    state_id = 1
    save_observables(state_old,system_old,doubleexcitation,distance_old,distance_old,outputpath,state_id,0)
    state_temp = state_old
    state_temp.full = state_temp.fulls[:,:,:,:,doubleexcitation]
    hf = orbitals(state_temp,system_old,state_id,distance_old,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
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
        maxexcitation_gen = doubleexcitation + abovedouble
        system_new = idea.system.System(xgrid,potential(xgrid,distance_new,potential_name),v_int,electrons=electronconfig)
        state_new = idea.methods.interacting.solve(system_new, k=-1, level=idea.methods.interacting._estimate_level(system_old,maxexcitation_gen))

        #compute inner product grid for these states
        innergrid_old_new = innerprodgrid(state_old,state_new,system_old,system_new,maxexcitation_gen)
        if debugging == True:
            save_innerprodgrid(innergrid_old_new,distance_old,distance_new,outputpath)

        #get value and index of highest inner product for old state double excitation
        de_innerprod_value = np.max(innergrid_old_new[doubleexcitation])
        de_innerprod_index = np.argmax(innergrid_old_new[doubleexcitation])

        #check if there is an inner product above 1-tolerance
        if (de_innerprod_value > (1-innerprod_tolerence)):
            #state found
            num_accepted = num_accepted + 1
            #is the current distance a multiple of the step distance?
            if (round(distance_new/distance_step,2)).is_integer():
                print(f"{datetime.datetime.now()}: Double excitation state found at distance {distance_new}, Innerproduct {de_innerprod_value}, DE={de_innerprod_index}",flush=True)
                sys.stdout.flush()
                state_id = state_id + 1
                doubleexcitation = de_innerprod_index
                save_observables(state_new,system_new,doubleexcitation,distance_new,distance_old,outputpath,state_id,innergrid_old_new)
                state_temp = state_new
                state_temp.full = state_temp.fulls[:,:,:,:,doubleexcitation]
                hf = orbitals(state_temp,system_new,state_id,distance_new,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
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
                doubleexcitation = de_innerprod_index
                save_observables(state_new,system_new,doubleexcitation,distance_new,distance_old,outputpath,state_id,innergrid_old_new)
                state_temp = state_new
                state_temp.full = state_temp.fulls[:,:,:,:,doubleexcitation]
                hf = orbitals(state_temp,system_new,state_id,distance_new,electronconfig,orbital_max_excitation,outputpath,potential_name,xgrid)
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

    idea.state.save_many_body_state(state_old.fulls[:,:,:,:,doubleexcitation],f"{outputpath}/doublestate.state")
    idea.system.save_system(system_old,f"{outputpath}/doublestate.system")
    return doubleexcitation, num_accepted, num_rejected, num_total
    
