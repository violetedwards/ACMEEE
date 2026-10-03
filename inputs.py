
import numpy as np
from potential import *
#------------------------------------------------
#================================================
#=====INDIGO'S DOUBLE EXCITATION ASSEMBLER=======
#====================Idea========================
#===============PARAMETER FILE===================
#================================================
#------------------------------------------------


#the spacial grid that the wavefunctions are solved over. This example spans -20<x<20 with 301 gridpoints
xgrid = np.linspace(-20,20,301)

#the name of the potential that will be used, see potential.py 
potential_name = "symgaussian"

#output extra info helpful for debugging, If true, outputs inner product grids for every state generated, even if rejected.
debugging = True 

#the initial excitation of the system, set to -1 if you want InDEX to identify the lowest energy double excitation
doubleexcitation = 1

#if finding double excitation, only excitations above this value will be searched. If double excitation is known, set to 0.
find_startpoint = 0 

#Initial distance of the wells from 0 in bohr. An initial distance of 5 will have the wells be separated by 10 bohr
initial_distance = 5

#Sensitivity of the double excitation peak finder. Default 5
sensitivity = 5 

#Maimum exictation number that will be searched when trying to find the lowest energy double excitation
limit = 50

#number of excitation above the target excitation that will be generated during the adiabatic movement. e.g. if state 7 is to be tracked, and abovedouble=5, the algorithm will generate states 0-12.
abovedouble = 2

#Inner product tolerance used for accepting states in the adiabatic movement. e.g. tol=0.1 will accept states that has an inner product >0.9
innerprod_tolerence = 0.5

#Initial distance step in bohr
distance_step = 0.25 

#max number of step divisions before the assembler gives up
maxdivisions = 30

#spin configuration of the electrons
electronconfig = "uu"

#The path for the outputs
outputpath = "../symgaussian-uu-e5"

#Select the job. plotpoential only outputs the potential plot. find only runs the double excitation finder. assemble runs the whole program, executing the adiabatic movement method.
job = "assemble" #"assemble" #"find" #"plotpotential"

#some other shit that needs sorting
non_interacting = True
hartree_fock = True
natural = True
orbital_max_excitation = 20
naturaltol = 5
