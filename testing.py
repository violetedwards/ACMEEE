import ACMEEE as acme
import matplotlib.pyplot as plt
import numpy as np

xgrid = np.linspace(-20,20,301)

#check initial state to be tracked
#acme.methods.plotstate(xgrid, "gaussian1", 10, "ud", 5)

acme.methods.assemble(xgrid,"gaussian1",10,0.25,"ud","../acme-test-ud",trackedstate=5,abovetracked=5)


