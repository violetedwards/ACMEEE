import ACMEEE as acme
import numpy as np

xgrid = np.linspace(-20,20,301)

acme.methods.assemble(xgrid,"gaussian1",10,0.25,"ud","../acme-test",trackedstate=0,abovetracked=1)
