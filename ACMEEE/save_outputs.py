import ACMEEE as acme

import numpy as np
import matplotlib.pyplot as plt
import iDEA as idea
import pickle
import glob
import contextlib
from PIL import Image


#save observable numbers & graphs

def density(s,states,excitation):
    spin_densities = np.zeros(shape=(s.x.shape[0], 2))
    for i in range(s.x.shape[0]):
        for k in range(2):
            spin_densities[i, k] = np.sum(abs(states.fulls[:,:,:,:,excitation][i, k, ...]) ** 2) * s.dx ** (s.count - 1) * s.count
    up_n = spin_densities[:, 0]
    down_n = spin_densities[:, 1]
    n = up_n + down_n
    return n

def save_observables(state,system,excitation,newdistance,olddistance,outputpath,stateid,innerprodgrid):

    #Save all states
    #statetemp = state
    #statetemp.full = statetemp.fulls[...,excitation]
    #statetemp.fulls = 0
    idea.state.save_many_body_state(state,f"{outputpath}/states/ID{str(stateid).zfill(4)}.state")

    #make wavefunction plot
    if system.electrons=="uu" or system.electrons=="dd":
    	plt.imshow(state.fulls[:,:,:,:,excitation][:,0,:,0], cmap="seismic", vmax=0.75, vmin=-0.75, extent=[-20, 20, 20, -20], aspect=1)
    elif system.electrons=="ud" or system.electrons=="du":
        plt.imshow(state.fulls[:,:,:,:,excitation][:,1,:,0], cmap="seismic", vmax=0.75, vmin=-0.75, extent=[-20, 20, 20, -20], aspect=1)
    plt.xlabel("x, position of electron 1 (Bohrs)")
    plt.ylabel("x', poisition of electron 2 (Bohrs)")
    plt.title(f"Distance from origin = {newdistance}, excitation {excitation}")
    plt.colorbar()
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.savefig(f"{outputpath}/wavefunctions/Wavefuntion-ID{str(stateid).zfill(4)}")
    plt.close()

    #make density plot
    plt.plot(system.x, density(system, state, excitation), "m-", label="Charge Density")
    plt.plot(system.x, system.v_ext, "g--", label="Potential")
    plt.xlabel("x (Bohrs)")
    plt.ylabel("v_ext / charge density")
    plt.title(f"Distance from origin = {newdistance}, excitation {excitation}")
    plt.legend()
    plt.savefig(f"{outputpath}/densities/Density-ID{str(stateid).zfill(4)}")
    plt.close()

    #make inner prod grid
    if np.sum(innerprodgrid) != 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(innerprodgrid,cmap="Purples",vmin=0,vmax=1)
        plt.ylabel(f"Excitation of state at distance={round(olddistance,2)}")
        plt.xlabel(f"Excitation of state at distance={round(newdistance,2)}")
        #for (j,i),label in np.ndenumerate(innerprodgrid):
        #    ax.text(i,j,round(label,2),ha="center", va="center")
        fig.colorbar(im,label="Inner product",boundaries=np.linspace(0, 1, 11))
        plt.savefig(f"{outputpath}/innerprods/innerprod-ID{str(stateid-1).zfill(4)}")
        plt.close()
    
    #save energy
    with open(f"{outputpath}/energies.txt","a") as file:
        file.write(f"{str(stateid).zfill(4)},{str(newdistance)},{str(excitation)},{str(state.energies[excitation])}\n")



    return

def save_innerprodgrid(innerprodgrid,olddistance,newdistance,outputpath):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(innerprodgrid,cmap="Purples",vmin=0,vmax=1)
    plt.ylabel(f"Excitation of state at distance={round(olddistance,2)}")
    plt.xlabel(f"Excitation of state at distance={round(newdistance,2)}")
    #for (j,i),label in np.ndenumerate(innerprodgrid):
    #    ax.text(i,j,round(label,2),ha="center", va="center")
    fig.colorbar(im,label="Inner product",boundaries=np.linspace(0, 1, 11))
    plt.savefig(f"{outputpath}/debugging/innerprod-D{str(newdistance)}.png")
    plt.close()
    file = open(f"{outputpath}/debugging/innerprod-D{str(newdistance)}.pkl","wb")
    pickle.dump(innerprodgrid,file)
    file.close()

    return
    
def save_density(state,excitation,system,outputpath):
    plt.plot(system.x, idea.observables.density(system, state=state), "m-", label="Charge Density")
    plt.plot(system.x, system.v_ext, "g--", label="Potential")
    plt.xlabel("x (Bohrs)")
    plt.ylabel("v_ext / charge density")
    plt.title(f"Finding DE, excitation {excitation}")
    plt.legend()
    plt.savefig(f"{outputpath}/debugging/Density-E{str(excitation).zfill(4)}")
    plt.close()
    return


def gif_wavefunctions(outputpath):
    #create gifs from saved plots
    fp_in = f"{outputpath}/wavefunctions/*.png"
    fp_out = f"{outputpath}/wavefunctions.gif"
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        # extract  first image from iterator
        img = next(imgs)
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=400, loop=0)
    return
        
def gif_densities(outputpath):
    #create gifs from saved plots
    fp_in = f"{outputpath}/densities/*.png"
    fp_out = f"{outputpath}/densities.gif"
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        # extract  first image from iterator
        img = next(imgs)
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=400, loop=0)
    return

def gif_innerproducts(outputpath):
    #create gifs from saved plots
    fp_in = f"{outputpath}/innerprods/*.png"
    fp_out = f"{outputpath}/innerproduct.gif"
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        # extract  first image from iterator
        img = next(imgs)
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=400, loop=0)
    return

def energy_graph(outputpath):
        data = []
        with open(f"{outputpath}/energies.txt","r") as file:
                arr = [line.strip().split(",") for line in file.readlines()]
                data.append(np.array(arr,dtype=np.float64))
        data = data[0]

        data = data.transpose()

        plt.plot(data[1],data[3])
        plt.gca().invert_xaxis()
        plt.xlabel("Distance from origin (Bohrs)")
        plt.ylabel("Energy of double excitation (Hartrees)")
        plt.savefig(f"{outputpath}/energies.png")
        return
