import numpy as np
import matplotlib.pyplot as plt
import iDEA as idea
import glob
import contextlib
from PIL import Image

def potential(x,d,potential_name):

    if potential_name == "triangular":
        potential1 = np.zeros(300)
        potential2 = np.zeros(300)
        for xindex in range(0,len(x)):
            if x[xindex]<=-d-2:
                potential1[xindex] = 0
            if -d-2<x[xindex]<=-d:
                potential1[xindex] = -4*(x[xindex]+d)-8
            if -d<x[xindex]<=-d+2:
                potential1[xindex] = 4*(x[xindex]+d)-8
            if -d+2<x[xindex]:
                potential1[xindex] = 0
        for xindex in range(0,len(x)):
            if x[xindex]<=d-2:
                potential2[xindex] = 0
            if d-2<x[xindex]<=d:
                potential2[xindex] = -4*(x[xindex]-d)-8.01
            if d<x[xindex]<=d+2:
                potential2[xindex] = 4*(x[xindex]-d)-8.01
            if d+2<x[xindex]:
                potential2[xindex] = 0

        return potential1 + potential2
    
    if potential_name == "gaussian1":
        return (-4*np.exp(-((x-d)**2)/10) - 4.005*np.exp(-((x+d)**2)/10))

    if potential_name == "symgaussian":
        return (-4*np.exp(-((x-d)**2)/10) - 4*np.exp(-((x+d)**2)/10))

    if potential_name == "square1":
        well_depth = 4
        well_width = 5
        xpoints = len(x)
        xpoint_range = abs(max(x))+abs(min(x))
        #create full array
        v_full = np.ones(xpoints)*well_depth
        #remove elements around the wells
        separation_x = d*(xpoints/xpoint_range)
        well_width_x = well_width*(xpoints/xpoint_range)
        v_full[int((xpoints/2)-separation_x-(well_width_x/2)):int((xpoints/2)-separation_x+(well_width_x/2))] = 0
        v_full[int((xpoints/2)+separation_x-(well_width_x/2)):int((xpoints/2)+separation_x+(well_width_x/2))] = 0.01

        v_ext = v_full-4
        return(v_ext)
            
    if potential_name == "square2":
        well_depth = 4
        well_width = 10
        xpoints = len(x)
        xpoint_range = abs(max(x))+abs(min(x))
        #create full array
        v_full = np.ones(xpoints)*well_depth
        #remove elements around the wells
        separation_x = d*(xpoints/xpoint_range)
        well_width_x = well_width*(xpoints/xpoint_range)
        v_full[int((xpoints/2)-separation_x-(well_width_x/2)):int((xpoints/2)-separation_x+(well_width_x/2))] = 0
        v_full[int((xpoints/2)+separation_x-(well_width_x/2)):int((xpoints/2)+separation_x+(well_width_x/2))] = 0.1

        v_ext = v_full-4
        return(v_ext)
    
    if potential_name == "square3":
        well_depth = 4
        well_width = 2
        xpoints = len(x)
        xpoint_range = abs(max(x))+abs(min(x))
        #create full array
        v_full = np.ones(xpoints)*well_depth
        #remove elements around the wells
        separation_x = d*(xpoints/xpoint_range)
        well_width_x = well_width*(xpoints/xpoint_range)
        v_full[int((xpoints/2)-separation_x-(well_width_x/2)):int((xpoints/2)-separation_x+(well_width_x/2))] = 0
        v_full[int((xpoints/2)+separation_x-(well_width_x/2)):int((xpoints/2)+separation_x+(well_width_x/2))] = 0.01

        v_ext = v_full-4
        return(v_ext)
    
    if potential_name == "hookes":
        # Initialize result array with zeros
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        # First condition: -4 <= (x - d) <= 4
        condition1 = (-4 <= (x - d)) & (x - d <= 4)
        
        # Second condition: -4 <= (x + d) <= 4
        condition2 = (-4 <= (x + d)) & (x + d <= 4)

        # Apply the first formula where the first condition is true
        arr1[condition1] = 0.25 * (x[condition1] - d)**2 - 4.3
        
        # Apply the second formula where the second condition is true
        arr2[condition2] = 0.25 * (x[condition2] + d)**2 - 4

        return arr1+arr2
    
    if potential_name == "trapezoid":
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        condition1= ((d - 1) <= x )& ((d+1) >= x)
        condition2= ((d - 2) <= x) & ((d-1) >= x)
        condition3= ((d + 1) <= x) & ((d+2) >= x)

        arr1[condition1] = -4
        arr1[condition2] = -4*(x[condition2]-d+2)
        arr1[condition3] = 4*(x[condition3]-d-2)

        condition1= ((-d - 1) <= x) & ((-d+1) >= x)
        condition2= ((-d - 2) <= x) & ((-d-1) >= x)
        condition3= ((-d + 1) <= x) & ((-d+2) >= x)

        arr2[condition1] = -4.01
        arr2[condition2] = -4*(x[condition2]+d+2)
        arr2[condition3] = 4*(x[condition3]+d-2)

        return arr1+arr2
    
    if potential_name == "cosine":
        # Initialize result array with zeros
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        # First condition: -pi <= (x - d) <= pi
        condition1 = (-np.pi <= (x - d)) & (x - d <= np.pi)
        
        # Second condition: -pi <= (x + d) <= pi
        condition2 = (-np.pi <= (x + d)) & (x + d <= np.pi)

        # Apply the first formula where the first condition is true
        arr1[condition1] = -2.1 * np.cos(x[condition1] - d) - 2.1
        
        # Apply the second formula where the second condition is true
        arr2[condition2] = -2 * np.cos(x[condition2] + d) - 2

        return arr1+arr2
    
    if potential_name == "snake":
        return
    
    if potential_name == "circular":
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        condition1= (d-4<=x) & (d+4>=x)
        condition2= (-d-4<=x) & (-d+4>=x)

        arr1[condition1] = -np.abs(np.sqrt((16-(x[condition1]-d)**2)))
        arr2[condition2] = -np.abs(np.sqrt((16-(x[condition2]+d)**2)))
        return arr1+arr2
    
    if potential_name == "house":
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        condition1 = (d<=x) & (x<=d+2)
        condition2 = (d-2<=x) & (x<=d)

        arr1[condition1] = (x[condition1]-d-2)-4
        arr1[condition2] = -(x[condition2]-d+2)-4

        condition1 = (-d<=x) & (x<=-d+2)
        condition2 = (-d-2<=x) & (x<=-d)

        arr2[condition1] = (x[condition1]+d-2)-4.01
        arr2[condition2] = -(x[condition2]+d+2)-4.01

        return arr1+arr2
    
    if potential_name == "r-triangle":
        arr1 = np.zeros_like(x)
        arr2 = np.zeros_like(x)

        condition1 = (d-4.01<=x) & (x<=d+4)
        condition2 = (-d-4<=x) & (x <= -d+4)

        arr1[condition1] = (2/3)*(x[condition1]-d-4.01)
        arr2[condition2] = (2/3)*(x[condition2]+d-4)

        return arr1 + arr2
    
    if potential_name == "psuedo":
        return
    
    if potential_name == "gaussian2":
        return (-8*np.exp(-((x-d)**2)/10) - 8.005*np.exp(-((x+d)**2)/10))
        
    
    if potential_name == "gaussian3":
        return (-2*np.exp(-((x-d)**2)/10) - 2.005*np.exp(-((x+d)**2)/10))
    
    if potential_name == "gaussian4":
        return
    
    if potential_name == "gaussian5":
        return
    
    if potential_name == "mypotential":
        return

def create_potential_gif(xgrid,initial_distance,potential_name,outputpath):
    for i in range(0,50):
        distance = initial_distance-(initial_distance/50)*i
        v_int = idea.interactions.softened_interaction(xgrid)
        system = idea.system.System(xgrid,potential(xgrid,distance,potential_name),v_int,electrons="uu")
        plt.plot(system.x, system.v_ext, "g--", label="Potential")
        plt.xlabel("x (Bohrs)")
        plt.ylabel("v_ext (Hartrees)")
        plt.legend()
        plt.savefig(f"{outputpath}/debugging/potentialplot-ID{str(i).zfill(4)}.png")
        plt.close()

    fp_in = f"{outputpath}/debugging/potentialplot-*.png"
    fp_out = f"{outputpath}/potential.gif"
    with contextlib.ExitStack() as stack:
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
        # extract  first image from iterator
        img = next(imgs)
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=100, loop=0)
    return
