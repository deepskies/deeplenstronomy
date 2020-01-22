# A module to run simulation functions

#Pass a command line argument of the number of images desired. Max is currently ~115,000
#Psss a command line argument of the mode
# -- Choices: 'lenses', 'foregrounds', 'galaxies'

#Example: python run_simulator.py 10000 lenses

import glob
import pandas as pd
import sys

import simulator

#interpret number of images command line argument
try:
    num_to_simualte = int(float(sys.argv[1]))
except:
    print("Specifiy the number of images desired as the first command-line argument")
    sys.exit()

if num_to_simualte <= 0:
    print("Number of images to simulate must be positive.")
    sys.exit()

#interpret simulation mode command line argument
try:
    mode = sys.argv[2]
except:
    print("Specify the simulation mode as the second command-line argument. \n\tChoose from ['lenses', 'foregrounds', 'galaxies']")
    sys.exit()

try:
    assert mode in ['lenses', 'foregrounds', 'galaxies']
except:
    print("Simulation mode must be chosen from ['lenses', 'foregrounds', 'galaxies']")
    sys.exit()

#obtain light curve files
data_dir = 'lcs_plus_gal_param/'
lcs_list = glob.glob(data_dir + '*.csv')


#trim list to desired number of light curves
#use all light curves if num_to_simulate > len(lcs_list)
if num_to_simualte > len(lcs_list):
    print("Warning: only enough light curves to simulate %i images. Defaulting to maximum." %len(lcs_list))
else:
    lcs_list = lcs_list[0:num_to_simulate]



#main loop for simulation
total = float(len(lcs_list))
counter = 0
num_pix = 100
images = np.zeros((num_to_simualte, num_pix, num_pix, 4))
for lc_file in lcs_list:
    #track progress
    counter += 1
    progress = counter / total * 100
    sys.stdout.write('\rProgress:  %.3f %%' %progress)
    sys.stdout.flush()
    
    #read data file
    data = pd.read_csv(lc_file)

    #simulate images based on mode
    if mode == 'lenses':
        mean_images = simulator.generate_mean_sim_lens(data)
    elif mode == 'foregrounds':
        mean_images = simulator.generate_mean_sim_non_lens_agn(data)
    elif mode == 'galaxies':
        mean_images = simulator.generate_mean_sim_non_lens_gal(data)
    else:
        #shouldn't get here
        pass

    
    #save images
    images[counter] = mean_images

#save images
np.save('output_%i_%s.npy' %(num_to_simulate, mode), images)

print("Done!")
