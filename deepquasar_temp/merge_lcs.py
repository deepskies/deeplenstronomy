# A module to merge and flatten light curves

import glob                                                                                                       
import numpy as np                                                                                                
import pandas as pd                                                   
import os

os.system('mkdir merged_lcs')                                            


# collect file names                                                                                                                  
def structure_data(files):                                                                                   
    ids = np.unique([x.split('/')[-1][0:-5] for x in files])                                                      
    id_dict = {}                                                                                                  
    for id in ids:                                                                                                
        id_dict[id] = {'g': 'extracted_data/%s_g.lc' %id,                                                         
                       'r': 'extracted_data/%s_r.lc' %id,                                                         
                       'i': 'extracted_data/%s_i.lc' %id,                                                         
                       'z': 'extracted_data/%s_z.lc' %id}                                                         
                                                                                                                  
    return ids, id_dict                                                                                           
                                                                                                                  
agn_lcs_files = glob.glob('extracted_data/*.lc')
agn_ids, agn_dict = structure_data(agn_lcs_files)

# iterate through the unique agn_ids, group by light curve, read into dataframe
for agn_id in agn_ids:
    
    #concatenate all light curves, throw a flag if a band is missing from a lightcurve
    g_flag, r_flag, i_flag, z_flag = False, False, False, False

    dfs = []
    for band, filename in agn_dict[agn_id].items():
        try:
            eval('dfs.append(pd.read_csv("%s"))'%filename)
        except:
            print(filename)
            eval(band + '_flag = True')

    df = pd.concat(dfs)

    #iterate through the nites, and make an entry for each nite
    nites = np.unique(df['NITE'].values)
    cols = ['NITE', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'psf_g', 'psf_r', 'psf_i', 'psf_z', 'skymag_g',
            'skymag_r', 'skymag_i', 'skymag_z', 'zpt_g', 'zpt_r', 'zpt_i', 'zpt_z', 'z_source']

    z_source = np.max(df['Z'].values)

    parameters = [] 
    for nite in nites:
        nite_df = df[df['NITE'] == nite]

        if not g_flag:
            g_nite_df = nite_df[nite_df['FLT'] == 'g']
            mag_g = g_nite_df['MAG'].values[0]
            psf_g = g_nite_df['PSF'].values[0]
            zpt_g = g_nite_df['ZPTMAG'].values[0] 
            sky_g = g_nite_df['SKYMAG'].values[0]
        else:
            mag_g, psf_g, zpt_g, sky_g = 99, 99, 99, 99

        if not r_flag:
            r_nite_df = nite_df[nite_df['FLT'] == 'r']
            mag_r = r_nite_df['MAG'].values[0]
            psf_r = r_nite_df['PSF'].values[0]
            zpt_r = r_nite_df['ZPTMAG'].values[0]
            sky_r = r_nite_df['SKYMAG'].values[0]
        else:
            mag_r, psf_r, zpt_r, sky_r = 99, 99, 99, 99

        if not i_flag:
            i_nite_df = nite_df[nite_df['FLT'] == 'i']
            mag_i = i_nite_df['MAG'].values[0]
            psf_i = i_nite_df['PSF'].values[0]
            zpt_i = i_nite_df['ZPTMAG'].values[0]
            sky_i = i_nite_df['SKYMAG'].values[0]
        else:
            mag_i, psf_i, zpt_i, sky_i = 99, 99, 99, 99
            
        if not z_flag:
            z_nite_df = nite_df[nite_df['FLT'] == 'z']
            mag_z = z_nite_df['MAG'].values[0]
            psf_z = z_nite_df['PSF'].values[0]
            zpt_z = z_nite_df['ZPTMAG'].values[0]
            sky_z = z_nite_df['SKYMAG'].values[0]
        else:
            mag_z, psf_z, zpt_z, sky_z = 99, 99, 99, 99
            

        nite_data = [nite,
                     mag_g, mag_r, mag_i, mag_z,
                     psf_g, psf_r, psf_i, psf_z,
                     sky_g, sky_r, sky_i, sky_z,
                     zpt_g, zpt_r, zpt_i, zpt_z,
                     z_source]

        parameters.append(nite_data)
    
    #output merged file
    pd.DataFrame(data=parameters, columns=cols).to_csv('merged_lcs/' + agn_id + '.lc')

