import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def get_patient_dct(df,ID):
    pt = df.loc[ID]
    pt_dct = {}
    pt_dct['ID'] = pt['IDs']
    pt_dct['mrs'] = pt['mrs_def']
    pt_dct['noEVT'] = 1
    pt_dct['noEVT*core_vol'] = int(pt['core_vol'])
    pt_dct['IVT'] = pt['ivt_given']
    pt_dct['age'] = pt['r_age']
    pt_dct['sex'] = pt['r_sex']
    return pt_dct

def simulate_IDs(IDs,df,Simulator):
    out = []
    tot_res = []
    for ID in tqdm(IDs):
        pt_dct = get_patient_dct(df,ID)
        #per patient get tmp_res
        # store all in a single df
        tmp = Simulator(pt_dct)
        out.append(tmp)
        #per patient get the total costs and QALYs
        tot = tmp.groupby(by=['ctp','treatment'])[['TotCosts','TotQALYs']].sum(axis=1)
        #get all totals in a single row per pt ID
        tot_res.append([pt_dct['ID'],
                       *list(tot.loc['CTP','EVT'].values),
                        *list(tot.loc['CTP','noEVT'].values),
                        *list(tot.loc['noCTP','EVT'].values)
                       ])
    extend_res = pd.concat(out,axis=0)
    
    totals_res = pd.DataFrame(tot_res,columns=['ID','C_ctp_evt','Q_ctp_evt',
                                           'C_ctp_noevt','Q_ctp_noevt',
                                           'C_noctp','Q_noctp'])
    totals_res.index = totals_res.ID
    
    return totals_res, extend_res






