import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from Utils.Outcomes import *

def get_patient_dct(df,ID):
    #converts patient data to dictionary 
    #that can be parsed by Simulator object
    pt = df.loc[ID]
    pt_dct = {}
    pt_dct['ID'] = pt['IDs']
    pt_dct['mrs'] = pt['mrs_def']
    pt_dct['noEVT'] = 1
    pt_dct['noEVT*core_vol'] = int(pt['core_vol'])
    pt_dct['IVT'] = pt['ivt_given']
    pt_dct['age'] = pt['r_age']
    pt_dct['sex'] = pt['r_sex']
    pt_dct['occloc'] = pt['bl_occloc']
    return pt_dct

def simulate_IDs(IDs,df,Simulator, verbal=True):
    #IDs: selection of IDs in df used for simulation (can be for subgroup only)
    #df: dataframe with index=ID, columns=variables and mrs
    #Simulator: class object initialized for simulations (see Simulate.py)
    # returns:
        # total_res: for each arm per patient in columns the Costs and Qalys per strategy
        # extended_res: each row contains per ID,treatment,CTP use, and year the mRS, Costs, Qalys
    out = []
    tot_res = []
    if verbal:
        IDs = tqdm(IDs)
    else:
        IDs = list(IDs)

    for ID in IDs:
        pt_dct = get_patient_dct(df,ID)
        #per patient get tmp_res: results
        # store all in a single df
        tmp = Simulator(pt_dct) #simulate a single patient
        out.append(tmp)
        #per patient get the total costs and QALYs
        tot = tmp.groupby(by=['ctp','treatment'])[['TotCosts','TotQALYs']].sum(axis=1)
        #get all totals in a single row per pt ID
        tot_res.append([pt_dct['ID'],
                       *list(tot.loc['CTP','EVT'].values),
                        *list(tot.loc['CTP','noEVT'].values),
                        *list(tot.loc['noCTP','EVT'].values),
                        *list(tot.loc['noCTP','noEVT'].values)
                       ])
    extend_res = pd.concat(out,axis=0)
    
    totals_res = pd.DataFrame(tot_res,columns=['ID','C_ctp_evt','Q_ctp_evt',
                                           'C_ctp_noevt','Q_ctp_noevt',
                                           'C_noctp_evt','Q_noctp_evt',
                                           'C_noctp_noevt','Q_noctp_noevt'])
    totals_res.index = totals_res.ID
    
    return totals_res, extend_res


def probabilistic_simulation(S,df,N_resamples,N_patients_per_cohort,seed=21):
    #S: initialized Simulate class object (contains all the data and methods)
    #df: dataframe with index=ID and columns with patient variables
    #N_resamples: Number of probabilistic resamples to run
    #N_patients_per_cohort: Number of patients per cohort
    #returns: similar to return of simulate_IDs but per N_resmaples iteration
    np.random.seed(seed)
    tots,exts = [],[]
    for i in tqdm(range(N_resamples)):
        #draw new parameters from initialized distributions
        S._probabilistic_resample()
        #sample a cohort of IDs with replaceemnt
        smpl = df.sample(N_patients_per_cohort,replace=True, random_state=seed).index
        #simulation output
        totals_res, extend_res = simulate_IDs(smpl,df,S,False)
        totals_res['simno'] = i
        extend_res['simno'] = i
        tots.append(totals_res)
        exts.append(extend_res)
    
    totals_res = pd.concat(tots)
    extend_res = pd.concat(exts)
    
    return totals_res, extend_res

def subgroup_psa(df,
                 col_subgroup,
                 Sim,
                 N_resamples,
                 N_patients_per_cohort,
                 thresholds=np.arange(0,151,10),
                 costs_per_ctp=0,
                 multiply_ctp_costs=0,
                 miss_percentage=[0],
                 WTP=80000,
                 seed=21):
    #performs PSA for each subgroup
    #just splits each input cohort
    np.random.seed(seed)
    #set all seeds!!
    bl_dct = df.to_dict(orient='index')
    CEA_res = []
    for sgroup in df[col_subgroup].unique():
        tmp = df[df[col_subgroup]==sgroup]
        totals_res,extend_res = probabilistic_simulation(Sim,tmp,
                                                        N_resamples,N_patients_per_cohort)

        outs, aggrs = probabilistic_cohort_outcomes(totals_res,
                                                    bl_dct,
                                                    costs_per_ctp=costs_per_ctp,
                                                    multiply_ctp_costs = multiply_ctp_costs,
                                                    miss_percentage = miss_percentage,
                                                    WTP=WTP)
        aggrs[col_subgroup] = str(sgroup)
        CEA_res.append(aggrs)
    CEA_res = pd.concat(CEA_res)
    return CEA_res


def shift_EVT_OR(CPG,shift):
    #Used to shift the treatment effect (EVT) odds ratio in original values (1.86)
    #CPG: ControlPatientGeneration class object 
    #shift: OR point shift (from original reported OR)
    cpg_params = CPG._get_params(current=False) #current=False returns original df and shifts the original data
    new_cpg_params = cpg_params.copy()
    new_cpg_params.loc['noEVT',['OR','CI_low','CI_high']] = 1/(1/cpg_params.loc['noEVT',['OR','CI_low','CI_high']]+shift)
    CPG._set_params(new_cpg_params)
    return CPG

def shift_EVT_core_volume_OR(CPG,shift):
    #used to shift EVT effect modification value by core_volume in original value (0.98)
    #CPG: ControlPatientGeneration class object 
    #shift: OR point shift (from original reported OR)
    cpg_params = CPG._get_params(current=False) #current=False returns original df and shifts the original data
    new_cpg_params = cpg_params.copy()
    
    #extract OR for effect modification (EVT*core_vol; input considers noEVT)
    params = (1/new_cpg_params.loc['noEVT*core_vol',['OR','CI_low','CI_high']]).astype('float')
    #return to per 10 mL (the input OR), add shift, return to per mL
    params = 1/np.exp(np.log(np.exp(np.log(params)*10)+shift)/10)
    new_cpg_params.loc['noEVT*core_vol',['OR','CI_low','CI_high']] = params
    CPG._set_params(new_cpg_params)
    return CPG

#PSA for a set of possible OR shifts
def OR_shift_psa(df,
                 OR_evt_shifts,
                 OR_evt_corevol_shifts,
                 Sim,
                 N_resamples,
                 N_patients_per_cohort,
                 thresholds=np.arange(0,151,10),
                 WTP=80000,
                 seed=21):
    #special type of sensitivity analyses:
    # the input 90d ORs are altered
    np.random.seed(seed)

    #set all seeds!!
    bl_dct = df.to_dict(orient='index')
    res_aggr = []
    res_extended = []
    for evt_shift in OR_evt_shifts:
        for evt_corevol_shift in OR_evt_corevol_shifts:
            Sim.CPG = shift_EVT_OR(Sim.CPG,evt_shift)
            Sim.CPG = shift_EVT_core_volume_OR(Sim.CPG,evt_corevol_shift)
            totals_res,extend_res = probabilistic_simulation(Sim,df,
                                                             N_resamples,N_patients_per_cohort)
            outs, aggrs = probabilistic_cohort_outcomes(totals_res,
                                                        bl_dct,
                                                        costs_per_ctp=0, #should only be used for M2 miss sims
                                                        multiply_ctp_costs = 0, #should only be used for M2 miss sims
                                                        miss_percentage = [0], #should only be used for M2 miss sims
                                                        WTP=WTP)
            aggrs['OR_evt_shift'] = evt_shift
            aggrs['OR_evt_corevol_shift'] = evt_corevol_shift

            outs['OR_evt_shift'] = evt_shift
            outs['OR_evt_corevol_shift'] = evt_corevol_shift
            
            res_aggr.append(aggrs)
            res_extended.append(outs)

    res_aggr = pd.concat(res_aggr)
    res_extended = pd.concat(res_extended)
    return res_extended, res_aggr

def M2_miss_psa(df,
                 Sim,
                 N_resamples, #10,000 in protocol --> use 1,000
                 N_patients_per_cohort, #100 pt in protocol
                 screening_multipliers, 
                 OR_evt_shift=0, #it is likely EVT effect is lower in M2
                 miss_percentage=[0,.1,.2,.3,.4,.5],
                 WTP=80000,
                 seed=21):
    np.random.seed(seed)
    #function runs PSA simulations
    
    #EVT initial shift
    if OR_evt_shift!=0:
        Sim.CPG = shift_EVT_OR(Sim.CPG,OR_evt_shift)
    
    #set all seeds!!
    #screening_multiplier: multiplied with CTP costs to 
    #represent number needed to image before a 'missed' M2 
    #is detected --> thus population costs are in the analysis
    bl_dct = df.to_dict(orient='index')
    
    
    totals_res,extend_res = probabilistic_simulation(Sim,
                                                     df[(df['bl_occloc']=='M2')|(df['bl_occloc']=='M3')],#only simulate for M2
                                                     N_resamples,
                                                     N_patients_per_cohort)
    ext_out, aggr_out = [],[]
    for NNI_multiplier in screening_multipliers:
        outs, aggrs = probabilistic_cohort_outcomes(totals_res, 
                                                    bl_dct,
                                                    thresholds=[2000], #nobody should be excluded
                                                    costs_per_ctp=Sim.C.costs_CTP,
                                                    multiply_ctp_costs = NNI_multiplier,
                                                    miss_percentage = miss_percentage,
                                                    WTP = WTP)
        outs['NNI_multiplier'] = NNI_multiplier
        aggrs['NNI_multiplier'] = NNI_multiplier
        ext_out.append(outs)
        aggr_out.append(aggrs)

    res_aggr = pd.concat(aggr_out)
    res_extended = pd.concat(ext_out)
    return res_extended, res_aggr 