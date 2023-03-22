import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from Utils.Outcomes import *

def get_patient_dct(df,ID, xvar='core_vol'):
    #converts patient data to dictionary 
    #that can be parsed by Simulator object
    pt = df.loc[ID]
    pt_dct = {}
    pt_dct['ID'] = pt['IDs']
    pt_dct['mrs'] = pt['mrs_def']
    pt_dct['noEVT'] = 1
    pt_dct['noEVT*core_vol'] = int(pt[xvar])
    pt_dct['IVT'] = pt['ivt_given']
    pt_dct['age'] = pt['r_age']
    pt_dct['sex'] = pt['r_sex']
    pt_dct['occloc'] = pt['bl_occloc']
    return pt_dct

def simulate_IDs(IDs,df,Simulator,xvar='core_vol', verbal=True):
    #IDs: selection of IDs in df used for simulation (can be for subgroup only)
    #df: dataframe with index=ID, columns=variables and mrs
    #Simulator: class object initialized for simulations (see Simulate.py)
    #xvar: what variable from df (core_vol, penumbra_vol, or mm-ratio) 
    #should be used with ORs to compute control arm
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
        pt_dct = get_patient_dct(df,ID,xvar)
        #per patient get tmp_res: results
        # store all in a single df
        tmp = Simulator(pt_dct) #simulate a single patient
        out.append(tmp)
        #per patient get the total costs and QALYs
        tot = tmp.groupby(by=['ctp','treatment'])[['TotCosts','TotQALYs']].sum(axis=1)
        #get all totals in a single row per pt ID
        tot_res.append([
                        pt_dct['ID'],
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


def probabilistic_simulation(S,df,N_resamples,N_patients_per_cohort,biased_threshold_sample=None,xvar='core_vol',seed=21):
    #S: initialized Simulate class object (contains all the data and methods)
    #df: dataframe with index=ID and columns with patient variables
    #N_resamples: Number of probabilistic resamples to run
    #N_patients_per_cohort: Number of patients per cohort
    #xvar: what variable from df (core_vol, penumbra_vol, or mm-ratio) 
    #should be used with ORs to compute control arm
    #returns: similar to return of simulate_IDs but per N_resmaples iteration
    #biased_threshold_sample: dictionary with variable_colume, threshold, smaller_larger, percentage bias

    np.random.seed(seed)
    tots,exts = [],[]
    for i in tqdm(range(N_resamples)):
        if N_patients_per_cohort=='auto':
            nppc = len(df)
        else:
            nppc = N_patients_per_cohort
        #draw new parameters from initialized distributions
        S._probabilistic_resample()
        #print(S.CPG.dct_OR)
        #sample a cohort of IDs with replaceemnt
        if biased_threshold_sample is None:
            smpl = df.sample(nppc,replace=True, random_state=i).index
        else:
            colvar = biased_threshold_sample['variable_column']
            thr = biased_threshold_sample['threshold']
            larger_smaller = biased_threshold_sample['larger_smaller']
            if larger_smaller=='larger':
                condition = (df[colvar]>=thr)
            elif larger_smaller=='smaller':
                condition = (df[colvar]<=thr)
            bias_n_pt = int(round(biased_threshold_sample['bias_percentage']*nppc))
            non_bias_pt = nppc-bias_n_pt

            bias_smpl = df[condition].sample(bias_n_pt,replace=True, random_state=i).index
            nonbias_smpl = df[~condition].sample(non_bias_pt,replace=True, random_state=i).index
            smpl = np.hstack([bias_smpl,nonbias_smpl])

        #simulation output
        totals_res, extend_res = simulate_IDs(smpl,df,S,xvar,False)
        totals_res['simno'] = i
        totals_res['OR_noEVT'] = S.CPG.dct_OR['noEVT']
        totals_res['OR_noEVT_corevol'] = S.CPG.dct_OR['noEVT*core_vol']
        extend_res['simno'] = i
        extend_res['OR_noEVT'] = S.CPG.dct_OR['noEVT']
        extend_res['OR_noEVT_corevol'] = S.CPG.dct_OR['noEVT*core_vol']

        tots.append(totals_res)
        exts.append(extend_res)
    
    totals_res = pd.concat(tots) #includes results (d_costs, d_qalys, NMB, ICER)
    extend_res = pd.concat(exts) #also has mRS per year and costs/qalys per arm
    
    return totals_res, extend_res

def subgroup_psa(df,
                 col_subgroup,
                 Sim,
                 N_resamples,
                 N_patients_per_cohort='auto',
                 thresholds=np.arange(0,151,10),
                 larger_smaller='larger',
                 costs_per_ctp=0,
                 multiply_ctp_costs=0,
                 miss_percentage=[0],
                 xvar='core_vol',
                 WTP=80000,
                 seed=21):
    #performs PSA for each subgroup
    #xvar: what variable from df (core_vol, penumbra_vol, or mm-ratio) 
    #should be used with ORs to compute control arm
    #just splits each input cohort
    np.random.seed(seed)

    #set all seeds!!
    bl_dct = df.to_dict(orient='index')
    CEA_res = []
    for sgroup in df[col_subgroup].unique():
        tmp = df[df[col_subgroup]==sgroup]
        if N_patients_per_cohort=='auto':
            nppc = len(tmp)
        else:
            nppc = N_patients_per_cohort
        totals_res,extend_res = probabilistic_simulation(Sim,tmp,
                                                        N_resamples,
                                                        nppc,xvar=xvar)

        outs, aggrs,__ = probabilistic_cohort_outcomes(totals_res,
                                                    bl_dct,
                                                    thresholds = thresholds,
                                                    larger_smaller=larger_smaller,
                                                    costs_per_ctp=costs_per_ctp,
                                                    multiply_ctp_costs = multiply_ctp_costs,
                                                    miss_percentage = miss_percentage,
                                                    WTP=WTP)
        aggrs[col_subgroup] = str(sgroup)
        CEA_res.append(aggrs)
    CEA_res = pd.concat(CEA_res)
    return CEA_res

#These two functions first reset default params --> 
#if used consecutively only the last change is used
def shift_EVT_OR(CPG,shift):
    #Used to shift the treatment effect (EVT) odds ratio in original values (1.86)
    #CPG: ControlPatientGeneration class object 
    #shift: OR point shift (from original reported OR)
    cpg_params = CPG._get_params(current=False) #current=False returns original df and shifts the original data
    new_cpg_params = cpg_params.copy()
    params = 1/(1/new_cpg_params.loc['noEVT',['OR','CI_low','CI_high']]+shift).astype('float')
    print(params)
    new_cpg_params.loc['noEVT',['OR','CI_low','CI_high']] = params
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

#this funciton shift ORs in one go
def shift_ORs(CPG, shift_dct):
    cpg_params = CPG._get_params(current=False)
    new_cpg_params = cpg_params.copy()
    for variable,shift in shift_dct.items():
        if variable=='EVT':
            params = 1/(1/new_cpg_params.loc['noEVT',['OR','CI_low','CI_high']]+shift).astype('float')
            new_cpg_params.loc['noEVT',['OR','CI_low','CI_high']] = params
        elif variable=='EVT*core_vol':
            params = (1/new_cpg_params.loc['noEVT*core_vol',['OR','CI_low','CI_high']]).astype('float')
            #return to per 10 mL (the input OR), add shift, return to per mL
            params = 1/np.exp(np.log(np.exp(np.log(params)*10)+shift)/10)
            new_cpg_params.loc['noEVT*core_vol',['OR','CI_low','CI_high']] = params
        else:
            print('Unkown variable and shift:',variable,shift)
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
                 larger_smaller='larger', #no EVT if above threshold
                 biased_threshold_sample = None, #biased_threshold_sample: dictionary with variable_column, threshold, smaller_larger, percentage bias
                 WTP=80000,
                 xvar = 'core_vol',
                 root_sav=None,
                 seed=21):
    #special type of sensitivity analyses:
    # the input 90d ORs are altered

    #xvar: what variable from df (core_vol, penumbra_vol, or mm-ratio) 
    #should be used with ORs to compute control arm
    
    np.random.seed(seed)

    #set all seeds!!
    bl_dct = df.to_dict(orient='index')
    res_aggr, res_extended = [],[] #cea res

    sim_tot, sim_ext = [], [] #sim res
    fullres = []
    for evt_shift in OR_evt_shifts:
        for evt_corevol_shift in OR_evt_corevol_shifts:
            if N_patients_per_cohort=='auto':
                nppc = len(df)
            else:
                nppc = N_patients_per_cohort
            f1 = os.path.join(root_sav,'EVT{}_EVT-ECV{}_aggregated_psa_res.pic'.format(evt_shift,evt_corevol_shift))
            f2 = os.path.join(root_sav,'EVT{}_EVT-ECV{}_extended_psa_res.pic'.format(evt_shift,evt_corevol_shift))
            if os.path.exists(f1):
                print('Skipping:', evt_shift, evt_corevol_shift)
                continue
            print('Running:', evt_shift, evt_corevol_shift)
            Sim.CPG = shift_ORs(Sim.CPG,shift_dct={'EVT':evt_shift, 'EVT*core_vol':evt_corevol_shift})
            totals_res,extend_res = probabilistic_simulation(Sim,
                                                             df,
                                                             N_resamples,
                                                             nppc,
                                                             biased_threshold_sample=biased_threshold_sample,
                                                             xvar=xvar)

            outs, aggrs, fr = probabilistic_cohort_outcomes(totals_res,
                                                        bl_dct,
                                                        thresholds=thresholds,
                                                        larger_smaller=larger_smaller,
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

            totals_res['OR_evt_shift'] = evt_shift
            totals_res['OR_evt_corevol_shift'] = evt_corevol_shift

            extend_res['OR_evt_shift'] = evt_shift
            extend_res['OR_evt_corevol_shift'] = evt_corevol_shift

            #sim_tot.append(totals_res)
            #sim_ext.append(extend_res)
            if root_sav is not None:
                if not os.path.exists(root_sav):
                    os.makedirs(root_sav)

                res_aggr = pd.concat(res_aggr)
                res_extended = pd.concat(res_extended)
                res_aggr.to_pickle(f1)
                res_extended.to_pickle(f2)
                res_aggr, res_extended = [],[]

    if root_sav is not None:
        res_aggr = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_aggregated_psa_res' in f])
        #res_extended = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_extended_psa_res' in f])

        res_aggr.to_pickle(os.path.join(root_sav,'full_aggregated_psa_res.pic'))
        #res_extended.to_pickle(os.path.join(root_sav,'full_extended_psa_res.pic'))
        return res_aggr
    else:
        res_aggr = pd.concat(res_aggr)
        res_extended = pd.concat(res_extended)
        #sim_tot = pd.concat(sim_tot)
        #sim_ext = pd.concat(sim_ext)
        return res_extended, res_aggr #, sim_ext, sim_tot

def M2_miss_psa(df,
                 Sim,
                 N_resamples, #10,000 in protocol --> use 1,000
                 N_patients_per_cohort, #100 pt in protocol
                 screening_multipliers, 
                 OR_evt_shift=[0], #it is likely EVT effect is lower in M2
                 miss_percentage=[0,.1,.2,.3,.4,.5],
                 WTP=80000,
                 root_sav=None,
                 seed=21):
    np.random.seed(seed)
    #function runs PSA simulations
    
    if root_sav is not None:
        if not os.path.exists(root_sav):
            os.makedirs(root_sav)
    #set all seeds!!
    #screening_multiplier: multiplied with CTP costs to 
    #represent number needed to image before a 'missed' M2 
    #is detected --> thus population costs are in the analysis
    bl_dct = df.to_dict(orient='index')

    tmp = df[(df['bl_occloc']=='M2')|(df['bl_occloc']=='M3')]
    if N_patients_per_cohort=='auto':
        nppc = len(tmp)
    else:
        nppc = N_patients_per_cohort
    res_ext, res_aggr = [],[]
    #simulate multiple EVT effects
    for evt_shift in OR_evt_shift:
        #check if result file exists already
        done = np.array(['EVT'+str(evt_shift)+'_' in f for f in os.listdir(root_sav)]).sum()
        print(evt_shift,done,len(screening_multipliers))
        if done==2*len(screening_multipliers):
            continue
        Sim.CPG = shift_ORs(Sim.CPG,shift_dct={'EVT':evt_shift})
        totals_res,extend_res = probabilistic_simulation(Sim,
                                                         tmp,#only simulate for M2
                                                         N_resamples,
                                                         nppc)

        #simulate multiple NNI thresholds
        for NNI_multiplier in screening_multipliers:
            f1 = os.path.join(root_sav,'EVT{}_NNI{}_aggregated_psa_res.pic'.format(evt_shift,NNI_multiplier))
            f2 = os.path.join(root_sav,'EVT{}_NNI{}_extended_psa_res.pic'.format(evt_shift,NNI_multiplier))
            if os.path.exists(f1):
                continue
            print('Running:',evt_shift,NNI_multiplier)
            outs, aggrs,__ = probabilistic_cohort_outcomes(totals_res, 
                                                        bl_dct,
                                                        thresholds=[2000], #nobody should be excluded
                                                        costs_per_ctp=Sim.C.costs_CTP,
                                                        multiply_ctp_costs = NNI_multiplier,
                                                        miss_percentage = miss_percentage, #also simpulate varying sensitivity gains due to CTP
                                                        WTP = WTP)

            outs['NNI_multiplier'] = NNI_multiplier
            aggrs['NNI_multiplier'] = NNI_multiplier
            outs['evt_shift'] = evt_shift
            aggrs['evt_shift'] = evt_shift

            res_ext.append(outs)
            res_aggr.append(aggrs)
            if root_sav is not None:
                res_aggr = pd.concat(res_aggr)
                res_extended = pd.concat(res_ext)
                res_aggr.to_pickle(f1)
                res_extended.to_pickle(f2)
                res_aggr, res_extended = [],[]

    if root_sav is not None:
        print('Creating final file')
        res_aggr = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_aggregated_psa_res' in f])
        #res_extended = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_extended_psa_res' in f])

        res_aggr.to_pickle(os.path.join(root_sav,'full_aggregated_psa_res.pic'))
        #res_extended.to_pickle(os.path.join(root_sav,'full_extended_psa_res.pic'))
        return res_aggr 
    else:
        res_aggr = pd.concat(res_aggr)
        res_extended = pd.concat(res_ext)
        return res_extended, res_aggr


def occl_detect_psa(df,
                 Sim,
                 N_resamples, #10,000 in protocol --> use 1,000
                 N_patients_per_cohort, #100 pt in protocol
                 screening_multipliers=[0], 
                 OR_evt_shift=[0], #it is likely EVT effect is lower in M2
                 miss_percentage=[0,.1,.2,.3,.4,.5],
                 WTP=80000,
                 root_sav=None,
                 seed=21):
    np.random.seed(seed)
    #function runs PSA simulations

    if root_sav is not None:
        if not os.path.exists(root_sav):
            os.makedirs(root_sav)
    #set all seeds!!
    #screening_multiplier: multiplied with CTP costs to 
    #represent number needed to image before a 'missed' M2 
    #is detected --> thus population costs are in the analysis
    bl_dct = df.to_dict(orient='index')

    #tmp = df[(df['bl_occloc']=='M2')|(df['bl_occloc']=='M3')]
    tmp = df
    if N_patients_per_cohort=='auto':
        nppc = len(tmp)
    else:
        nppc = N_patients_per_cohort
    res_ext, res_aggr = [],[]
    #simulate multiple EVT effects
    NNI_multiplier = 0
    for evt_shift in OR_evt_shift:
        f1 = os.path.join(root_sav,'EVT{}_NNI{}_aggregated_psa_res.pic'.format(evt_shift,NNI_multiplier))
        f2 = os.path.join(root_sav,'EVT{}_NNI{}_extended_psa_res.pic'.format(evt_shift,NNI_multiplier))
               
        #check if result file exists already
        done = np.array(['EVT'+str(evt_shift)+'_' in f for f in os.listdir(root_sav)]).sum()
        print(evt_shift,done,len(screening_multipliers))
        if done==2*len(screening_multipliers):
            continue
        Sim.CPG = shift_ORs(Sim.CPG,shift_dct={'EVT':evt_shift})
        totals_res,extend_res = probabilistic_simulation(Sim,
                                                         tmp,#only simulate for M2
                                                         N_resamples,
                                                         nppc)


        outs, aggrs,__ = probabilistic_cohort_outcomes(totals_res, 
                                                    bl_dct,
                                                    thresholds=[20000], #nobody should be excluded
                                                    costs_per_ctp=Sim.C.costs_CTP,
                                                    multiply_ctp_costs = NNI_multiplier, #add CTP costs after simulations
                                                    miss_percentage = miss_percentage, #also simpulate varying sensitivity gains due to CTP
                                                    WTP = WTP)

        outs['NNI_multiplier'] = NNI_multiplier
        aggrs['NNI_multiplier'] = NNI_multiplier
        outs['evt_shift'] = evt_shift
        aggrs['evt_shift'] = evt_shift

        res_ext.append(outs)
        res_aggr.append(aggrs)
        if root_sav is not None:
            res_aggr = pd.concat(res_aggr)
            res_extended = pd.concat(res_ext)
            res_aggr.to_pickle(f1)
            res_extended.to_pickle(f2)
            res_aggr, res_extended = [],[]

    if root_sav is not None:
        print('Creating final file')
        res_aggr = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_aggregated_psa_res' in f])
        #res_extended = pd.concat([pd.read_pickle(os.path.join(root_sav,f)) for f in os.listdir(root_sav) if '_extended_psa_res' in f])

        res_aggr.to_pickle(os.path.join(root_sav,'full_aggregated_psa_res.pic'))
        #res_extended.to_pickle(os.path.join(root_sav,'full_extended_psa_res.pic'))
        return res_aggr 
    else:
        res_aggr = pd.concat(res_aggr)
        res_extended = pd.concat(res_ext)
        return res_extended, res_aggr

def oneway_data(S,up=.1,down=.1):
    data = S._get_all_params(current=False)
    data['obj_var'] = [ '{}_{}'.format(row['object'],row['variable']) for c,row in data.iterrows()]
    data.index = data['obj_var']
    data['up'] = data['value'].copy()*(1+up)
    data['down'] = data['value'].copy()*(1-down)
    return data

def BL_sim(S,df, thresholds=[70], larger_smaller='larger', WTP=80000):
    bl_cols = ['core_vol', 'penumbra_vol', 'mm_ratio', 'bl_occloc',
           'r_age', 'bl_nihss_sum', 't_otg','r_sex','ivt_given', 
            'bl_collaterals','bl_hist_premrs','iat_post_etici',
            'age_groups','otg_groups']
    BL = df[bl_cols]
    
    totals_res, extend_res = simulate_IDs(df.index,df,S)
    #add clinical basline info for plots
    totals_res = pd.concat([totals_res,BL],axis=1)
    #aggr contains average results in NMB, ICER, d_costs,d_qalys
    # of the cohort per different decision threshold
    __,aggr = cohort_outcome(totals_res,
                            thresholds=thresholds, 
                            larger_smaller=larger_smaller,
                            costs_per_ctp=0, #used for miss rate sim
                            multiply_ctp_costs = 0,#used for miss rate sim -->perform only on M2
                            miss_percentage = [0],#used for miss rate sim
                            WTP=WTP)
    return aggr


def oneway_sensitivity(S,data,df,larger_smaller='larger', thresholds=[70], root_sav=None):
    #S: simulation object
    #data: contains columns up and down 
    # used as range for oneway sensitivity analysis
    #df: dataframe with patients
    bl = BL_sim(S,df,thresholds=thresholds,larger_smaller=larger_smaller)
    #add lower upper for dataframe consistency
    lower, upper = bl.copy(), bl.copy()
    lower.columns = [c+'_lower' for c in lower]
    upper.columns = [c+'_upper' for c in upper]
    row = pd.concat([lower,upper],axis=1)
    row['variable'] = 'baseline'

    out = [row]
    for ix,row in tqdm(data.iterrows()):
        tmpdata = data.copy()
        #run lower sim
        tmpdata.at[ix,'value'] = row['down']
        S._set_all_params(tmpdata)
        lower = BL_sim(S,df, thresholds=thresholds, larger_smaller=larger_smaller)
        #run upper sim
        tmpdata.at[ix,'value'] = row['up']
        S._set_all_params(tmpdata)
        upper = BL_sim(S,df, thresholds=thresholds, larger_smaller=larger_smaller)

        lower.columns = [c+'_lower' for c in lower]
        upper.columns = [c+'_upper' for c in upper]
        row = pd.concat([lower,upper],axis=1)
        row['variable'] = ix

        out.append(row)
    out = pd.concat(out)
    
    if root_sav is not None:
        if not os.path.exists(root_sav):
            os.makedirs(root_sav)
        out.to_excel(os.path.join(root_sav,'oneway_sensitivity.xlsx'))
    return out

def BL_sim_occlusion_detect(S,
                            df,
                            sens_dct={'ICA':.08,'M1':.16,'M2':.16},
                            NNI=8.3,
                            WTP=80000, 
                            use_median=False):

    bl_cols = ['core_vol', 'penumbra_vol', 'mm_ratio', 'bl_occloc',
           'r_age', 'bl_nihss_sum', 't_otg','r_sex','ivt_given', 
            'bl_collaterals','bl_hist_premrs','iat_post_etici',
            'age_groups','otg_groups']

    #run sim for each separate occlusion location
    tmp_df = df[(df['bl_occloc']=='ICA')|(df['bl_occloc']=='ICA-T')]
    res_ICA, __ = simulate_IDs(tmp_df.index,df,S)
    res_ICA['occloc'] = 'ICA'
    res_ICA['sens_gain'] = sens_dct['ICA']

    tmp_df = df[(df['bl_occloc']=='M1')]
    res_M1, __ = simulate_IDs(tmp_df.index,df,S)
    res_M1['occloc'] = 'M1'
    res_M1['sens_gain'] = sens_dct['M1']

    tmp_df = df[(df['bl_occloc']=='M2')]
    res_M2, __ = simulate_IDs(tmp_df.index,df,S)
    res_M2['occloc'] = 'M2'
    res_M2['sens_gain'] = sens_dct['M2']

    res_tot = pd.concat([res_ICA, res_M1, res_M2])
    res_tot['NNI'] = NNI

    __, aggr = cohort_outcome_occlusion_detection(res_tot.copy(), 
                                           costs_per_ctp=S.C.costs_CTP, 
                                           WTP = 80000, use_median=use_median)

    return aggr

def oneway_sensitivity_occlusion_detect(S,data,df, root_sav=None,name='', use_median=False, NNI=8.3):
    #S: simulation object
    #data: contains columns up and down 
    # used as range for oneway sensitivity analysis
    #df: dataframe with patients
    bl = BL_sim_occlusion_detect(S,df, use_median=use_median, NNI=NNI)
    #add lower upper for dataframe consistency
    lower, upper = bl.copy(), bl.copy()
    lower.columns = [c+'_lower' for c in lower]
    upper.columns = [c+'_upper' for c in upper]
    row = pd.concat([lower,upper],axis=1)
    row['variable'] = 'baseline'

    out = [row]
    for ix,row in tqdm(data.iterrows()):
        tmpdata = data.copy()
        #run lower sim
        tmpdata.at[ix,'value'] = row['down']
        S._set_all_params(tmpdata)
        lower = BL_sim_occlusion_detect(S,df, use_median=use_median, NNI=NNI)
        #run upper sim
        tmpdata.at[ix,'value'] = row['up']
        S._set_all_params(tmpdata)
        upper = BL_sim_occlusion_detect(S,df, use_median=use_median, NNI=NNI)

        lower.columns = [c+'_lower' for c in lower]
        upper.columns = [c+'_upper' for c in upper]
        row = pd.concat([lower,upper],axis=1)
        row['variable'] = ix

        out.append(row)
    out = pd.concat(out)
    
    if root_sav is not None:
        if not os.path.exists(root_sav):
            os.makedirs(root_sav)
        out.to_excel(os.path.join(root_sav,'oneway_sensitivity{}.xlsx'.format(name)))
    return out
