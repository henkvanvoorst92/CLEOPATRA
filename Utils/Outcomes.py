import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def probabilistic_cohort_outcomes(df_psa, 
                                bl_dct,
                               thresholds=np.arange(0,151,10),
                               costs_per_ctp=0,
                               multiply_ctp_costs = 0,
                               miss_percentage = [0],
                               WTP = 80000):
    
    #cohort_outcome_per_threshold for PSA simulations
    #df_res: per ID per simulation, 
    # costs and qalys (columns) for each strategy (in rows) 
    #bl-dct: should contain key=ID value=core_vol
    #WTP is used for NMB calculations
    #computes the outcomes per core_volume threshold per cohort average
    outs, aggrs, fullres = [],[],[]
    for simno in df_psa['simno'].unique():
        df_res = df_psa[df_psa['simno']==simno]
        df_res['core_vol'] = [bl_dct[ID]['core_vol'] for ID in df_res.index]
        df_res = df_res.reset_index(drop=True)
        #compute cohort results separately
        out,aggr = cohort_outcome(df_res,
                                thresholds=thresholds, 
                                costs_per_ctp=costs_per_ctp,
                                multiply_ctp_costs=multiply_ctp_costs,
                                miss_percentage=miss_percentage,
                                WTP=WTP)
        aggr['simno'] = simno
        outs.append(out)
        aggrs.append(aggr)
        fullres.append(df_res)

        
    outs = pd.concat(outs)
    aggrs = pd.concat(aggrs)
    aggrs = aggrs.reset_index(drop=True)
    fullres = pd.concat(fullres)
    return outs, aggrs, fullres

def cohort_outcome(df_res, 
                   thresholds=np.arange(0,151,10),
                   costs_per_ctp=0, 
                   multiply_ctp_costs=0, #NNI multiplier
                   miss_percentage=[0],
                   WTP = 80000):
     #converts totals_res to depict results per core_volume decision threshold
     #also can be used to simulate miss_percentage of M2s
     # aggr considers average values, out results per patient in the cohort 
     # results for thresholds in column

    #df_res = df_res[df_res['bl_occloc']=='M2']

    # results for thresholds in column
    out, aggrs = [],[]
    for mp in miss_percentage:
        #missed M2 percentage modelling
        add_ctp_costs = (multiply_ctp_costs-1)*costs_per_ctp
        miss_vec = np.array([1-mp,1-mp,mp,mp])
        for thr in thresholds:
            ##control arm without CTP --> all have EVT
            # optional --> M2s are missed thus noEVT in percentage of pt
            noCTP = df_res[['ID','C_noctp_evt','Q_noctp_evt',
                          'C_noctp_noevt','Q_noctp_noevt']].drop(columns='ID')
            #this steps simulates that a % of patients are missed
            noCTP = noCTP*miss_vec
            noCTP['C_noctp'] = noCTP['C_noctp_evt']+noCTP['C_noctp_noevt']
            noCTP['Q_noctp'] = noCTP['Q_noctp_evt']+noCTP['Q_noctp_noevt']

            ###control arm with CTP
            noEVT = df_res[df_res['core_vol']>=thr][['ID','C_ctp_noevt','Q_ctp_noevt']]
            noEVT.columns = ['ID','C_ctp', 'Q_ctp']
            noEVT['altered_decision'] = True
            EVT = df_res[df_res['core_vol']<thr][['ID','C_ctp_evt','Q_ctp_evt']]
            EVT.columns = ['ID','C_ctp', 'Q_ctp']
            EVT['altered_decision']=False
            #combine noEVT and EVT with CTP vertically (every patient only once)
            CTP = pd.concat([noEVT,EVT])
            #combine CTP and noCTP arms horizontally
            outcome = pd.concat([CTP,noCTP],axis=1,join='outer')
            outcome.index = outcome['ID']
            outcome['C_ctp'] = outcome['C_ctp']+add_ctp_costs
            outcome['d_costs'] = outcome['C_ctp']-outcome['C_noctp']
            outcome['d_qalys'] = outcome['Q_ctp']-outcome['Q_noctp']
            outcome['NMB'] = outcome['d_qalys']*WTP+outcome['d_costs']*-1
            outcome['missed_M2'] = mp
            aggr = outcome[[c for c in outcome if 'ID' not in c]].sum()/len(outcome)
            aggr['ICER'] = aggr['d_costs']/aggr['d_qalys']
            aggr['n_above_threshold'] = len(noEVT)
            aggr['threshold'] = thr
            aggr['missed_M2'] = mp

            outcome.columns = [c+'_'+str(thr) for c in outcome.columns]
            out.append(outcome)
            aggrs.append(aggr)
    
    out = pd.concat(out,axis=1)
    aggr = pd.concat(aggrs,axis=1).T
    return out, aggr

def median_iqr_results(df_median,df_p25,df_p75, threshold, colname='', qaly_multiplier=1):
    variables = df_median.columns
    
    n_digits = 7
    if qaly_multiplier>100:
        n_digits = 2
    
    combined = []
    for v in variables:
        if 'c_' in v.lower() or 'q_' in v.lower() or \
        'NMB' in v or 'ICER' in v or \
        'd_costs' in v.lower() or 'd_qalys' in v.lower():
            m = df_median.loc[threshold].loc[v]
            p25 = df_p25.loc[threshold].loc[v]
            p75 = df_p75.loc[threshold].loc[v]
            if 'q_' in v.lower() or 'd_qalys' in v.lower():
                value = '{}({};{})'.format(round(m*qaly_multiplier,n_digits),
                                           round(p25*qaly_multiplier,n_digits),
                                           round(p75*qaly_multiplier,n_digits))
                combined.append(value)
            else:
                value = '{}({};{})'.format(m.astype(int),p25.astype(int),p75.astype(int))
                combined.append(value)
        else:
            value = df_median.loc[threshold].loc[v]
            combined.append(value)
        #print(v,value)
    #return pd.DataFrame(data=combined,index=variables,columns=[colname])
    return combined

def OR_shift_results(aggr_res,
                     OR_evt_shifts = [-.3,-.2,-.1,0,.1,.2,.3,.82],
                     OR_evt_corevol_shifts = [-.1,-.05,0], 
                     savloc=None):
    #computes results for pivot table of OR shift results
    out = []
    res_grouper = aggr_res.groupby(['OR_evt_shift', 'OR_evt_corevol_shift','threshold'])
    for evt_shift in OR_evt_shifts:
        for evt_corevol_shift in OR_evt_corevol_shifts:
            tmp_median = res_grouper.median().reset_index(drop=False)
            tmp_median = tmp_median[(tmp_median['OR_evt_shift']==evt_shift)&(tmp_median['OR_evt_corevol_shift']==evt_corevol_shift)]
            tmp_median.index = tmp_median['threshold']

            tmp_p25 = res_grouper.quantile(0.25).reset_index(drop=False)
            tmp_p25 = tmp_p25[(tmp_p25['OR_evt_shift']==evt_shift)&(tmp_p25['OR_evt_corevol_shift']==evt_corevol_shift)]
            tmp_p25.index = tmp_p25['threshold']
            
            tmp_p75 = res_grouper.quantile(0.75).reset_index(drop=False)
            tmp_p75 = tmp_p75[(tmp_p75['OR_evt_shift']==evt_shift)&(tmp_p75['OR_evt_corevol_shift']==evt_corevol_shift)]
            tmp_p75.index = tmp_p75['threshold']
            #use optimal NMB to identify optimal threshold
            max_NMB_threshold = tmp_median['NMB'].idxmax()      

            row = median_iqr_results(tmp_median,tmp_p25,tmp_p75, max_NMB_threshold, qaly_multiplier=365)
            out.append(row)

    out = pd.DataFrame(out,columns = [*list(tmp_median.columns)]) 

    if savloc is not None:
        if not os.path.exists(savloc):
            os.makedirs(savloc)
        
        out.to_excel(os.path.join(savloc,'OR_shift.xlsx'))

    __ = pivot_results(out,
                        xvar='OR_evt_shift', 
                        yvar='OR_evt_corevol_shift',
                        outcomes=['threshold','NMB','ICER','d_qalys','d_costs'],
                        name='OR_shift_outcomes',
                        savloc=savloc)
    
    return out

def pivot_results(data,xvar,yvar,outcomes,name='',savloc=None):
    #outcomes: ['threshold','NMB','ICER','d_qalys','d_costs']
    if savloc is not None:
        if not os.path.exists(savloc):
            os.makedirs(savloc)
    pivot_out = []
    for outcome in outcomes:
        tmp = data.pivot(index=xvar, 
                        columns=yvar, 
                        values=outcome)
        tmp['outcome'] = outcome
        pivot_out.append(tmp)
    pivot_out = pd.concat(pivot_out)
    if savloc is not None:
        pivot_out.to_excel(os.path.join(savloc,'{}.xlsx'.format(name)))
    return pivot_out


