import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def probabilistic_cohort_outcomes(df_psa, 
                                bl_dct,
                               thresholds=np.arange(0,151,10),
                               larger_smaller='larger',
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
                                larger_smaller=larger_smaller,
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
                   larger_smaller='larger',
                   costs_per_ctp=0, 
                   multiply_ctp_costs=0, #NNI multiplier
                   miss_percentage=[0], #used for occlusion detection
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

        add_ctp_costs = max((multiply_ctp_costs-1)*costs_per_ctp,0) #if negative returns zero

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
            #threshold larger or smaller than threshold
            if larger_smaller=='larger':#larger than threshold means no EVT
                noEVT = df_res[df_res['core_vol']>=thr][['ID','C_ctp_noevt','Q_ctp_noevt']]
                EVT = df_res[df_res['core_vol']<thr][['ID','C_ctp_evt','Q_ctp_evt']]
            elif larger_smaller=='smaller': #smaller than threshold means no EVT
                noEVT = df_res[df_res['core_vol']<=thr][['ID','C_ctp_noevt','Q_ctp_noevt']]
                EVT = df_res[df_res['core_vol']>thr][['ID','C_ctp_evt','Q_ctp_evt']]

            noEVT.columns = ['ID','C_ctp', 'Q_ctp']
            noEVT['altered_decision'] = True
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
            aggr['sens_gain'] = mp

            outcome.columns = [c+'_'+str(thr) for c in outcome.columns]
            out.append(outcome)
            aggrs.append(aggr)
    
    out = pd.concat(out,axis=1)
    aggr = pd.concat(aggrs,axis=1).T
    return out, aggr

def cohort_outcome_occlusion_detection(df_res, 
                                       costs_per_ctp=0, 
                                       WTP = 80000, use_median=False):
    mp = df_res['sens_gain'].values
    miss_mat = np.array([1-mp,1-mp,mp,mp]).T

    #control arm without CTP --> a percentage is missed
    noCTP = df_res[['ID','C_noctp_evt','Q_noctp_evt',
                  'C_noctp_noevt','Q_noctp_noevt']].copy().drop(columns='ID')
    #this steps simulates that a % of patients are missed
    noCTP = noCTP*miss_mat
    noCTP['C_noctp'] = noCTP['C_noctp_evt']+noCTP['C_noctp_noevt']
    noCTP['Q_noctp'] = noCTP['Q_noctp_evt']+noCTP['Q_noctp_noevt']

    CTP = df_res[['ID','C_ctp_evt','Q_ctp_evt']].copy().drop(columns='ID')
    CTP.columns = ['C_ctp', 'Q_ctp']
    #add ctp costs
    CTP['C_ctp'] += np.clip((df_res['NNI']-1)*costs_per_ctp,0,1e9)

    outcome = pd.DataFrame()
    outcome[['C_ctp','Q_ctp']] = CTP[['C_ctp', 'Q_ctp']] 
    outcome[['C_noctp','Q_noctp']] = noCTP[['C_noctp', 'Q_noctp']] 
    outcome['d_costs'] = CTP['C_ctp']-noCTP['C_noctp']
    outcome['d_qalys'] = CTP['Q_ctp']-noCTP['Q_noctp']
    outcome['NMB'] = outcome['d_qalys']*WTP+outcome['d_costs']*-1
    
    if use_median:
        aggr = outcome.median()
    else:
        aggr = outcome.mean()
    aggr['ICER'] = aggr['d_costs']/aggr['d_qalys']
    aggr['NNI'] = df_res['NNI'].values[0]
    aggr['sens_gain'] = list(np.unique(mp))

    return outcome, pd.DataFrame(aggr).T

def median_iqr_results(df_median,df_p25,df_p75, threshold, colname='', qaly_multiplier=1, verbal=False):
    variables = df_median.columns
    
    combined = []
    for v in variables:
        if 'c_' in v.lower() or 'q_' in v.lower() or \
        'NMB' in v or 'ICER' in v or \
        'd_costs' in v.lower() or 'd_qalys' in v.lower():
            m = df_median.loc[threshold].loc[v]
            p25 = df_p25.loc[threshold].loc[v]
            p75 = df_p75.loc[threshold].loc[v]
            if 'q_' in v.lower() or 'd_qalys' in v.lower():
                n_digits = 3
                if qaly_multiplier>100:
                    n_digits = 2
                value = '{} ({};{})'.format(round(m*qaly_multiplier,n_digits),
                                           round(p25*qaly_multiplier,n_digits),
                                           round(p75*qaly_multiplier,n_digits))
                combined.append(value)
            else:
                m, p25,p75 = round(m,2),round(p25,2),round(p75,2)
                value = '{} ({};{})'.format(m.astype(int),p25.astype(int),p75.astype(int))
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


def combine_occloc_results(loc, #path to stored results )(f1)
                           frac_dct, #proportion of each occloc in database (for averaging) dct[occloc]  = fraction
                           costs_CTP, #costs of each CTP
                           fu_years = [5], #fu years
                           NNI_multipliers=[10], #NNI multiplier to add CTP screening costs
                           OR_evt_shift=[-.3,0,.3,.82], #treament effect ORs for EVT considered (in filename f1)
                           WTP = 80000):
    
    OUT1, OUT2 = [], []
    for year in fu_years:
        for NNI in NNI_multipliers:
            for evt_shift in OR_evt_shift:

                occloc  = 'ICA'
                root_sav = os.path.join(loc,occloc,str(year)+'y')
                f1 = os.path.join(root_sav,'EVT{}_NNI{}_aggregated_psa_res.pic'.format(evt_shift,0))
                res_ICA = pd.read_pickle(f1)
                bl = (res_ICA['missed_M2']-.08).astype(np.float32).round(2)
                res_ICA['sens_gain_BL'] = bl #np.where((bl<-.04)|(bl>.04),np.full_like(bl,np.NaN),bl)
                #print('ICA',res_ICA['sens_gain_BL'].unique())

                occloc  = 'M1'
                root_sav = os.path.join(loc,occloc,str(year)+'y')
                f1 = os.path.join(root_sav,'EVT{}_NNI{}_aggregated_psa_res.pic'.format(evt_shift,0))
                res_M1 = pd.read_pickle(f1)
                bl = (res_M1['missed_M2']-.16).astype(np.float32).round(2)
                res_M1['sens_gain_BL'] = bl #np.where((bl<-.04)|(bl>.04),np.full_like(bl,np.NaN),bl)
                #print('M1',res_M1['sens_gain_BL'].unique())

                occloc  = 'M2'
                root_sav = os.path.join(loc,occloc,str(year)+'y')
                f1 = os.path.join(root_sav,'EVT{}_NNI{}_aggregated_psa_res.pic'.format(evt_shift,0))
                res_M2 = pd.read_pickle(f1)
                bl = (res_M2['missed_M2']-.16).astype(np.float32).round(2)
                res_M2['sens_gain_BL'] = bl #np.where((bl<-.04)|(bl>.04),np.full_like(bl,np.NaN),bl)
                #print('M2',res_M2['sens_gain_BL'].unique())

                out1, out2 = [], []
                for data,occloc in [(res_ICA,'ICA'), (res_M1,'M1'), (res_M2,'M2')]:
                    data['missed_M2'] = data['missed_M2'].astype(np.float32).round(2)
                    data = data.sort_values(by=['missed_M2','simno'])

                    #separate for each location attach NNI screening costs
                    combined_data = data[~data['sens_gain_BL'].isna()]
                    combined_data = combined_data.set_index(['simno','sens_gain_BL'])
                    combined_data = combined_data[['d_costs','d_qalys','NMB']]
                    combined_data['d_costs_NNI'] = combined_data['d_costs']+costs_CTP*NNI*frac_dct[occloc]
                    combined_data['NMB_NNI'] = WTP*combined_data['d_qalys']+combined_data['d_costs_NNI']*-1
                    combined_data.columns = [c+'_'+occloc for c in combined_data.columns]
                    out1.append(combined_data)

                    data = data.set_index(['simno','missed_M2'])
                    data = data[['d_costs','d_qalys','NMB']]
                    data['d_costs_NNI'] = data['d_costs']+costs_CTP*NNI*frac_dct[occloc]
                    data.columns = [c+'_'+occloc for c in data.columns]
                    out2.append(data)

                #create a total res file taking the weight of each location
                cd = pd.concat(out1,axis=1)
                cd['d_qalys_tot'] = cd['d_qalys_ICA']*frac_dct['ICA']+\
                                    cd['d_qalys_M1']*frac_dct['M1']+\
                                    cd['d_qalys_M2']*frac_dct['M2']

                cd['d_costs_tot'] = cd['d_costs_ICA']*frac_dct['ICA']+\
                                    cd['d_costs_M1']*frac_dct['M1']+\
                                    cd['d_costs_M2']*frac_dct['M2']+\
                                    costs_CTP*NNI
                cd['NMB_tot'] = WTP*cd['d_qalys_tot']+cd['d_costs_tot']*-1

                #for figure making
                cd['NMB_NNI_tot'] = cd['NMB_tot']
                cd['d_costs_NNI_tot'] = cd['d_costs_tot']

                cd['NNI']  = NNI
                cd['fu_years'] = year
                cd['evt_shift'] = evt_shift 

                data = pd.concat(out2,axis=1)
                data['NNI']  = NNI
                data['fu_years'] = year
                data['evt_shift'] = evt_shift

                OUT1.append(cd)
                OUT2.append(data)

    OUT1 = pd.concat(OUT1,axis=0).reset_index()
    OUT2 = pd.concat(OUT2,axis=0).reset_index()
    
    OUT1.to_pickle(os.path.join(loc,'combined_detection_res.pic'))
    OUT2.to_pickle(os.path.join(loc,'separate_detection_res.pic'))

    return OUT1, OUT2
