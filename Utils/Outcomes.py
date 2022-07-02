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
    outs, aggrs = [],[]
    for simno in df_psa['simno'].unique():
        df_res = totals_res[df_psa['simno']==simno]
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
        
    outs = pd.concat(outs)
    aggrs = pd.concat(aggrs)
    aggrs = aggrs.reset_index(drop=True)
    return outs, aggrs

def cohort_outcome(df_res, 
                   thresholds=np.arange(0,151,10),
                   costs_per_ctp=0,
                   multiply_ctp_costs = 0,
                   miss_percentage = [0],
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
        miss_vec = np.array([1-mp,1-mp,
                             mp,mp])
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
            outcome['d_costs'] = outcome['C_ctp']-outcome['C_noctp']
            outcome['d_qalys'] = outcome['Q_ctp']-outcome['Q_noctp']
            outcome['NMB'] = outcome['d_qalys']*WTP+outcome['d_costs']*-1
            outcome['missed_M2'] = mp
            aggr = outcome[[c for c in outcome if 'ID' not in c]].sum()/len(outcome)
            aggr['ICER'] = aggr['d_costs']/aggr['d_qalys']
            aggr['threshold'] = thr
            aggr['missed_M2'] = mp

            outcome.columns = [c+'_'+str(thr) for c in outcome.columns]
            out.append(outcome)
            aggrs.append(aggr)
    
    out = pd.concat(out,axis=1)
    aggr = pd.concat(aggrs,axis=1).T
    return out, aggr


def ICER_plot(aggrs,name, min_x=-.15,max_x =.15):
    d1 = aggrs[(aggrs.threshold>=50)&(aggrs.threshold<70)]
    d1['group'] = '>50mL'
    d2 = aggrs[(aggrs.threshold>=70)&(aggrs.threshold<100)]
    d2['group'] = '>70mL'
    d3 = aggrs[aggrs.threshold>=100]
    d3['group'] = '>100mL'
    dataplot = pd.concat([d1, d2, d3])
    dataplot.group.unique()
    
    x = np.linspace(min_x, max_x)
    y_b = np.arange(min_y, max_y,50)
    y = 80000*x
    sns.lineplot(x,np.zeros_like(x),color='black',linewidth=1)
    sns.lineplot(np.zeros_like(y_b),y_b,color='black',linewidth=1,estimator=None)
    sns.scatterplot(data=dataplot,x='d_qalys',y='d_costs', hue='group')
    sns.lineplot(x,y,color='black')
    plt.ylim(min_y, max_y)
    plt.xlim(min_x, max_x)
    plt.ylabel('Difference Costs(€) (CTP-noCTP)')
    plt.xlabel('Difference QALYS (CTP-noCTP)')
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join(root_fig,'ICER_{}.png'.format(name)))
    plt.show()

    
folder = 'EVT_effect'
d = os.path.join(root_sav,'PSA',folder)
CEA_res = pd.read_pickle(os.path.join(d,'probabilistic_ICER_results.pic'))
CEA_res.subgroup = [s[:15 ]for s in CEA_res.subgroup]
for sg in CEA_res.subgroup.unique():
    tmp = CEA_res[CEA_res==sg]
    plot_res(aggrs,sg)
