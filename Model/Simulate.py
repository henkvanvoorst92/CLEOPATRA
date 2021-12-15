import numpy as np
import math
import pandas as pd
import time
import re
import os

def compute_shift_no_evt(pt_dct, ORs):
	"""
	pt_dct: dict with keys are also keys in ORs 
			values are used to compute absolute shift
	ORs: dict with keys of Tx values are Odds Ratios for favorable mRS
	
	return one hot encoded probabilities of mRS for given patient
	"""
	p_shift = 0
	for name,OR in ORs.items():
		if name in pt_dct.keys():
			p_shift -= pt_dct[name]*np.log(OR)
	mRS = np.zeros(7)
	#compute one hot embedding
	fl_new_mrs = pt_dct['mRS']+p_shift
	new_mrs = int(np.floor(fl_new_mrs))
	p_new_mrs = 1-(fl_new_mrs-new_mrs)
	mRS[new_mrs] = p_new_mrs 
	mRS[new_mrs+1] = 1-p_new_mrs 
	return mRS

def get_mRS_EVT_noEVT(df,ORs):
	"""  
	df is a pandas.DataFrame existing of patients with EVT:
		index=ID
		columns: mRS and keys in ORs
	ORs is a dictionary with Odds Ratios for treatment effect (OR for favorable outcome)
	
	returns: one hot encoded mRS of patient if EVT was given or not (noEVT)
	"""
	dct = df.to_dict(orient='index')
	out = []
	for pt_dct in dct.values():
		mrs = compute_shift_no_evt(pt_dct, ORs)
		out.append(mrs.tolist())
	noEVT = pd.DataFrame(out, columns=[*['noEVT_mRS_'+str(i) for i in range(7)]], index=dct.keys())
	EVT = pd.get_dummies(df['mRS'], prefix='EVT_mRS')
	df_out = pd.concat([EVT,noEVT], ignore_index=False, axis=1)
	return df_out

def mRS_90d_2arms(input_dist,ID_treat_select, return_relative=False):
	evt_cols = [c for c in input_dist.columns if not 'noevt' in c.lower()]
	noevt_cols = [c for c in input_dist.columns if 'noevt' in c.lower()]

	select_EVT = input_dist.loc[ID_treat_select][evt_cols]
	exclude_EVT = input_dist[~np.isin(input_dist.index,ID_treat_select)][noevt_cols]

	p_evt_90d_mrs = select_EVT.sum(axis=0)
	p_noevt_90d_mrs = exclude_EVT.sum(axis=0)
	if return_relative:
		p_evt_90d_mrs = p_evt_90d_mrs /p_evt_90d_mrs.sum()
		p_noevt_90d_mrs = p_noevt_90d_mrs/p_noevt_90d_mrs.sum()
		
	return p_evt_90d_mrs, p_noevt_90d_mrs, select_EVT, exclude_EVT


# simulate long term follow-up
class SimulateLongTerm(object):
    """
    Inputs:
        M: class initialized for mortality simulation
        RS: class initialized for recurrent stroke simulation
        C: class initialized to compute costs
        Q: class initialized to compute QALYs
        
    Returns:
        Result over the simulated follow up period as 
        np.array with columns: [orgID,uniqueID,year,
                                mRS01,mRS2,mRS3,mRS4,mRS5,mRS6,
                                costs,QALYs]
    """
    
    def __init__(self, 
                 years_to_simulate,
                 M,RS,C,Q,
                 start_year = 2022
                ):
        self.M = M
        self.RS = RS
        self.C = C
        self.Q = Q
        
        self.start_year=start_year
        self.years_to_simulate = years_to_simulate
    
    def _probabilistic_resample(self):
        # resample all parameters for probabilistic sensitivity analyses (PSA)
        self.M._probabilistic_resample()
        self.RS._probabilistic_resample()
        self.C._probabilistic_resample()
        self.Q._probabilistic_resample()
    
    def __call__(self,gender,age,mrs_start, IDs=[None]):
        cur_mrs = mrs_start.copy()
        cur_age = age
        cur_year = self.start_year
        out = []
        for yearno in range(1,self.years_to_simulate+1):
            cur_age+=1
            cur_year+=1
            # simulate mortality
            cur_mrs, p_mort = self.M(gender,cur_year,cur_age,cur_mrs)
            # simulate stroke recurrence
            cur_mrs, p_restroke = self.RS(yearno,cur_age,cur_mrs)
            #compute costs and qalys
            costs = self.C(cur_mrs,yearno)
            qalys = self.Q(cur_mrs,yearno)
            #write everything to output format
            row = [*IDs,yearno,*list(cur_mrs),*list(costs),*list(qalys)]
            out.append(row)
        return out

