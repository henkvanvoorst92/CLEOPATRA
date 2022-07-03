import numpy as np
import math
import pandas as pd
import time
import re
import os
from Utils.utils import sum_mrs01



#short term model: 90day outcome for EVT and noEVT arms
class ControlPatientGeneration(object):
    """
    Generates outcome of controle arm
    initiated with a dict containing the OR per unit change
    of patient variable
    requires a pt_dct with the value of these variables
    and the observed mrs to alter the outcome
    """
    
    def __init__(self, 
                 p_OR, #excel file with OR and dist for EVT and EVT*core_vol
                 p_nums_mrs, #excel file with mrs column and distribution for EVT/noEVT arms 
                 dataset='mrclean', #alternative is hermes data (not reccomended)
                 verbal=False
                 ):
        
        self.verbal = verbal  
        
        #initialize dictionary for OR extraction
        self.p_OR = p_OR
        df = pd.read_excel(self.p_OR)
        #_set_params enables alteration of OR values --> uses self.p_OR
        self._set_params(df)
        
        self.nums_mrs = pd.read_excel(p_nums_mrs)
        self.dataset = dataset
        #total inclusions for each arm
        self.n_tot_evt = self.nums_mrs.sum()['evt_'+self.dataset]
        #self.n_tot_noevt = nums_mrs.sum()['noevt_'+self.dataset] #not required
              
    def _combined_OR(self,pt_dct):
        #step 1 of the supplementary methods
        #combine all OR in a single OR
        #by summing over each ln_OR activation
        summed_ln_OR = 0
        for variable,value in pt_dct.items():
            if variable not in self.dct_OR.keys():
                continue
            ln_OR = np.log(self.dct_OR[variable])
            summed_ln_OR += ln_OR*value
        return np.exp(summed_ln_OR) #return an OR
    
    def _init_odds_good_mrs_evt(self, mrs):
        #step 2 of the supplementary methods
        #computes the probability of good mrs given evt
        n_goodmrs_evt = self.nums_mrs.loc[:mrs].sum()['evt_'+self.dataset]
        p_good_mrs_evt = n_goodmrs_evt/self.n_tot_evt
        odds_goodmrs_evt = p_good_mrs_evt/(1-p_good_mrs_evt)
        if self.verbal:
            print('p(goodmRS|EVT), n(goodmRS|EVT)',p_good_mrs_evt, n_goodmrs_evt)
            print('Odds(goodmrs|EVT)',odds_goodmrs_evt)
        return odds_goodmrs_evt
    
    def _new_p_good_mrs_noevt(self,OR_new,odds_goodmrs_evt):
        #step 4 of the supplementary methods
        # from the odds(goodmrs|evt) and the OR_new
        # compute p(goodmrs|noevt) for simulating the control arm
        odds_goodmrs_control = OR_new*odds_goodmrs_evt
        p_good_mrs_control = odds_goodmrs_control/(1+odds_goodmrs_control)
        if self.verbal:
            print('Odds(goodmRS|control)',odds_goodmrs_control)
            print('p(goodmRS|control)',p_good_mrs_control)
        return p_good_mrs_control
    
    def _init_OR(self,mode='default'):
        #initializes a dictionary of OR to use
        self.dct_OR = {}
        for k,v in self.OR.items():
            if  mode=='default':
                self.dct_OR[k] = v['OR']
            elif mode=='probabilistic':
                oddsratio,ci_low,ci_high = v['OR'],v['CI_low'],v['CI_high']
                self.dct_OR[k] = np.random.lognormal(mean=np.log(oddsratio), 
                                    sigma=abs(np.log(ci_low)-np.log(ci_high))/3.92)
            
            if self.verbal:
                print(k,v['OR'])

    def _mrs_distribution(self,p_good_mrs_noevt,mrs, mrs01=True):

        mrs_noevt = np.zeros(7)
        mrs_noevt[mrs] = p_good_mrs_noevt
        mrs_noevt[min(mrs+1,6)] = 1-p_good_mrs_noevt #use min as mrs cannot be>6
        mrs_evt = np.zeros(7)
        mrs_evt[mrs] = 1
        #print(mrs_evt,mrs_noevt)
        if sum_mrs01:
            mrs_evt = sum_mrs01(mrs_evt)
            mrs_noevt = sum_mrs01(mrs_noevt)

        return mrs_noevt, mrs_evt

    def _probabilistic_resample(self):
        #self._init_ppy(probabilistic=True) # NO RESAMPLING of p_restroke
        self._init_OR(mode='probabilistic')

    def _get_params(self,current=True):
        #if current returns the current variables
        #else the original df is loaded --> used for PSA/OneWay
        if not current:
            df = pd.read_excel(self.p_OR)
            self._set_params(df)
        #returns values used for simulations
        out = pd.DataFrame(self.OR).T #converts back to input df
        out['object'] = 'CPG'
        return out

    def _set_params(self,df):
        #sets values used for simulation
        df.index = df.varname
        self.OR = df.to_dict(orient='index')
        self._init_OR(mode='default') #initialize base on self.OR 

    def __call__(self,pt_dct):
        #pt_dct represents 
        # pt_dct has key= 'mrs' or 'noEVT'/'noEVT*core_volume'
        # and values based on the measurements

        mrs = pt_dct['mrs']
        if mrs<6:
        
            OR_new = self._combined_OR(pt_dct)
            if self.verbal:
                print('OR new',OR_new)
            odds_goodmrs_evt = self._init_odds_good_mrs_evt(mrs)
            p_good_mrs_noevt = self._new_p_good_mrs_noevt(OR_new,odds_goodmrs_evt)
            #pm use a gamma distribution with p_good_mrs_control 
            #as pdf to distribute across poorer mRS values
            mrs_noevt, mrs_evt = self._mrs_distribution(p_good_mrs_noevt,mrs, mrs01=True)
        else:
            mrs_noevt = np.array([0,0,0,0,0,1])
            mrs_evt = np.array([0,0,0,0,0,1])

        #generate new mRS distributions

        return mrs_evt, mrs_noevt

class Simulate(object):
    """
    Inputs:
        CPG: class initialize for control arm generation
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
                 CPG,M,RS,C,Q,
                 years_to_simulate=10,
                 start_year = 2023,
                 mrs01=True, #takes mrs 0 and 1 together
                 verbal=False
                ):
        #if verbal=True prints stuff for debugging
        self.verbal = verbal

        self.CPG = CPG
        self.M = M
        self.RS = RS
        self.C = C
        self.Q = Q
        self.mrs01 = mrs01
        
        self.start_year=start_year
        self.years_to_simulate = years_to_simulate

        if self.verbal:
            self.CPG.verbal = self.verbal
            self.M.verbal = self.verbal
            self.RS.verbal = self.verbal
            self.C.verbal = self.verbal
            self.Q.verbal = self.verbal

    
    def _probabilistic_resample(self):
        # resample all parameters for probabilistic sensitivity analyses (PSA)
        self.CPG._probabilistic_resample()
        self.M._probabilistic_resample()
        self.RS._probabilistic_resample()
        self.C._probabilistic_resample()
        self.Q._probabilistic_resample()
    
    def _90d_both_arms(self,pt_dct):
        ID = pt_dct['ID']
        cur_mrs_evt, cur_mrs_noevt = self.CPG(pt_dct)
        
        if self.mrs01 and len(cur_mrs_evt)>6:
            cur_mrs_evt = sum_mrs01(cur_mrs_evt)
            cur_mrs_noevt = sum_mrs01(cur_mrs_noevt)
      

        # IVT costs should allways be included (if IVT is given)
        C_ivt = self.C.costs_IVT*pt_dct['IVT']
        
        #CTP arm withouth evt
        C_treatment_CTP_noevt = C_ivt + self.C.costs_CTP
        cur_mrs_CTP_noevt = cur_mrs_noevt.copy()
        costs_ctp_noevt = self.C(cur_mrs_CTP_noevt,1)
        qalys_ctp_noevt = self.Q(cur_mrs_CTP_noevt,1)
        row_ctp_noevt = [ID,'noEVT','CTP',self.start_year,1,
                           *list(cur_mrs_CTP_noevt), 
                            C_treatment_CTP_noevt,
                           *list(costs_ctp_noevt),
                           *list(qalys_ctp_noevt),
                            ]
        
        #CTP arm with evt
        cur_mrs_CTP_evt = cur_mrs_evt.copy()
        C_treatment_CTP_evt = C_treatment_CTP_noevt + self.C.costs_EVT
        costs_ctp_evt = self.C(cur_mrs_CTP_evt,1)
        qalys_ctp_evt = self.Q(cur_mrs_CTP_evt,1)
        row_ctp_evt = [ID,'EVT','CTP',self.start_year,1,
                       *list(cur_mrs_CTP_evt), 
                        C_treatment_CTP_evt,
                       *list(costs_ctp_evt),
                       *list(qalys_ctp_evt),
                        ]
        

        #if no CTP used all are treated
        C_treatment_noCTP_evt = C_ivt + self.C.costs_EVT
        cur_mrs_noCTP_evt = cur_mrs_evt.copy()  
        costs_noctp = self.C(cur_mrs_noCTP_evt,1)
        qalys_noctp = self.Q(cur_mrs_noCTP_evt,1)

        row_noctp_evt = [ID,'EVT','noCTP',self.start_year,1,
                   *list(cur_mrs_noCTP_evt),
                   C_treatment_noCTP_evt,
                   *list(costs_noctp),
                   *list(qalys_noctp)
                    ]

        #if no CTP used all are treated --> add a miss rate for M2-3
        C_treatment_noCTP_noevt = C_ivt
        cur_mrs_noCTP_noevt = cur_mrs_noevt.copy()  
        costs_noctp = self.C(cur_mrs_noCTP_noevt,1)
        qalys_noctp = self.Q(cur_mrs_noCTP_noevt,1)

        row_noctp_noevt = [ID,'noEVT','noCTP',self.start_year,1,
                   *list(cur_mrs_noCTP_noevt),
                   C_treatment_noCTP_noevt,
                   *list(costs_noctp),
                   *list(qalys_noctp)
                    ]

        out = [row_ctp_evt, row_ctp_noevt, row_noctp_evt, row_noctp_noevt]
        return cur_mrs_CTP_evt, cur_mrs_CTP_noevt, cur_mrs_noCTP_evt, cur_mrs_noCTP_noevt, out
    
    def _simulate_year(self,cur_mrs,ID,
                       treatment,ctp_noctp,
                       cur_year,yearno,
                       sex,cur_age):
        
        # simulate mortality and stroke recurrence of ctp arm
        cur_mrs, __ = self.M(sex,cur_year,cur_age,cur_mrs)
        cur_mrs, __ = self.RS(yearno,cur_age,cur_mrs)

        #compute costs and qalys of ctp arm
        costs = self.C(cur_mrs,yearno)
        qalys = self.Q(cur_mrs,yearno)

        row = [ID,treatment,ctp_noctp,
                cur_year,yearno, 
                *list(cur_mrs), #mrs disttribution
                0,*list(costs), #intervention costs and LT-costs
                *list(qalys) # LT sim QALYs
                ] 
        return cur_mrs, row
          
    def _simulate_long_term(self,pt_dct,
                            out,
                            cur_mrs_CTP_evt,
                            cur_mrs_CTP_noevt, 
                            cur_mrs_noCTP_evt,
                            cur_mrs_noCTP_noevt,
                           ):
        
        #LT sims start after the first year (all should have +1)
        cur_year = self.start_year+1
        yearno = 2
        cur_age = pt_dct['age']+1
        if pt_dct['sex']==0:
            sex = 'F'
        else:
            sex = 'M'
        
        ID = pt_dct['ID']
        
        #simulate per year
        for yearno in range(yearno,self.years_to_simulate+1):         
            #simulate mRS|CTP&EVT
            cur_mrs_CTP_evt, row = self._simulate_year(
                                                    cur_mrs_CTP_evt,
                                                    ID,'EVT','CTP',
                                                    cur_year,yearno,
                                                    sex,cur_age)
            out.append(row)

            #simulate mRS|CTP&noEVT
            cur_mrs_CTP_noevt, row = self._simulate_year(
                                                    cur_mrs_CTP_noevt,
                                                    ID,'noEVT','CTP',
                                                    cur_year,yearno,
                                                    sex,cur_age)         
            out.append(row)

            #simulate mRS|noCTP&EVT
            cur_mrs_noCTP_evt, row = self._simulate_year(
                                                    cur_mrs_noCTP_evt,
                                                    ID,'EVT','noCTP',
                                                    cur_year,yearno,
                                                    sex,cur_age)         
            out.append(row)
            #simulate mRS|noCTP&noEVT
            cur_mrs_noCTP_noevt, row = self._simulate_year(
                                                    cur_mrs_noCTP_noevt,
                                                    ID,'noEVT','noCTP',
                                                    cur_year,yearno,
                                                    sex,cur_age)         
            out.append(row)
            #mortality is calculated with that start age per year
            cur_age+=1
            cur_year+=1
            if self.verbal:
                print('year {} mRS CTP+EVT:'.format(yearno),cur_mrs_CTP_evt)
                print('year {} mRS CTP+noEVT:'.format(yearno),cur_mrs_CTP_noevt)
                print('year {} mRS noCTP+EVT:'.format(yearno),cur_mrs_noCTP_evt)
                print('year {} mRS noCTP+noEVT:'.format(yearno),cur_mrs_noCTP_noevt)

        return out
        
    def __call__(self,pt_dct):
        
        #sim_control can be set to false since control arm is only required once
        
        #simulate 90-day outcome (as 1-year outcome in costs)
        cur_mrs_CTP_evt, cur_mrs_CTP_noevt, \
        cur_mrs_noCTP_evt, cur_mrs_noCTP_noevt, out = self._90d_both_arms(pt_dct)
        
        if self.verbal:
            print('mRS 90d CTP+EVT:',cur_mrs_CTP_evt)
            print('mRS 90d CTP+noEVT:',cur_mrs_CTP_noevt)
            print('mRS 90d noCTP+EVT:',cur_mrs_noCTP_evt)
            print('mRS 90d noCTP+noEVT:',cur_mrs_noCTP_noevt)

        #perform long term simulation
        out = self._simulate_long_term(pt_dct, out,
                                        cur_mrs_CTP_evt,
                                        cur_mrs_CTP_noevt, 
                                        cur_mrs_noCTP_evt,
                                        cur_mrs_noCTP_noevt)
        #store all data
        cols = [
                'ID','treatment','ctp','year','yearno',
                *['mRS'+str(i) for i in range(1,7)],
                'treat_Costs', *['Costs'+str(i) for i in range(1,7)],
                *['QALYs'+str(i) for i in range(1,7)],
                ]
        
        df_out = pd.DataFrame(out,columns=cols)
        df_out['TotCosts'] = df_out[[c for c in df_out if 'Costs' in c]].sum(axis=1)
        df_out['TotQALYs'] = df_out[[c for c in df_out if 'QALYs' in c]].sum(axis=1)
        return df_out




# def compute_shift_no_evt(pt_dct, ORs):
#     """
#     pt_dct: dict with keys are also keys in ORs 
#             values are used to compute absolute shift
#     ORs: dict with keys of Tx values are Odds Ratios for favorable mRS
    
#     return one hot encoded probabilities of mRS for given patient
#     """
#     p_shift = 0
#     for name,OR in ORs.items():
#         if name in pt_dct.keys():
#             p_shift -= pt_dct[name]*np.log(OR)
#     mRS = np.zeros(7)
#     #compute one hot embedding
#     fl_new_mrs = pt_dct['mRS']+p_shift
#     new_mrs = int(np.floor(fl_new_mrs))
#     p_new_mrs = 1-(fl_new_mrs-new_mrs)
#     mRS[new_mrs] = p_new_mrs 
#     mRS[new_mrs+1] = 1-p_new_mrs 
#     return mRS

# def get_mRS_EVT_noEVT(df,ORs):
#     """  
#     df is a pandas.DataFrame existing of patients with EVT:
#         index=ID
#         columns: mRS and keys in ORs
#     ORs is a dictionary with Odds Ratios for treatment effect (OR for favorable outcome)
    
#     returns: one hot encoded mRS of patient if EVT was given or not (noEVT)
#     """
#     dct = df.to_dict(orient='index')
#     out = []
#     for pt_dct in dct.values():
#         mrs = compute_shift_no_evt(pt_dct, ORs)
#         out.append(mrs.tolist())
#     noEVT = pd.DataFrame(out, columns=[*['noEVT_mRS_'+str(i) for i in range(7)]], index=dct.keys())
#     EVT = pd.get_dummies(df['mRS'], prefix='EVT_mRS')
#     df_out = pd.concat([EVT,noEVT], ignore_index=False, axis=1)
#     return df_out

# def mRS_90d_2arms(input_dist,ID_treat_select, return_relative=False):
#     evt_cols = [c for c in input_dist.columns if not 'noevt' in c.lower()]
#     noevt_cols = [c for c in input_dist.columns if 'noevt' in c.lower()]

#     select_EVT = input_dist.loc[ID_treat_select][evt_cols]
#     exclude_EVT = input_dist[~np.isin(input_dist.index,ID_treat_select)][noevt_cols]

#     p_evt_90d_mrs = select_EVT.sum(axis=0)
#     p_noevt_90d_mrs = exclude_EVT.sum(axis=0)
#     if return_relative:
#         p_evt_90d_mrs = p_evt_90d_mrs /p_evt_90d_mrs.sum()
#         p_noevt_90d_mrs = p_noevt_90d_mrs/p_noevt_90d_mrs.sum()
        
#     return p_evt_90d_mrs, p_noevt_90d_mrs, select_EVT, exclude_EVT
