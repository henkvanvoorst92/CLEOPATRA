import numpy as np
import pandas as pd

class Costs(object):
    """
    Inputs:
        file: used to load a pd dataframe --> 
        structure should fit through _init_cost_vector
        start_inflation: inflation factor used to transform default period
        costs to the simulatiion starting year (= % change + 1)
        discounting_rate: yearly discounting rate rate (= % change + 1)
        inflation_rate: yearly inflation of costs (= % change + 1)
        
    Returns:
        costs per mRS markov state
    """
    
    def __init__(self, 
                 file,
                 costs_IVT, costs_CTP, costs_EVT,
                 start_inflation=1.000, # inflation factor before simulation (2015 -> 2023)
                 discounting_rate=1.04,
                 inflation_rate=1.04,
                 ):
        
        self.df = pd.read_excel(file)
        self.start_inflation = start_inflation
        self.discounting_rate = discounting_rate
        self.inflation_rate = inflation_rate
        #starting costs of treatment
        self.costs_IVT = costs_IVT*self.start_inflation
        self.costs_CTP = costs_CTP*self.start_inflation #correct prices in ESJ consider year 2015
        self.costs_EVT = costs_EVT*self.start_inflation
        self._init_cost_vector()

        #original data for sens analyses
        self.df_org = self.df.copy()
        self.costs_IVT_org = self.costs_IVT.copy()
        self.costs_CTP_org = self.costs_CTP.copy()
        self.costs_EVT_org = self.costs_EVT.copy()

    def _init_cost_vector(self,mode='default'):
        
        self.df = self.df.sort_values(by=['year','mRS'], ascending=True)
        #function initializes the mrs-costs vector
        self.dct_year_costs = {}
        for y in self.df['year'].unique():
            tmp = self.df[self.df['year']==y]
            if mode=='default':
                # perform baseline simulations with mean values
                 v = tmp['mean'].values
            elif mode=='probabilistic':
                #sample from gamma distribution
                tmp.index = tmp['mRS']
                v = np.zeros_like(tmp['mean'].values)
                for mrs in tmp['mRS']:
                    mean = tmp.loc[mrs,'mean']
                    std = tmp.loc[mrs,'std']
                    # alpha and beta params for sampling
                    A = mean**2/std**2
                    B = 1/(mean/std**2)
                    #sample new datapoint from gamma distribution
                    v[mrs-1] = np.random.gamma(A,B)
            #elif mode=='deterministic_low' 

            #elif mode=='deterministic_high'

            #adjust costs to current year (default prices were in 2015)
            if self.start_inflation!=1.000:
                v = self._presim_inflation_adjust(v)
            
            #mrs6 only attributes to costs once (if death no extra costs)
            self.costs_once_mrs6 = v[-1] 
            v[-1] = 0
            self.dct_year_costs[y] = v

    def _probabilistic_resample(self):
        self._init_cost_vector(mode='probabilistic')
    
    def _presim_inflation_adjust(self,costs):
        return costs*self.start_inflation
    
    def _inflation_and_discouting(self,costs,year):
        infl = self.inflation_rate**year
        disc = self.discounting_rate**(1/year)
        return costs*infl*disc
    
    def _get_params(self,current=True):
        #if current returns the current variables
        #else the original df is loaded --> used for PSA/OneWaySens
        if not current:
            self.df_ = self.df_org.copy()
            self.costs_IVT = self.costs_IVT_org.copy()
            self.costs_CTP = self.costs_CTP_org.copy()
            self.costs_EVT = self.costs_EVT_org.copy()
        
        tmp1 = pd.DataFrame(self.df['mean'].values, columns=['value'])
        tmp1['mRS'] = self.df['mRS']
        tmp1['year'] = self.df['year']
        tmp1['variable_type'] = 'costs_py'
        tmp1['variable'] = ['mrs{}_year{}'.format(mrs,y) for mrs,y in zip(self.df['mRS'],self.df['year'])]

        tmp2 = pd.DataFrame([self.costs_IVT, self.costs_CTP,self.costs_EVT], columns=['value'])
        tmp2['variable'] = ['costs_IVT', 'costs_CTP', 'costs_EVT']
        tmp2['variable_type'] = 'costs_treat'

        out = pd.concat([tmp1,tmp2])
        out['object'] = 'C'
        return out

    def _set_params(self,data):
        #set all params according to a dataframe (data)
        self.df = data[(data['object']=='C') & (data['variable_type']=='costs_py')]
        self.df['mean'] = self.df.sort_values(by=['year','mRS'])['value'].astype(np.float32)
        self._init_cost_vector(mode='default') #cannot be used for PSA

        self.costs_IVT = np.float(data[data['variable']=='costs_IVT']['value'].values[0])
        self.costs_CTP = np.float(data[data['variable']=='costs_CTP']['value'].values[0])
        self.costs_EVT = np.float(data[data['variable']=='costs_EVT']['value'].values[0])

    def __call__(self,mrs_dist,year,prev_mrs_dist=None):
        #mrs_dist:current mrs distribution
        #prev_mrs_dist: previous year distribution

        # costs beyond year 3 are set equal to year 3
        #baseline costs mrs01-5
        mrs_costs = mrs_dist*self.dct_year_costs[np.clip(year,1,3)]
        #single time mrs6 costs
        #transition cost death based on new deaths (not death in previous year)
        if prev_mrs_dist is None:
            prev_mrs_dist = np.zeros_like(mrs_dist)
            mrs_costs[-1] = np.clip(mrs_dist[-1]-prev_mrs_dist[-1],0,1)*self.costs_once_mrs6

        #correct for discounting and inflation
        mrs_costs = self._inflation_and_discouting(mrs_costs,year)
        # print(mrs_costs, type(mrs_costs))
        # self.mrs_costs = mrs_costs
        # self.mrs = mrs_dist
        return mrs_costs


class QALYs(object):
    """
    Inputs:
        file: used to load a pd dataframe --> 
        structure should fit through _init_qaly_vector
        discounting_rate: yearly discounting rate rate (= % change + 1)
        
    Returns:
        qalys per mRS markov state
    """
    
    def __init__(self, 
                 file,
                 discounting_rate=1.04,
                 ):
        
        self.df = pd.read_excel(file)
        self.discounting_rate = discounting_rate
        
        self._init_qaly_vector()
        
        #set original variables for sens analyses
        self.df_org = self.df.copy()

    def _init_qaly_vector(self,mode='default'):
        
        self.df = self.df.sort_values(by=['mRS'], ascending=True)
        #function initializes the mrs-costs vector
        
        if mode=='default':
            # perform baseline simulations with mean values
             v = self.df['mean'].values
        elif mode=='probabilistic':
            mean = self.df['mean'].values
            std = self.df['std'].values
            #sample from gamma distribution
            A = ((mean**2)*(1-mean))/(std**2)-mean 
            # wiki: A=mean*(((mean*(1-mean))/std**2)-1)
            B = (1-mean)*(((1-mean)*mean)/std**2-1) 
            # wiki: B=(1-mean)*((mean*(1-mean))/std**2-1)
            v = np.zeros_like(self.df['mean'].values)
            for ix,(a,b) in enumerate(zip(A,B)):
                if a==0:
                    v[ix] = 0
                else:
                    v[ix] = np.random.beta(a,b)

        self.qalys_once_mrs6 = v[-1]
        v[-1] = 0
        self.qaly_vector = v
            
    def _probabilistic_resample(self):
        self._init_qaly_vector(mode='probabilistic')
    
    def _discounting(self,qalys,year):
        return qalys*self.discounting_rate**(1/year)
    
    def _get_params(self,current=True):
        #if current returns the current variables
        #else the original df is loaded --> used for PSA/OneWaySens
        if not current:
            self.df = self.df_org.copy()
        
        out = pd.DataFrame(self.df['mean'].values, columns=['value'])
        out['mRS'] = self.df['mRS']
        out['variable_type'] = 'qalys_per_mrs'
        out['variable'] = ['mrs{}'.format(mrs) for mrs in self.df['mRS']]
        out['object'] = 'Q'
        return out

    def _set_params(self,data):
        #set all params according to a dataframe (data)
        self.df = data[data['object']=='Q']
        self.df['mean'] = self.df['value'].values
        self._init_qaly_vector(mode='default')#cannot be used for PSA


    def __call__(self,mrs_dist,year,prev_mrs_dist=None):
        # costs beyond year 3 are set equal to year 3
        mrs_qalys = mrs_dist*self.qaly_vector
        #single transition qalys if pt died
        if prev_mrs_dist is None:
            prev_mrs_dist = np.zeros_like(mrs_dist)
            mrs_qalys[-1] = np.clip(mrs_dist[-1]-prev_mrs_dist[-1],0,1)*self.qalys_once_mrs6
        #correct for discounting and inflation
        mrs_qalys = self._discounting(mrs_qalys,year)
        return mrs_qalys

