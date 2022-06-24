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
                 start_inflation=1.000, # inflation factor before simulation (2015 -> 2022)
                 discounting_rate=1.04,
                 inflation_rate=1.04,
                 ):
        
        self.df = pd.read_excel(file)
        self.start_inflation = start_inflation
        self.discounting_rate = discounting_rate
        self.inflation_rate = inflation_rate
        
        self._init_cost_vector()
        
    def _init_cost_vector(self,probabilistic=False):
        
        self.df = self.df.sort_values(by=['year','mRS'], ascending=True)
        #function initializes the mrs-costs vector
        self.dct_year_costs = {}
        for y in self.df['year'].unique():
            tmp = self.df[self.df['year']==y]
            if not probabilistic:
                # perform baseline simulations with mean values
                 v = tmp['mean'].values
            else:
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
            
            #adjust costs to current year (default prices were in 2015)
            if self.start_inflation!=1.000:
                v = self._presim_inflation_adjust(v)
            
            self.dct_year_costs[y] = v

    def _probabilistic_resample(self):
        self._init_cost_vector(probabilistic=True)
    
    def _presim_inflation_adjust(self,costs):
        return costs*self.start_inflation
    
    def _inflation_and_discouting(self,costs,year):
        infl = self.inflation_rate**year
        disc = self.discounting_rate**(1/year)
        return costs*infl*disc
    
    def __call__(self,mrs_dist,year):
        # costs beyond year 3 are set equal to year 3
        mrs_costs = mrs_dist*self.dct_year_costs[np.clip(year,1,3)]
        #correct for discounting and inflation
        mrs_costs = self._inflation_and_discouting(mrs_costs,year)
        return np.round(mrs_costs,2)


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
        
    def _init_qaly_vector(self,probabilistic=False):
        
        self.df = self.df.sort_values(by=['mRS'], ascending=True)
        #function initializes the mrs-costs vector
        
        if not probabilistic:
            # perform baseline simulations with mean values
             v = self.df['mean'].values
        else:
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
                
        self.qaly_vector = v
            
    def _probabilistic_resample(self):
        self._init_qaly_vector(probabilistic=True)
    
    def _discounting(self,qalys,year):
        return qalys*self.discounting_rate**(1/year)
    
    def __call__(self,mrs_dist,year):
        # costs beyond year 3 are set equal to year 3
        mrs_qalys = mrs_dist*self.qaly_vector
        #correct for discounting and inflation
        mrs_qalys = self._discounting(mrs_qalys,year)
        return mrs_qalys

