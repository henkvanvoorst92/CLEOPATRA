import numpy as np
import pandas as pd

class Mortality(object):
    """
    Inputs:
        p_mort_file: downloaded excel from https://www.ag-ai.nl/view.php?Pagina_Id=611
                     represents the forcasted probability of death by gender, age, and year
        HR_mrs: Hazard rates per mRS group (0,1,2,3,4,5)
        years: optional; years to select
    
    Returns:
        new mrs distribution and mortality rate
    """
    
    def __init__(self, 
                 p_mort_file, 
                 HR_mrs = np.array([1.5325,2.175,3.172,4.5525,6.55]),
                 delta_HR_mrs = np.array([1.54,2.18,3.18,4.56,6.5575]),
                 years = np.arange(2021,2030)):
        #mortality probabilities
        mort_male = pd.read_excel(p_mort_file, 'male').iloc[:,:30]
        mort_male.index = mort_male['age']
        mort_female = pd.read_excel(p_mort_file, 'female').iloc[:,:30]
        mort_female.index = mort_female['age']

        self.dct_mort = {'M':mort_male[years].to_dict(),
                        'F':mort_female[years].to_dict()}
        
        # HR from data by Hong et al: https://pubmed.ncbi.nlm.nih.gov/20133917/
        self.HR_mrs_deterministic = HR_mrs
        self.delta_HR_mrs = delta_HR_mrs
        self._init_HR()

    def _init_HR(self, mode='default'):
        if mode=='default':
            self.HR_mrs = self.HR_mrs_deterministic
        elif mode=='probabilistic':
            out = []
            for hr,dhr in zip(self.HR_mrs_deterministic,self.delta_HR_mrs):
                mean = np.log(hr)
                sigma = np.sqrt(abs(mean-np.log(dhr))*2)
                #print('Fill in sqrt(N) for accurate sigma!!!!' )
                out.append(np.random.lognormal(mean=mean, sigma=sigma))
            self.HR_mrs = np.array(out)
        #elif mode=='deterministic_low' 

        #elif mode=='deterministic_high'

    def _probabilistic_resample(self):
        self._init_HR(mode='probabilistic')
        
    def __call__(self,sex,year,age, mrs_dist=None):
        # input mRS distribution (mrs_dist) should always sum up to 1

        p_mort = self.dct_mort[sex][year][age]
        if mrs_dist is not None:
            p_mort = p_mort*self.HR_mrs*mrs_dist[:-1]
            #compute survival
            psurv = 1-mrs_dist[:-1]*p_mort
            #output is new distribution
            new_dist = mrs_dist[:-1]*psurv
            out = np.append(new_dist,1-new_dist.sum()) # add mortality
        else:
            out = None
        return out, p_mort

