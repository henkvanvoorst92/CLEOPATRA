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
        new mrs distribution or survival rate
    """
    
    def __init__(self, p_mort_file, 
                 HR_mrs = np.array([1.54,2.17,3.18,4.55,6.55]),
                 years = np.arange(2021,2030)):
        #mortality probabilities
        mort_male = pd.read_excel(p_mort_file, 'male').iloc[:,:30]
        mort_male.index = mort_male['age']
        mort_female = pd.read_excel(p_mort_file, 'female').iloc[:,:30]
        mort_female.index = mort_female['age']

        self.dct_mort = {'M':mort_male[years].to_dict(),
                        'F':mort_female[years].to_dict()}
        
        self.HR_mrs = HR_mrs
        
    def _init_prob_resample(self):
        print('To implement probabilistic resampling of HR')
        
    def __call__(self,gender,year,age, mrs_dist=None):
        # input mRS distribution (mrs_dist) should always sum up to 1

        p_mort = self.dct_mort[gender][year][age]
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

