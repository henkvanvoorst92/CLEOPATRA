import numpy as np
import pandas as pd


# recurrent stroke is based on age (HR) and year post index stroke (prob)
class RecurrentStroke(object):
    """
    Inputs:
        p_restroke_year: probability of stroke recurrence 
                        per year post index strok (dict)
        age_HR: HR by patient age (dict)
        mrs_post_restroke: mrs distribution after restroke
    
    """
    
    def __init__(self, 
                 file_p_HR,
                 p_mrs_postrestroke = \
                 np.array([11/233, 21/233,18/233,22/233,6/233]) #without mortality
                ):
        #Probability per year post stroke and HR from file
        self.file_p_HR = file_p_HR
        self.ppy = pd.read_excel(self.file_p_HR,'prob_per_year')
        self.HR = pd.read_excel(file_p_HR,'HR')
        
        # received from mrs
        self.p_mrs_postrestroke = p_mrs_postrestroke
        if self.p_mrs_postrestroke.sum()!=1:
            self.p_mrs_postrestroke/=self.p_mrs_postrestroke.sum()
    
        self._init_ppy()
        self._init_HR()
        
    def _init_ppy(self, probabilistic=False):
        self.p_restroke_per_year = {}
        for y, p in zip(self.ppy['year'], self.ppy['p']):
            self.p_restroke_per_year[y] = p
        if probabilistic:
            print('To implement probabilistic ppy')
            #implement a sampling strategy for binomial probabilities
    
    def _init_HR(self, probabilistic=False):
        
        # per year HR 
        age = self.HR[self.HR['variable']=='age'].values[0,1:]
        self.HR_age, self.HR_age_lowCI, self.HR_age_upCI = age
        #self.HR_age was used for age 64, others are relative to this
        
        DM = self.HR[self.HR['variable']=='DM'].values[0,1:]
        self.HR_DM, self.HR_DM_lowCI, self.HR_DM_upCI = DM
        HR_t = self.HR[self.HR['variable']=='2004_2008'].values[0,1:]
        self.HR_t, self.HR_t_lowCI, self.HR_t_upCI = HR_t
        if probabilistic:
            print('To implement probabilistic HR')
        
    def _init_prob_resample(self):
        self._init_ppy(probabilistic=True)
        self._init_HR(probabilistic=True)
    
    def _compute_age_HR(self,age):
        ref_age = 64
        HR_age = self.HR_age**(age-ref_age)
        return HR_age
    
    def _mrs_dist_post_restroke(self,mrs_dist):
        # computes new distribution of mrs
        # weighted for input mrs_dist
        # assumes all have a restroke
        
        out = np.zeros_like(mrs_dist[:-1])
        for mrs in np.nonzero(mrs_dist[:-1])[0]:
            new = np.zeros_like(self.p_mrs_postrestroke)
            new[mrs:] = self.p_mrs_postrestroke[mrs:]
            # adjust for input distribution
            out += (new/new.sum())*mrs_dist[mrs]
        return out
    
    def __call__(self,year_post_index_stroke,age, mrs_dist=None, DM=None):
		# input mRS distribution (mrs_dist) should always sum up to 1
        #compute HR age
        HR = self._compute_age_HR(age)*self.HR_t #pm remove this (HR_t is period '04-'08)
        if DM==1 or DM==True:
            HR*= self.HR_DM
        # compute recurrent stroke
        p_restroke = self.p_restroke_per_year[year_post_index_stroke]*HR
        if mrs_dist is not None:
            mrs_after_stroke = self._mrs_dist_post_restroke(mrs_dist)*p_restroke
            mrs_dist[:-1] = mrs_dist[:-1]*(1-p_restroke) + mrs_after_stroke
            out = mrs_dist
        else:
            out = None
            
        return out, p_restroke
