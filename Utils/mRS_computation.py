import numpy as np
import math
import pandas as pd
import time
import re
import os
import pickle
#from missingpy import MissForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
import sys

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def plogis(x):
	return (1+ math.tanh(x/2))/2

def ordinal_mrs_pred(pt_dct, 
					 tx_vars={},
					 intercepts = [-1.95789,-0.423007,0.704584,1.47090,2.16359,2.52240]
					 ):
	#pt_dct = {'EVT':1,'collaterals':0, 'onsettogroin':210,'core_vol':90}
	#tx_vars = {'collaterals':0, 'onsettogroin':0, 'core_vol':0}
	#intercepts mrs 0-6
	
	# treatment effect
	tx = 0.401368*pt_dct['EVT']

	if np.sum(list(tx_vars.values()))>0 and pt_dct['EVT']!=0:
		tx+=((0.230021*pt_dct['EVT']*pt_dct['collaterals']*tx_vars['collaterals'])\
		-(0.00116301*pt_dct['EVT']*pt_dct['onsettogroin']*tx_vars['onsettogroin'])\
		-(0.0*pt_dct['EVT']*pt_dct['core_vol']*tx_vars['core_vol']))

	out = []
	track_mrs = 0
	for mrs,intr in enumerate(intercepts):
		prob = plogis(tx+intr)- track_mrs  
		out.append(prob)
		track_mrs+=prob
	out.append(1-np.sum(out))
	return out

def mRS012_MRPREDICTS(dct): # the dct used is derived from MR CLEAN registry variable names
	nihss = dct['NIHSS_BL']
	diabetes = dct['prev_dm']
	IVT = dct['ivtrom']
	ASPECTS = dct['ASPECTS_BL']
	location = dct['occlsegment_c_short']
	prevstroke = dct['prev_str']
	age = dct['age1']
	premrs = dct['premrs']
	collaterals = dct['collaterals']
	togroin = dct['togroin']
	EVT = dct['EVT']
	bloodpressure = dct['rr_syst']

	linearpred =  0.85443756*EVT\
	-0.0041355388*age\
	-2.9309762e-05*math.pow(np.maximum(age-47,0),3)\
	+7.133317e-05*math.pow(np.maximum(age-66.949692,0),3)\
	-4.2023408e-05*math.pow(np.maximum(age-80.863859,0),3)\
	-0.067628368*nihss\
	-0.37661632*premrs\
	-0.48940477*diabetes\
	-0.00054061263*bloodpressure\
	-2.9407133e-06*math.pow(np.maximum(bloodpressure-117,0),3)\
	+5.1462482e-06*math.pow(np.maximum(bloodpressure-144,0),3)\
	-2.205535e-06*math.pow(np.maximum(bloodpressure-180,0),3)\
	+0.5280263*IVT\
	+0.11190797*ASPECTS\
	+0.51530329*(location==3.0)\
	+0.82431954*(location==4.0)\
	+0.39137525*collaterals\
	-0.1103293*prevstroke\
	-0.0028771638*togroin\
	-0.60324649*EVT*prevstroke\
	-0.0015778723*EVT*togroin\
	+0.11861626*EVT*collaterals\
	-0.9776372498

	#p = logistic.cdf(linearpred)
	#p = sigmoid(linearpred)
	p = plogis(linearpred) # own implementation of R script
	return p

# given a dictionary of patient characteristics return a full mRS distribution bar
# dictionary input is of keys=R-numbers, values=dictionary of keys=varname, values=variable value
def full_mRS_pred(dct):
	#for k,v in dct.items():
	 #   globals().update({k: v})

	nihss = dct['NIHSS_BL']
	diabetes = dct['prev_dm']
	IVT = dct['ivtrom']
	ASPECTS = dct['ASPECTS_BL']
	location = dct['occlsegment_c_short']
	prevstroke = dct['prev_str']
	if 'age1' in list(dct.keys()):
		age = dct['age1']
	else:
		age = dct['age']
	premrs = dct['premrs']
	collaterals = dct['collaterals']
	togroin = dct['togroin']
	#EVT = dct['EVT']
	bloodpressure = dct['rr_syst']

	# linear predictor   
	lp = -0.0041355388*age\
	-2.9309762e-05*math.pow(np.maximum(age-47,0),3)\
	+7.133317e-05*math.pow(np.maximum(age-66.949692,0),3)\
	-4.2023408e-05*math.pow(np.maximum(age-80.863859,0),3)\
	-0.067628368*nihss\
	-0.37661632*premrs\
	-0.48940477*diabetes\
	-0.00054061263*bloodpressure\
	-2.9407133e-06*math.pow(np.maximum(bloodpressure-117,0),3)\
	+5.1462482e-06*math.pow(np.maximum(bloodpressure-144,0),3)\
	-2.205535e-06*math.pow(np.maximum(bloodpressure-180,0),3)\
	+0.5280263*IVT\
	+0.11190797*ASPECTS\
	+0.51530329*(location==3.0)\
	+0.82431954*(location==4.0)\
	+0.39137525*collaterals\
	-0.1103293*prevstroke\
	-0.0028771638*togroin
	
	#treatment effect
	tx = 0.85443756\
	-0.60324649*prevstroke\
	-0.0015778723*togroin\
	+0.11861626*collaterals
	
	# probabilities
	mrs0_EVT = (plogis(-3.5647832851 + lp + tx))
	mrs1_EVT = (plogis(-2.1009284376 + lp + tx)) - mrs0_EVT
	mrs2_EVT = (plogis(-0.9776372498 + lp + tx)) - (mrs0_EVT + mrs1_EVT)
	mrs3_EVT = (plogis(-0.0970143526 + lp + tx)) - (mrs0_EVT + mrs1_EVT + mrs2_EVT) 
	mrs4_EVT = (plogis(1.2347802741 + lp + tx)) - (mrs0_EVT + mrs1_EVT + mrs2_EVT + mrs3_EVT)
	mrs5_EVT = (plogis(1.8207475278 + lp + tx)) - (mrs0_EVT + mrs1_EVT + mrs2_EVT + mrs3_EVT + mrs4_EVT)
	mrs6_EVT = 1 - (mrs0_EVT + mrs1_EVT + mrs2_EVT + mrs3_EVT + mrs4_EVT + mrs5_EVT)

	mrs0_noEVT = (plogis(-3.5647832851 + lp))
	mrs1_noEVT = (plogis(-2.1009284376 + lp)) - mrs0_noEVT
	mrs2_noEVT = (plogis(-0.9776372498 + lp)) - (mrs0_noEVT + mrs1_noEVT)
	mrs3_noEVT = (plogis(-0.0970143526 + lp)) - (mrs0_noEVT + mrs1_noEVT + mrs2_noEVT) 
	mrs4_noEVT = (plogis(1.2347802741 + lp)) - (mrs0_noEVT + mrs1_noEVT + mrs2_noEVT + mrs3_noEVT)
	mrs5_noEVT = (plogis(1.8207475278 + lp)) - (mrs0_noEVT + mrs1_noEVT + mrs2_noEVT + mrs3_noEVT + mrs4_noEVT)
	mrs6_noEVT = 1 - (mrs0_noEVT + mrs1_noEVT + mrs2_noEVT + mrs3_noEVT + mrs4_noEVT + mrs5_noEVT)
	
	df_out = pd.DataFrame(columns = ['mRS_0','mRS_1', 'mRS_2','mRS_3','mRS_4','mRS_5','mRS_6'])
	df_out.loc[len(df_out)] = [mrs0_EVT,mrs1_EVT,mrs2_EVT,mrs3_EVT,mrs4_EVT,mrs5_EVT,mrs6_EVT]
	df_out.loc[len(df_out)] = [mrs0_noEVT,mrs1_noEVT,mrs2_noEVT,mrs3_noEVT,mrs4_noEVT,mrs5_noEVT,mrs6_noEVT]
	df_out['mRS_012'] = df_out[['mRS_0','mRS_1', 'mRS_2']].sum(axis=1)
	df_out['mRS_345'] = df_out[['mRS_3','mRS_4', 'mRS_5']].sum(axis=1)
	df_out.index = ['EVT', 'noEVT']
	
	return df_out

def load_MRPREDICTS_vars(cols):
	pp.final_df(load=True, sav=False)
	df = pp.cleaned_df[cols]
	df = pd.concat([pp.outcome_df, df], axis=1)
	#mdling.preprocess_all(mdling.train_ID, mdling.test_ID, load=True, sav=False, imp_method_load='RF')
	#catname_dct = pp.catname_dict()
	
	#df.columns = [col_dct[c] for c in df]
	#ix_tr = np.isin(df.index,mdling.train_ID)
	#ix_tst = np.isin(df.index,mdling.test_ID)

	return df #df[ix_tr],df[ix_tst]

# if you want to fit an imputer on a train set 
#and impute on other set with the same imputer
def impute_tr_tst(train, test,impcols, imptr):
	train = train[impcols]
	test = test[impcols]
	
	itrain, itest = mdling.impute(train,test, imputer=imptr)
	itrain = pd.DataFrame(itrain, columns = impcols, index = train.index)
	itest = pd.DataFrame(itest, columns = impcols, index = test.index)
	
	# adjust location of infarct since this is modelled as a dummy
	itrain['occlsegment_c_short'] = itrain['occlsegment_c_short'].round()
	itest['occlsegment_c_short'] = itest['occlsegment_c_short'].round()
	
	return itrain, itest

# function to make the predictions of a dataframe of pt with 
# def MR_PREDICTS_pred_mrs02(df, pred_varname='mrs02_pred'):
#     ptd = df.to_dict(orient='index')
	
#     pred = []
#     for ID,pt_dct in ptd.items():
#         pt_dct['EVT'] = 1
#         pred.append(mRS012_MRPREDICTS(pt_dct))
	
#     df['mrs02_pred'] = pred 
	
#     outcome = mdling.metrics_outcome(y_true=(df['mrs']<3).values, score=df[pred_varname].values)
#     return df, outcome

# maybe an outdated function
def occl_segm_transform(df):
	df['occlsegment_c_short'][df['occlsegment_c_short']==3]='M1'
	df['occlsegment_c_short'][df['occlsegment_c_short']==4]='M2'
	return df

# ### CE per patient functions ####
# def binary_crossentropy(y_true, y_pred):
#     out = (-1*y_true)*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
#     return out

# def crossentropy_per_class(y_true, y_pred):
#     y0 = abs((1-y_true)*np.log(1-y_pred))
#     y1 = abs((-1*y_true)*np.log(y_pred))
#     both = y0+y1
#     return y0,y1,both

# def crossentropy_per_pt(y_true,y_pred):
#     out = np.zeros((3,y_true.shape[0]))
#     for i in range(y_true.shape[0]):
#         out[:,i] = crossentropy_per_class(y_true[i], y_pred[i])
#     return out

# # from here could be in a class
# def MR_PRED_CE(df, yt_var, yp_var):
#     y_true = (df[yt_var]<3).values*1
#     y_pred = df[yp_var].values
#     ce = crossentropy_per_pt(y_true,y_pred)
#     df_out = pd.DataFrame(np.vstack([y_true,y_pred,ce]), 
#                          columns=df.index, index=['y_true', 'y_pred','y0CE','y1CE','CE']).T
#     return df_out

# def MR_PREDICTS_predictions(args, cols, imptr, catvar_ix):
#     df = load_MRPREDICTS_vars(cols)
#     df, imptr = impute_data(df, cols, imptr, catvar_ix)



# def LR_model_CE(args):
#     __, best_clf, __, __, __ = pickle.load(open(args.loc_modelling+'\\'+'LR_data.pic','rb'))
#     data_dict = pickle.load(open(args.loc_modelling+'\\'+'pp'+'\\'+'data_dict.pic', "rb"))
#     x_train, y_train, x_test, y_test, final_use_vars = data_dict['final2use']
	
#     y_true = pd.concat([y_train,y_test],axis=0)
#     y_pred = best_clf.predict_proba(np.vstack([x_train,x_test]))[:,1]
#     ce = crossentropy_per_pt(y_true.values*1,y_pred)

#     df_out = pd.DataFrame(np.vstack([y_true.values*1,y_pred,ce]), 
#                      columns=y_true.index, index=['y_true', 'y_pred','y0CE','y1CE','CE']).T

#     return df_out


# class Predict():
	
#     def __init__(self, args):
#         super(Predict, self).__init__()
#         # load saving location (also for loading)
#         self.savloc = args.loc_modelling+'\\'+'CE'
#         #load the data for processing
#         pp.load_final_df()
#         self.cleaned_df = pp.cleaned_df
#         self.outcome_df = pp.outcome_df.dropna()
#         self.outcomevar = args.outcomevar

#     def load_MR_PREDICTS_data(self):
#         self.modelname = 'MR_PREDICTS'
#         self.cols = ['ASPECTS_BL', 'NIHSS_BL', 'age1', 
#             'collaterals', 'ivtrom','occlsegment_c_short', 'premrs', 
#             'prev_dm', 'prev_str','rr_syst', 'togroin']    
		
#         df = self.cleaned_df[self.cols]
#         self.df = pd.concat([self.outcome_df, df], axis=1)
#         # array depicting the column indices of categorical variables
#         self.catvar_ix = np.array([4,5,7,8])

#     def load_other_model(self):
#         print('construct a model and do stuff this is not used')
#         # in this function at least the following needs to be retained:
#         # df of all the variables used to impute (R-ID aligned with outcome vars)
#         # cols for which columns to use in the df
#         # catvar_ix; a column index that depicts which categorical variables to impute as categories

#     # impute on one dataset requires: columns, df, catvar_ix (if required)
#     def impute_data(self, imptr, catvar_ix=None):
#         t1 = time.time()
#         if catvar_ix==None:
#             catvar_ix = self.catvar_ix

#         x = self.df[self.cols]
#         imptr.fit(x, cat_vars=catvar_ix) #random_state=None (a mistake made)
#         x_imp = imptr.transform(x)
#         # return the imputed dataframe, the imputer and the catvars_ix used
#         self.idf = pd.DataFrame(x_imp, columns = self.cols, index = x.index)
#         self.idf = pd.concat([self.outcome_df, self.idf], axis=1)
#         self.imputer = imptr
#         self.catvar_ix = catvar_ix

#         t2 = time.time()
#         print('Imputing time:',round(t2-t1))
	
#     ## make predictions
#     def main_MR_PREDICTS(self, imptr):
#         self.load_MR_PREDICTS_data()
#         self.impute_data(imptr)
#         self.idf,self.outcome_metrics = MR_PREDICTS_pred_mrs02(self.idf, pred_varname='mrs02_pred')
#         print('MR PREDICTS performance:', self.outcome_metrics)
#         self.CE_df = MR_PRED_CE(self.idf, self.outcomevar, 'mrs02_pred')
		
#     def save(self):
#         self.pred_dct = {'model':self.modelname, 
#             'prediction_df':self.idf,
#             'performance':self.outcome_metrics,
#             'CE_df':self.CE_df,
#             'imputer': self.imputer,
#             'catvars_imp':self.catvar_ix}
#         savloc = self.savloc+'\\'+'CE_pred_'+self.modelname+'.pic'
#         pickle.dump(self.pred_dct, open(savloc, "wb"))
 
#     def load(self, modelname):   
#         savloc = self.savloc+'\\'+'CE_pred_'+modelname+'.pic'
#         self.pred_dct = pickle.load(open(savloc, "rb"))










