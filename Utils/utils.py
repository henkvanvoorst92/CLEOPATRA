import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys, time,subprocess, glob, timeit,json
from numba import jit, njit, prange
from tqdm import tqdm

def sum_mrs01(mrs_dist):
    mrs_dist[1] = mrs_dist[0]+mrs_dist[1]
    return mrs_dist[1:]

def store_df(input, sav_loc, type, cols=None):

    if isinstance(input,list):
        df_out = pd.DataFrame(input).reset_index(drop=True)
    elif isinstance(input,pd.DataFrame):
        df_out = input
    else:
        print('input type not compatible',type(input))
    sav_loc += type

    if isinstance(cols,list):
        df_out.columns = cols

    if os.path.exists(sav_loc):
        if type=='.xlsx':
            tmp = pd.read_excel(sav_loc)
        elif type=='.pic':
            tmp = pd.read_pickle(sav_loc)
        elif type=='.ftr':
            tmp = pd.read_feather(sav_loc)
        else:
            print('Error wrong type of extension:', type)
        if tmp.shape != (0,0): #sometimes written tmp is empty
            if 'Unnamed: 0' in tmp.columns:
                tmp = tmp.drop(columns=['Unnamed: 0'])
            if isinstance(cols,list):
                tmp.columns = cols
            df_out = pd.concat([df_out,tmp], axis=0).reset_index(drop=True)

    if type=='.xlsx':
        df_out.to_excel(sav_loc)
    elif type=='.pic':
        df_out.to_pickle(sav_loc)
    elif type=='.ftr':
        df_out.to_feather(sav_loc)

def store_opt_json(opt):
    p_json = os.path.join(opt.loc_checkpoints,'opt.json')
    dct = vars(opt)
    for k,v in dct.items():
        if isinstance(v,type):
            dct[k] = str(v)
        elif isinstance(v,np.ndarray):
            dct[k] = str(v)
    with open(p_json, 'w', encoding='utf-8') as f:
        json.dump(dct, f, ensure_ascii=False, indent=4)

def load_opt_json(root):
    p_json = os.path.join(root,'opt.json')
    with open(p_json) as f:
        dct = json.load(f)
    for k,v in dct.items():
        if k=='norm':
            #print(k,v)
            dct[k] = getattr(nn,v.strip("<>''").split('.')[-1])
    dct['loc_checkpoints'] = root
    return argparse.Namespace(**dct)

def optimal_icv_threshold(df):
    res = df.groupby(['threshold']).median().reset_index(drop=False)
    
    res.index = res['threshold']
    max_NMB_threshold = res['NMB'].idxmax()
    print('Optimal NMB threshold:',max_NMB_threshold)
    print('Optimal NMB:',res['NMB'].max())
    return df[df['threshold']==max_NMB_threshold]

