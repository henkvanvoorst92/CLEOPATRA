import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys 
from tqdm import tqdm

def reorder_labels_ix(curr_order,pref_order):
    new_order = []
    for lo in pref_order:
        new_order.append(curr_order.index(lo))
    return new_order
    

def ICER_plot(aggrs, name, root_fig=None, min_x=-.15,max_x =.15, min_y=-12000, max_y=12000, wtp=80000):

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
    y = wtp*x
    sns.lineplot(x,np.zeros_like(x),color='black',linewidth=1)
    sns.lineplot(np.zeros_like(y_b),y_b,color='black',linewidth=1,estimator=None)
    sns.scatterplot(data=dataplot,x='d_qalys',y='d_costs', hue='group')
    sns.lineplot(x,y,color='black')
    plt.ylim(min_y, max_y)
    plt.xlim(min_x, max_x)
    plt.ylabel('Difference in costs(€) (CTP - noCTP)')
    plt.xlabel('Difference in QALYS (CTP - noCTP)')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:  
        if not os.path.exists(root_fig):
            os.makedirs(root_fig)
        plt.savefig(os.path.join(root_fig,'ICER_{}.tiff'.format(name)),dpi=300)
    plt.show()

def single_ICER_plot(dataplot, 
                     name, 
                     root_fig=None, #if a path the figure is saved
                     minmax_xy=None,#[-.15, .15,-12000,12000],
                     color='black',
                     wtp=80000):
    
    if minmax_xy is None:
        min_x = dataplot.d_qalys.min()*0.9
        max_x = dataplot.d_qalys.max()*1.1
        min_y = dataplot.d_costs.min()*0.9
        max_y = dataplot.d_costs.max()*1.1
    else:
        min_x, max_x, min_y, max_y = minmax_xy
    
    print(min_x, max_x, min_y, max_y)
    x = np.linspace(min_x, max_x)
    y_b = np.arange(min_y, max_y,50)
    y = wtp*x
    sns.lineplot(x,np.zeros_like(x),color='grey',linewidth=1)
    sns.lineplot(np.zeros_like(y_b),y_b,color='grey',linewidth=1,estimator=None)
    sns.scatterplot(data=dataplot,x='d_qalys',y='d_costs',s=10,color=color)
    sns.lineplot(x,y,color='black', linestyle="dashed")
    plt.ylim(min_y, max_y)
    plt.xlim(min_x, max_x)
    plt.ylabel('Difference in costs(€) (CTP - no CTP)')
    plt.xlabel('Difference in QALYs (CTP - no CTP)')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:  
        if not os.path.exists(root_fig):
            os.makedirs(root_fig)
        plt.savefig(os.path.join(root_fig,'ICER_{}.tiff'.format(name)),dpi=300)
    plt.show()

def plot_outcome_per_threshold(aggrs,
                               name,
                               root_fig=None,
                               colors = ['black','black','black','black']):
    #write funciton for this
    sns.lineplot(data=aggrs,x='threshold',y='d_costs',
                 color=colors[0],ci='sd')
    plt.ylabel('Difference in costs(€) (CTP - no CTP)')
    plt.xlabel('Ischemic core volume decision threshold (mL)')
    #plt.title('Costs')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:
        plt.savefig(os.path.join(root_fig,'Costs_{}.tiff'.format(name)),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='d_qalys',
                 color=colors[1],ci='sd')
    plt.ylabel('Difference in QALYS (CTP - no CTP)')
    plt.xlabel('Ischemic core volume decision threshold (mL)')
    #plt.title('QALY')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:
        plt.savefig(os.path.join(root_fig,'QALYs_{}.tiff'),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='NMB', 
                 color=colors[2],ci='sd')
    plt.xlabel('Ischemic core volume decision threshold (mL)')
    #plt.title('NMB')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:
        plt.savefig(os.path.join(root_fig,'NMB_{}.tiff'.format(name)),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='ICER', 
                 color=colors[3],ci='sd')
    plt.xlabel('Ischemic core volume decision threshold (mL)')
    #plt.title('ICER')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:
        plt.savefig(os.path.join(root_fig,'ICER_thr_{}.tiff'.format(name)),dpi=300)
    plt.show()
    
def M2_detection_plotter(aggr, nni_mult,name='',fig_loc=None,plette='bwr'): #, label_order=None
    aggr['ICER'] = aggr['d_costs']/aggr['d_qalys']
    if fig_loc is not None:
        graggr = aggr[aggr['NNI_multiplier']==nni_mult].groupby('missed_M2')
        out = []
        for outcome in ['ICER','NMB','d_costs','d_qalys']:
            o = graggr[outcome].describe()
            o['outcome'] = outcome
            out.append(o)
        out = pd.concat(out)
        out.to_excel(os.path.join(fig_loc,'Outcomes_{}_nni_{}_results.xlsx'.format(name,nni_mult)))
        
    aggr = aggr[aggr['missed_M2']!=0.37]
    
    # make a more interpretable legend for colors
    or_evt = []
    for e in aggr['evt_shift']:
#         if e==0:
#             or_evt.append('MR CLEAN trial (OR=1.67)')
#         elif e==.82:
#             or_evt.append('HERMES pooling (OR=2.49)')
#         else:
        or_evt.append('OR={}'.format(round(1.67+e,2)))
    aggr['EVT effect'] = or_evt
    aggr = aggr.sort_values(by='evt_shift')
    #plot NMB
    sns.boxplot(data=aggr[aggr['NNI_multiplier']==nni_mult],x='missed_M2',y='NMB',
            hue='EVT effect',
            palette=plette)
    plt.xlabel('Sensitivity gain due to CTP')
    plt.gcf().subplots_adjust(left=0.15)
    # if label_order is not None:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = reorder_labels_ix(labels,label_order)
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    if fig_loc is not None:
        plt.savefig(os.path.join(fig_loc,'NMB_{}_NNI_{}.tiff'.format(name,nni_mult)),dpi=300)
    plt.show()
    
    #plot diff costs
    sns.boxplot(data=aggr[aggr['NNI_multiplier']==nni_mult],x='missed_M2',y='d_costs',
            hue='EVT effect',
            palette=plette)
    plt.xlabel('Sensitivity gain due to CTP')
    plt.ylabel('Difference in costs(€) (CTP - no CTP)')
    plt.gcf().subplots_adjust(left=0.15)
    # if label_order is not None:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = reorder_labels_ix(labels,label_order)
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    if fig_loc is not None:
        plt.savefig(os.path.join(fig_loc,'costs_{}_NNI_{}.tiff'.format(name,nni_mult)),dpi=300)
    plt.show()
    
    #plot diff qalys
    sns.boxplot(data=aggr[aggr['NNI_multiplier']==nni_mult],x='missed_M2',y='d_qalys',
            hue='EVT effect',
            palette=plette)
    plt.xlabel('Sensitivity gain due to CTP')
    plt.ylabel('Difference in QALYs (CTP - no CTP)')
    plt.gcf().subplots_adjust(left=0.15)
    # if label_order is not None:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = reorder_labels_ix(labels,label_order)
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    if fig_loc is not None:
        plt.savefig(os.path.join(fig_loc,'qalys_{}_NNI_{}.tiff'.format(name,nni_mult)),dpi=300)
    plt.show()
    
    #plot diff ICER
    sns.boxplot(data=aggr[(aggr['NNI_multiplier']==nni_mult)&(aggr['missed_M2']!=0.01)],x='missed_M2',y='ICER',
            hue='EVT effect',
            palette=plette)
    plt.xlabel('Sensitivity gain due to CTP')
    plt.gcf().subplots_adjust(left=0.15)
    if nni_mult<100:
        plt.ylim(-.1*1e7,.1*1e7)
    else:
        plt.ylim(0,.4*1e7)
    # if label_order is not None:
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = reorder_labels_ix(labels,label_order)
    #     plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    if fig_loc is not None:
        plt.savefig(os.path.join(fig_loc,'ICER_{}_NNI_{}.tiff'.format(name,nni_mult)),dpi=300)
    plt.show()
