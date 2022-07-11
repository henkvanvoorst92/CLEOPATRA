import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys 
from tqdm import tqdm

def ICER_plot(aggrs, name, root_fig=None, min_x=-.15,max_x =.15, min_y=-12000, max_y=12000):

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
    y = 80000*x
    sns.lineplot(x,np.zeros_like(x),color='black',linewidth=1)
    sns.lineplot(np.zeros_like(y_b),y_b,color='black',linewidth=1,estimator=None)
    sns.scatterplot(data=dataplot,x='d_qalys',y='d_costs', hue='group')
    sns.lineplot(x,y,color='black')
    plt.ylim(min_y, max_y)
    plt.xlim(min_x, max_x)
    plt.ylabel('Difference Costs(€) (CTP-noCTP)')
    plt.xlabel('Difference QALYS (CTP-noCTP)')
    plt.gcf().subplots_adjust(left=0.15)
    if root_fig is not None:  
        if not os.path.exists(root_fig):
            os.makedirs(root_fig)
        plt.savefig(os.path.join(root_fig,'ICER_{}.tiff'.format(name)),dpi=300)
    plt.show()

def plot_outcome_per_threshold(aggrs,
                               root_fig,
                               name):
    #write funciton for this
    sns.lineplot(data=aggrs,x='threshold',y='d_costs',color='blue')
    plt.ylabel('Difference Costs(€) (CTP-noCTP)')
    plt.xlabel('Core volume decision threshold (mL)')
    plt.title('Costs')
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join(root_fig,'Costs_{}.tiff'.format(name)),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='d_qalys',color='red')
    plt.ylabel('Difference QALYS (CTP-noCTP)')
    plt.xlabel('Core volume decision threshold (mL)')
    plt.title('QALY')
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join(root_fig,'QALYs_{}.tiff'),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='NMB', color='purple')
    plt.xlabel('Core volume decision threshold (mL)')
    plt.title('NMB')
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join(root_fig,'NMB_{}.tiff'.format(name)),dpi=300)
    plt.show()

    sns.lineplot(data=aggrs,x='threshold',y='ICER', color='green')
    plt.xlabel('Core volume decision threshold (mL)')
    plt.title('ICER')
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig(os.path.join(root_fig,'ICER_thr_{}.tiff'.format(name)),dpi=300)
    plt.show()


