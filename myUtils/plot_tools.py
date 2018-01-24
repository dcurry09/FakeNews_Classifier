##########################################
#  Plot_tools.py
#
#  Useful Plots for Pandas datframes
#
# By David Curry,  Oct 2017
#
##########################################

import csv as csv
import numpy as np
import pandas as pd
import pylab as py
import sys
import matplotlib.pyplot as plt
 

def TwoLists_CompareBar(x_pos, Label, value, value2):
    '''
    Plots two lists as bars for comparison.
    returns matplot figure
    '''

    fig = plt.figure(figsize=(22,12))
    plt.bar(x_pos, value, align='center', alpha=0.5, label='Fake')
    plt.bar(x_pos, value2, align='center', alpha=0.5, label='Real')
    plt.xticks(x_pos, Label) 
    plt.legend()
    plt.ylabel('Frequency')
    plt.show()
    


    
    
