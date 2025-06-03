from matplotlib import pyplot as plt
import numpy as np
import platform

def set_plot_defaults():
    system = platform.system()
    
    if system == 'Windows':
        font_family = 'Arial'
    elif system == 'Linux':
        font_family = 'Liberation Sans'
    else:
        font_family = 'sans-serif'  # Fallback
    
    font = {
        'family': font_family,
        'weight': 'normal',
        'size': 7
    }
    
    plt.rc('font', **font)
    plt.rcParams['figure.figsize'] = [8.8 / 2.54, 6.22 / 2.54]


def make_name(cat,db):
    if 'F' in cat:
        return f'F$_{{{int(np.round(db["TTBP_IR_BAS (umol/gram)"].loc[cat]))}}}$'
    else:
        return cat[0]+f'$_{{{int(np.round(db["TTBP_IR_BAS (umol/gram)"].loc[cat]))}}}$'
    