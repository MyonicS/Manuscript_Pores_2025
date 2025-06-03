
import pandas as pd
import numpy as np
import os
#catalyst_database.csv in parent directory
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'catalyst_database.csv')

def set_path(new_path):
    global path
    path = new_path
    print(f"Path updated to: {path}")


def write_db(catalyst,property,value,db_path=path):
    df = pd.read_csv(db_path, sep=',',header=0,index_col=0)
    df.loc[catalyst,property] = value
    df.to_csv(db_path, sep=',',header=True,index=True)

def read_db(catalyst,property,db_path=path):
    df = pd.read_csv(db_path, sep=',',header=0,index_col=0)
    return df.loc[catalyst,property]

def read_unit(property,db_path=path):
    df = pd.read_csv(db_path, sep=',',header=0,index_col=0)
    return df.loc['Unit',property]

def get_db(db_path=path):
    # print(db_path)
    return pd.read_csv(db_path, sep=',',header=0,index_col=0)
    
    

def make_name(cat,db=get_db()):
    if 'F' in cat:
        return f'F$_{{{int(np.round(db["TTBP_IR_BAS (umol/gram)"].loc[cat]))}}}$'
    elif 'crushed' in cat:
        ref = 'M03'
        return cat[0]+f'$_{{{int(np.round(db["TTBP_IR_BAS (umol/gram)"].loc[ref]))}}}$'+' (crushed)'
    else:
        return cat[0]+f'$_{{{int(np.round(db["TTBP_IR_BAS (umol/gram)"].loc[cat]))}}}$'


def get_cat(filename):
    no_csv = filename.split('.')[0]
    return no_csv.split('_')[-1]