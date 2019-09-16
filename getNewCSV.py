import pandas as pd
import numpy as np
import sys

dont_want = ['CHARGE_CITY_CD', 'CONTACT_CITY_CD']
not_binary = ['AGE','L1YR_A_ISSUE_CNT','L1YR_B_ISSUE_CNT','CHANNEL_A_POL_CNT','CHANNEL_B_POL_CNT','APC_CNT','INSD_CNT','LIFE_CNT','AG_CNT','AG_NOW_CNT','CLC_CUR_NUM','BANK_NUMBER_CNT','IM_CNT','TOOL_VISIT_1YEAR_CNT','LIFE_INSD_CNT','L1YR_GROSS_PRE_AMT','CUST_9_SEGMENTS_CD']

def genNewCsv(oldName, newName):
    df = pd.read_csv(oldName, encoding='Big5').dropna(axis=1)
    """
    df = df.drop(columns=dont_want)
    """
    df.to_csv(newName, index=False)
    return df

df = genNewCsv('train.csv', 'newtrain.csv')
print('train\ndataNum: {}   featureNum: {}'.format(len(df), len(df.columns)))
df = genNewCsv('test.csv', 'newtest.csv')
print(' test\ndataNum: {}   featureNum: {}'.format(len(df), len(df.columns)))
