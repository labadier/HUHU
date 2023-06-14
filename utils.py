#%%
class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class params:

  MODEL = 'Tbert-base-multilingual-cased'
  PHASE = 'train'  
  MODE = 'transformer'
  EPOCHES = 10
  BS = 32
  LR = 1e-5
  DECAY = 2e-5
  MULTIPLIER = 1
  INCREASE = 0.1
  IL = 64
  OUTPUT = 'output'

if __name__ == '__main__':
 
  #%%
  import pandas as pd
  from glob import glob

  ls = glob('../final_annotation_step/*.xlsx')
  df = None
  for l in ls:
    if df is None:
      df = pd.read_excel(l).fillna(0)
    else:
      df = pd.concat([df, pd.read_excel(l).fillna(0)], axis=0)

  df_train = None
  df_test = None


  to_take = ['prejudice_woman', 'prejudice_lgbtiq', 'prejudice_inmigrant_race', 'gordofobia']
  taken = {i:0 for i in to_take}
  total = {i:0 for i in to_take}
  to_take.sort(key= lambda x: sum(df[x].to_list()))


  for i in range(len(to_take)):

    mask = df[(df[to_take[i]] == 1)&(df['humor'] == 1) ]
    total[to_take[i]] = len(mask)
    mask_sorted = mask.assign(f = sum(mask[i] for i in to_take)).sort_values('f', ascending=False).drop('f', axis=1)
    selected = mask_sorted.iloc[:int((total[to_take[i]] - taken[to_take[i]])*0.8)]

    for j in range(1, len(selected)):
      for k in to_take:
        if selected.iloc[j][k] == 1:
          taken[k] += 1
      
    df_train = pd.concat([df_train, selected], axis=0) if df_train is not None else selected
    df = df[~df['index'].isin(mask_sorted['index'])]
    mask_sorted = mask_sorted[~mask_sorted['index'].isin(selected['index'])]
    df_test = pd.concat([df_test, mask_sorted], axis=0) if df_test is not None else mask_sorted
  #%%
  df = df[df['humor'] != 1]
  # %%

  taken = {i:0 for i in to_take}
  total = {i:0 for i in to_take}
  to_take.sort(key= lambda x: sum(df[x].to_list()))

  for i in range(len(to_take)):

    mask = df[df[to_take[i]] == 1 ]
    total[to_take[i]] = len(mask)
    mask_sorted = mask.assign(f = sum(mask[i] for i in to_take)).sort_values('f', ascending=False).drop('f', axis=1)
    selected = mask_sorted.iloc[:int((total[to_take[i]] - taken[to_take[i]])*0.8)]

    for j in range(1, len(selected)):
      for k in to_take:
        if selected.iloc[j][k] == 1:
          taken[k] += 1
      
    df_train = pd.concat([df_train, selected], axis=0) if df_train is not None else selected
    df = df[~df['index'].isin(mask_sorted['index'])]
    mask_sorted = mask_sorted[~mask_sorted['index'].isin(selected['index'])]
    df_test = pd.concat([df_test, mask_sorted], axis=0) if df_test is not None else mask_sorted

  # %%
  df_train.to_csv('../data/train.csv', index=False)
  df_test.to_csv('../data/test.csv', index=False)


# %%
