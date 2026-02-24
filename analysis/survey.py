#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import prince
from pandas_plink import read_plink
from semopy.polycorr import polychoric_corr, estimate_intervals
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt


# In[ ]:


my_bucket = os.getenv('WORKSPACE_BUCKET')
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
fig_root = Path("figs/ukb/ohe_sdoh_data_unhot")


# In[ ]:


output_dir = Path("ohe_sdoh_data_unhot")
output_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


split_onehot_doh = pd.read_csv("ohe_sdoh_survey_files/survey/split_onehot_doh.csv")
split_onehot_doh = split_onehot_doh.drop(columns=['Unnamed: 0'])
split_onehot_doh.head()


# In[ ]:


ordered_options = {
    ('Detached single-family housing', 'Mix of single-family residences and townhouses', 'Townhouses', 'Apartments or condos of 4-12 stories', 'Apartments or condos of more than 12 stories'), # from single-family to multi-family
    ('Strongly disagree', 'Somewhat disagree', 'Somewhat agree', 'Strongly agree'),
    ('Never true', 'Sometimes true', 'Often true'),
    ('Never', 'Almost Never', 'Sometimes', 'Fairly Often', 'Very Often'),
    ('I am not religious', 'Never or almost never', 'Less than once per month', '1 to 3 times per month', 'Once a week', 'More than once a week'),
    ('I am not religious', 'Never or almost never', 'Once in a while', 'Some days', 'Most days', 'Every day', 'Many times a day'),
    ('None of the time', 'A little of the time', 'Some of the time', 'Most of the time', 'All of the time'),
    ('Never', 'Less than once a year', 'A few times a year', 'A few times a month', 'At least once a week', 'Almost everyday'),
    ('Never', 'Rarely', 'Sometimes', 'Often'),
    ('Never', 'Rarely', 'Sometimes', 'Most of the time', 'Always'),
    # This is probably a little controversial
    ('Strongly disagree', 'Somewhat disagree', 'Does not apply to my neighborhood', 'Somewhat agree', 'Strongly agree'),
    ('I do not believe in God (or a higher power)', 'Never or almost never', 'Once in a while', 'Some days', 'Most days', 'Every day', 'Many times a day'),
    ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'),
    ('No', 'Yes'),
    ('Never or almost never', 'Once in a while', 'Some days', 'Most days', 'Every day', 'Many times a day'),
    ('Strongly disagree', 'Disagree', 'Agree', 'Strongly agree'),
    ('Not at all', 'Not well', 'Well', 'Very well'),
    ('Strongly disagree', 'Disagree', 'Neutral (neither agree nor disagree)', 'Agree', 'Strongly agree'),
}

unordered_options = {
    frozenset(('Bug infestation', 'Inadequate heat', 'Lead paint or pipes', 'Mold', 'No or not working smoke detector', 'Oven or stove not working', 'Water leaks', 'None of the above')),
    frozenset(('Your age', 'Your ancestry or national origins', 'Your education or income level', 'Your gender', 'Your height', 'Your race', 'Your religion', 'Your sexual orientation', 'Your weight', 'Some other aspect of your physical appearance', 'Other (specify)')),
}


# In[ ]:


questions_map = {}
for col in split_onehot_doh.columns:
    if col == 'person_id':
        continue
    question, category = col.split("_")
    if question not in questions_map:
        questions_map[question] = set()
    questions_map[question].add(category)

questions_to_convert = {}
processed_cols = set()
for question, categories_set in questions_map.items():
    for option_tuple in {*ordered_options, *unordered_options}:
        if set(option_tuple) == categories_set:
            onehot_cols = [f"{question}_{cat}" for cat in option_tuple]
            questions_to_convert[question] = (option_tuple in ordered_options, option_tuple, onehot_cols)
            processed_cols.update(onehot_cols)
            break
    else:
        raise ValueError(f"Unknown option tuple: {option_tuple}")

person_ids = split_onehot_doh['person_id']
result_data = {
    'person_id': person_ids
}

for question, (ordered, option_tuple, onehot_cols) in questions_to_convert.items():
    onehot_array = split_onehot_doh[onehot_cols].values.astype(int)
    row_sums = onehot_array.sum(axis=1)

    if not np.all((row_sums == 0) | (row_sums == 1)):
        invalid_mask = (row_sums != 0) & (row_sums != 1)
        invalid_rows = np.where(invalid_mask)[0]
        raise ValueError(
            f"Question '{question}' has invalid data at rows {invalid_rows[:10].tolist()}... "
            f"with sums {row_sums[invalid_rows][:10].tolist()}. "
            f"Expected sum of 0 or 1."
        )

    max_indices = onehot_array.argmax(axis=1)
    cats = list(option_tuple)
    cat_values = np.where(row_sums == 1, 
                         [cats[idx] for idx in max_indices], 
                         np.nan)

    result_data[question] = pd.Categorical(cat_values, categories=cats, ordered=ordered)
    
remaining_cols = [col for col in split_onehot_doh.columns 
                 if col not in processed_cols and col != 'person_id']
assert not remaining_cols

split_unhot_doh = pd.DataFrame(result_data)
split_unhot_doh.to_parquet(output_dir / 'split_unhot_doh.parquet')
split_unhot_doh.to_csv(output_dir / 'split_unhot_doh.csv')
split_unhot_doh.head()


# In[ ]:




