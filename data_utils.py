import pandas as pd
import numpy as np


# load from path and filter data to target words of interest
def load_bec_data(path):
    df = pd.read_csv(path, delimiter='\t')

    return df \
    .loc[df['Person'].isin(['He','She'])] \
    .sort_values(by='Profession')


# create df with values replaced, e.g., non-binary pronouns instead of male ones
def synthesize_bec_data(df, target_word, target_gender):
    df['Person'] = target_word
    df['Gender'] = target_gender

    return df \
    .sort_values(by='Profession')

# return profession + gender-level average scores
def get_profession_mean_scores(df):
    score_cols = [col for col in df.columns.tolist() if 'score' in col]
    return df[['Profession', 'Prof_Gender', 'Gender'] + score_cols] \
    .groupby(['Profession', 'Prof_Gender', 'Gender']).mean() \
    .reset_index()

# # get average scores at the gender + profession gender level (e.g., Male pronouns, Female professions)
def get_prof_gender_mean_scores(df):
    score_cols = [col for col in df.columns.tolist() if 'score' in col]
    df = df[['Prof_Gender', 'Gender'] + score_cols] \
    .groupby(['Prof_Gender', 'Gender']).mean() \
    .reset_index()

    # re-format as long for charts
    df_long = pd.melt(
    df, 
    id_vars=['Prof_Gender', 'Gender'], 
    value_vars=['bert_score', 'gpt4_1_nano_score', 'gpt5_4_score', 'llama_3_1_8b_score'],
    var_name='Model',  
    value_name='Score' 
    )

    # clean up model names
    df_long['Model'] = df_long['Model'].str.replace('_score', '').str.replace('_', ' ').str.title()

    return df_long