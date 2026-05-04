import pandas as pd
import numpy as np

def load_bec_data(path):
    df = pd.read_csv(path, delimiter='\t')
    
    return df \
    .loc[df['Person'].isin(['He','She'])] \
    .sort_values(by='Profession')


def synthesize_bec_data(df, target_word, target_gender):
    df['Person'] = target_word
    df['Gender'] = target_gender

    return df \
    .sort_values(by='Profession')



def add_scores(func, df):
    target_probabilities = []
    prior_probabilities = []
    scores = []

    for row in df.iloc:  # remember, anything with an index is iterable
        target_probabilities.append( func(row['Sent_TM'], target=row['Person']) )
        prior_probabilities.append( func(row['Sent_TAM'], target=row['Person']) )
        scores.append( np.log( np.exp(target_probabilities[-1]) / np.exp(prior_probabilities[-1]) ) )

    df['scores'] = scores

    return df


def get_gender_scores(df):
    score_cols = [col for col in df.columns.tolist() if 'score' in col]
    return df[['Profession', 'Prof_Gender', 'Gender'] + score_cols] \
    .groupby(['Profession', 'Prof_Gender', 'Gender']).mean() \
    .reset_index()
8