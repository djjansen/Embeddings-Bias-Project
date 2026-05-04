import plotly.express as px

def plot_mean_profession_scores_for_model(df, model_score_col, model_name, prof_gender):
    fig = px.histogram(df.loc[df['Prof_Gender'] == prof_gender], x="Profession", y=model_score_col,
                color='Gender', barmode='group',
                height=400, title=f"{model_name} Mean Scores for {prof_gender.title()} Professions")

    fig.show()


def plot_mean_prof_gender_scores(df, prof_gender):
    fig = px.histogram(df.loc[df['Prof_Gender'] == prof_gender], x="Model", y="Score",
                color='Gender', barmode='group',
                height=400, title=f"Mean Scores by Model and Target Gender for {prof_gender.title()} Professions")

    fig.show()