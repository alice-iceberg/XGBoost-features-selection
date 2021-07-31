import pandas as pd

symptoms = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']
groups = ['A', 'B', 'C']
general_columns = ['user_id', 'timestamp', 'symptom', 'symptom_score', 'depr_group',
                   'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
                   'add_gender', 'add_age', 'binary_label']


def split_all_features_combined_file_to_symptom_files(days):
    df = pd.read_csv(f'combined/combined_features_symptoms_as_rows_{days}day(s).csv')

    for symptom in symptoms:
        df_symptom = df.query(f'symptom=="{symptom}"')
        df_symptom.to_csv(f'symptoms/ABC/{days}day(s)/{symptom}_features_{days}day(s)_ABC.csv',
                          index=False)
        for group in groups:
            df_group = df_symptom.query(f'depr_group=="{group}"')
            df_group.to_csv(
                f'symptoms/{group}/{days}day(s)/{symptom}_features_{days}day(s)_{group}.csv',
                index=False)


def add_binary_label_column(days):
    df = pd.read_csv(f'combined/combined_features_symptoms_as_rows_{days}day(s).csv')
    binary_labels = []

    for row in df.itertuples():
        if row.symptom_score == 1:
            binary_labels.append(0)
        else:
            binary_labels.append(1)

    df['binary_label'] = binary_labels

    df.to_csv(f'combined/combined_features_symptoms_as_rows_{days}day(s).csv', index=False)


def ema_symptoms_as_rows(days):
    filename_out = f'ema/ema_symptoms_as_rows_{days}day(s).csv'
    df = pd.read_csv(f'ema/ema_average_{days}day(s).csv')
    user_ids_out = []
    timestamps_out = []
    symptoms_out = []
    symptom_scores_out = []
    depr_groups_out = []
    binary_labels_out = []

    for row in df.itertuples():
        for symptom in symptoms:
            user_ids_out.append(row.user_id)
            timestamps_out.append(row.timestamp)
            symptoms_out.append(symptom)
            depr_groups_out.append(row.depr_group)
            if symptom == symptoms[0]:
                symptom_scores_out.append(row.phq1)
                if row.phq1 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[1]:
                symptom_scores_out.append(row.phq2)
                if row.phq2 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[2]:
                symptom_scores_out.append(row.phq3)
                if row.phq3 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[3]:
                symptom_scores_out.append(row.phq4)
                if row.phq4 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[4]:
                symptom_scores_out.append(row.phq5)
                if row.phq5 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[5]:
                symptom_scores_out.append(row.phq6)
                if row.phq6 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[6]:
                symptom_scores_out.append(row.phq7)
                if row.phq7 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[7]:
                symptom_scores_out.append(row.phq8)
                if row.phq8 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
            elif symptom == symptoms[8]:
                symptom_scores_out.append(row.phq9)
                if row.phq9 == 1:
                    binary_labels_out.append(0)
                else:
                    binary_labels_out.append(1)
    df_out = pd.DataFrame()
    df_out['user_id'] = user_ids_out
    df_out['timestamp'] = timestamps_out
    df_out['symptom'] = symptoms_out
    df_out['symptom_score'] = symptom_scores_out
    df_out['depr_group'] = depr_groups_out
    df_out['binary_label'] = binary_labels_out

    df_out.to_csv(filename_out, index=False)


def combined_features_symptoms_as_rows(days):
    out_filename = f'combined/combined_features_symptoms_as_rows_{days}day(s).csv'
    df_data = pd.read_csv(f'combined/combined_processed_features_{days}day(s).csv')
    df_ema = pd.read_csv(f'ema/ema_symptoms_as_rows_{days}day(s).csv')

    df_out = pd.merge(df_ema, df_data, on=['user_id', 'timestamp', 'depr_group'])
    df_out.to_csv(out_filename, index=False)





