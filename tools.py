import pandas as pd

symptoms = ['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']
groups = ['A', 'B', 'C', 'ABC']
general_columns = ['user_id', 'timestamp', 'symptom', 'symptom_score', 'depr_group',
                   'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9',
                   'add_gender', 'add_age', 'binary_label']

users_remove = [82, 84, 99, 107, 115, 122, 135, 155, 159, 170]


def split_all_features_combined_file_to_symptom_files(days):
    df = pd.read_csv(f'combined/combined_features_symptoms_as_rows_{days}day(s).csv')

    #  region selected data 14 days
    # df = df[~df.user_id.isin(users_remove)]
    # df = df.drop(df[(df['user_id'] == 85) & (df['timestamp'] == 1608735599000)].index)
    # df = df.drop(df[(df['user_id'] == 118) & (df['timestamp'] == 1607785199000)].index)
    # df = df.drop(df[(df['user_id'] == 128) & (df['timestamp'] == 1609685999000)].index)
    # df = df.drop(df[(df['user_id'] == 131) & (df['timestamp'] == 1609081199000)].index)
    # df = df.drop(df[(df['user_id'] == 132) & (df['timestamp'] == 1609772399000)].index)
    # df = df.drop(df[(df['user_id'] == 119) & (df['timestamp'] == 1608562799000)].index)
    # df = df.drop(df[(df['user_id'] == 119) & (df['timestamp'] == 1609772399000)].index)
    # df = df.drop(df[(df['user_id'] == 126) & (df['timestamp'] == 1608735599000)].index)
    # df = df.drop(df[(df['user_id'] == 126) & (df['timestamp'] == 1609945199000)].index)
    # df = df.drop(df[(df['user_id'] == 153) & (df['timestamp'] == 1609772399000)].index)
    # df = df.drop(df[(df['user_id'] == 169) & (df['timestamp'] == 1611413999000)].index)

    #  endregion

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


def add_3classes_label_column(days):
    for group in groups:
        for symptom in symptoms:
            filename = f'symptoms/{group}/{days}day(s)/{symptom}_features_14day(s)_{group}_sel.csv'
            three_classes = []
            df = pd.read_csv(filename)
            for row in df.itertuples():
                if row.symptom_score == 1:
                    three_classes.append(0)
                elif row.symptom_score == 2 or row.symptom_score == 3:
                    three_classes.append(1)
                elif row.symptom_score == 4 or row.symptom_score == 5:
                    three_classes.append(2)
            df['three_labels'] = three_classes
            df.to_csv(filename, index=False)


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


def concat_ml_results(days, classes):
    output_filename = f'results/{days}days/{classes}/ml_results_{days}day(s)_selected_{classes}.csv'
    frames = []
    for group in groups:
        for symptom in symptoms:
            df = pd.read_csv(f'results/{days}days/{classes}/ml_results_{days}day(s)_{symptom}_{group}_selected.csv')
            frames.append(df)

    df_out = pd.concat(frames)
    df_out = df_out.sort_values(by=['depr_group', 'symptom', 'features_num'])
    df_out.to_csv(output_filename, index=False)

