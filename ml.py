import statistics

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from tools import symptoms

groups = ['ABC', 'A', 'B', 'C']

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score)
}


def xgboost_algorithm(days):
    output_filename = f'results/ml_results_{days}day(s).csv'

    depr_groups_out = []
    symptoms_out = []
    number_features_out = []
    accuracies_out = []
    precisions_out = []
    recalls_out = []
    f1s_out = []

    for symptom in symptoms:
        print(f'Symptom: {symptom}')
        for group in groups:
            print(f'Depression group: {group}')
            dataset = pd.read_csv(f'symptoms/{group}/{days}day(s)/{symptom}_features_{days}day(s)_{group}.csv')
            X = dataset.iloc[:, 17:].values
            y = dataset.iloc[:, 5].values
            logo_groups = dataset['user_id']
            logo = LeaveOneGroupOut()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=44)
            model.fit(X_train, y_train)

            for i in range(1, len(X)):
                accuracies_inner = []
                precisions_inner = []
                recalls_inner = []
                f1s_inner = []
                for train_index, test_index in logo.split(X, y, logo_groups):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    selection = SelectFromModel(model, prefit=True, max_features=i)
                    select_X_train = selection.transform(X_train)

                    # train model
                    selection_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    selection_model.fit(select_X_train, y_train)
                    # eval model
                    select_X_test = selection.transform(X_test)
                    y_pred = selection_model.predict(select_X_test)

                    accuracy_res = accuracy_score(y_test, y_pred)
                    precision_res = precision_score(y_test, y_pred, zero_division=0)
                    recall_res = recall_score(y_test, y_pred, zero_division=0)
                    f1_res = f1_score(y_test, y_pred, zero_division=0)

                    accuracies_inner.append(accuracy_res)
                    precisions_inner.append(precision_res)
                    recalls_inner.append(recall_res)
                    f1s_inner.append(f1_res)

                mean_accuracy = statistics.mean(accuracies_inner)
                mean_precision = statistics.mean(precisions_inner)
                mean_recall = statistics.mean(recalls_inner)
                mean_f1 = statistics.mean(f1s_inner)

                print(
                    f'n={i}: accuracy={mean_accuracy}, precision={mean_precision}, recall={mean_recall}, f1={mean_f1}')
                number_features_out.append(i)
                accuracies_out.append(mean_accuracy)
                recalls_out.append(mean_recall)
                precisions_out.append(mean_precision)
                f1s_out.append(mean_f1)
                depr_groups_out.append(group)
                symptoms_out.append(symptom)

    df_out = pd.DataFrame()
    df_out['depr_group'] = depr_groups_out
    df_out['symptom'] = symptoms_out
    df_out['features_num'] = number_features_out
    df_out['accuracy'] = accuracies_out
    df_out['precision'] = precisions_out
    df_out['recall'] = recalls_out
    df_out['f1'] = f1s_out
    df_out['window_size'] = days

    df_out.to_csv(output_filename, index=False)
