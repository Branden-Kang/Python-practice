import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import base64
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA


@st.cache
def load_data(filename=None):
    filename_default = 'training.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    df = df.dropna(axis=0, how='all')
    df.time = [datetime.datetime.strptime(
        x, '%m/%d/%Y %H:%M') for x in df.time]
    return df, df.shape[0], df.shape[1], filename


def drop_useless_columns(df, manual_drop_list=None):
    if manual_drop_list:
        return df.drop(columns=manual_drop_list), manual_drop_list

    drop_list = []
    for col in df.columns:
        if df[col].nunique() <= 1 or df[col].nunique() >= df.shape[0] * 0.95:
            drop_list.append(col)
    return df.drop(columns=drop_list), drop_list


def datetime_processing(df):
    if 'time' not in df.columns:
        return df

    df['weekday'] = [x.weekday() for x in df.time]
    df['hour'] = [int(x.strftime('%H')) for x in df.time]
    max_time = df.time.max()
    min_time = df.time.min()
    min_norm, max_norm = -1, 1
    df['date'] = (df.time - min_time) * (max_norm - min_norm) / \
        (max_time - min_time) + min_norm
    return df


def data_preprocessing(df, manual_drop_list=None):
    """Return processed data and column list that need to be dropped."""

    df, drop_list = drop_useless_columns(df, manual_drop_list)

    df = datetime_processing(df)

    df.C2 = df.C1 + df.C3
    df = df.drop(columns=['time', 'X'])
    df['sum_X'] = df.iloc[:, 9:33].sum(axis=1)

    return df.replace({'status': {'Approved': 0, 'Declined': 1}}), drop_list


def feature_summary(data):
    print('DataFrame shape')
    print('rows:', data.shape[0])
    print('cols:', data.shape[1])
    col_list = ['Null', 'Unique_Count', 'Data_type',
                'Max/Min', 'Mean', 'Std', 'Skewness', 'Sample_values']
    df = pd.DataFrame(index=data.columns, columns=col_list)
    df['Null'] = list([len(data[col][data[col].isnull()])
                       for i, col in enumerate(data.columns)])
    df['Unique_Count'] = list([len(data[col].unique())
                               for i, col in enumerate(data.columns)])
    df['Data_type'] = list(
        [data[col].dtype for i, col in enumerate(data.columns)])
    for i, col in enumerate(data.columns):
        if 'float' in str(data[col].dtype) or 'int' in str(data[col].dtype):
            df.at[col, 'Max/Min'] = str(round(data[col].max(), 2)) + \
                '/' + str(round(data[col].min(), 2))
            df.at[col, 'Mean'] = data[col].mean()
            df.at[col, 'Std'] = data[col].std()
            df.at[col, 'Skewness'] = data[col].skew()
        df.at[col, 'Sample_values'] = list(data[col].unique())

    return(df.fillna('-'))


@st.cache(allow_output_mutation=True)
def XGB_train_metrics(df, params_set):
    """Return metrics and model for XGB."""

    X = df.drop(columns=['status'])
    Y = df.status
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    # Fit model
    model_xgb = XGBClassifier(max_depth=params_set[0], eta=params_set[1], min_child_weight=params_set[2],
                              subsample=params_set[3], colsample_bylevel=params_set[4], colsample_bytree=params_set[5])
    # model = XGBClassifier()
    model_xgb.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model_xgb.predict(X_test)

    # Evaluate predictions
    accuracy_xgb = accuracy_score(y_test, y_pred)
    f1_xgb = f1_score(y_test, y_pred)
    roc_auc_xgb = roc_auc_score(y_test, y_pred)
    recall_xgb = recall_score(y_test, y_pred)
    precision_xgb = precision_score(y_test, y_pred)
    return accuracy_xgb, f1_xgb, roc_auc_xgb, recall_xgb, precision_xgb, model_xgb


@st.cache(suppress_st_warning=True)
def logistic_train_metrics(df):
    """Return metrics and model for Logistic Regression."""

    X = df.drop(columns=['status'])
    Y = df.status

    std_scaler = StandardScaler()
    std_scaled_df = std_scaler.fit_transform(X)
    std_scaled_df = pd.DataFrame(std_scaled_df, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        std_scaled_df, Y, random_state=0)

    # Fit model
    model_reg = LogisticRegression(max_iter=1000)
    model_reg.fit(X_train.fillna(0), y_train)

    # Make predictions for test data
    y_pred = model_reg.predict(X_test.fillna(0))

    # Evaluate predictions
    accuracy_reg = accuracy_score(y_test, y_pred)
    f1_reg = f1_score(y_test, y_pred)
    roc_auc_reg = roc_auc_score(y_test, y_pred)
    recall_reg = recall_score(y_test, y_pred)
    precision_reg = precision_score(y_test, y_pred)

    return accuracy_reg, f1_reg, roc_auc_reg, recall_reg, precision_reg, model_reg


def decision_boundary(df):
    """Plot decision boundary for Logistic Regression."""

    X = df.drop(columns=['status'])
    Y = df.status

    # Scaling
    std_scaler = StandardScaler()
    std_scaled_df = std_scaler.fit_transform(X)
    std_scaled_df = pd.DataFrame(std_scaled_df, columns=X.columns)
    X_train, _, y_train, _ = train_test_split(std_scaled_df, Y, random_state=0)

    # Projection to 2d from 47d
    pca = PCA(n_components=2)
    pca.fit(X_train.fillna(0))
    pca_train = pca.transform(X_train.fillna(0))
    df_train_pca = pd.DataFrame(data=pca_train, columns=['pca-1', 'pca-2'])

    model = LogisticRegression(max_iter=1000)
    model.fit(df_train_pca.fillna(0), y_train)
    plot_decision_regions(df_train_pca.values, y_train.values,
                          clf=model, res=0.02, zoom_factor=5)
    st.pyplot()


def upload_different_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    rows = df.shape[0]
    columns = df.shape[1]

    # Drop rows with all Null
    df = df.dropna(axis=0, how='all')
    df.time = [datetime.datetime.strptime(
        x, '%m/%d/%Y %H:%M') for x in df.time]
    data, drop_list = data_preprocessing(df)
    return df, data, drop_list, 'Uploaded file', rows, columns


def home_page_builder(df, data, rows, columns):
    st.title("Streamlit Demo")
    st.write('')
    st.write('')
    st.subheader('INTRODUCTION')
    st.write('')
    st.write(
        'Using machine learning algorithms to predict approval status of application')
    st.write('')
    st.write('')

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Show Data'):
        st.subheader("Raw data")
        st.write(
            f'Input dataset includes **{rows}** rows and **{columns}** columns')
        st.write(df.head())
        st.subheader('Processed data')
        st.write(
            f'After Pre-processing the data for modeling, dataset includes **{data.shape[0]}** rows and **{data.shape[1]}** columns')
        st.write(data.head())

    # show data visulization
    if st.checkbox('Show Visualization'):
        fig = px.histogram(df.status, x='status',
                           title='Distribution of Target Variable "status"')
        st.plotly_chart(fig)
        st.write('We can see Approved is about three times of Decliened, which may bring an imbalanced issue for prediction - we will deal with this issue during modeling.')
        st.write('-'*60)
        fig = px.histogram(df.time, x='time',
                           title='Distribution of Date Time')
        st.plotly_chart(fig)
        st.write(
            'The distribution of date time, we can see most of the data are from recent two months.')
        st.write('-'*60)

    # Show feature summary
    if st.checkbox('Show Feature Summary'):
        st.write('Raw data after dropping rows that have NULL for every column; ')
        st.write('Also converted column "time" to datetime format')
        st.write(feature_summary(df))
        st.write('For each columns in our original dataset, we can see the statistics summary (Null Value Count, Unique Value Count, Data Type, etc.)')


def logistic_page_builder(data):
    start_time = datetime.datetime.now()
    accuracy_reg, f1_reg, roc_auc_reg, recall_reg, precision_reg, model_reg = logistic_train_metrics(
        data)
    st.subheader('Model Introduction')
    st.write('')
    st.write('Logistic Regression is a very popular Linear classification model, people usually use it as a baseline model and build the decision boundary.')
    st.write('See more from Wiki: https://en.wikipedia.org/wiki/Logistic_regression')
    st.write('')
    st.subheader('Logistic Regression metrics on testing dataset')
    st.write('')
    st.write('')
    st.write('')
    st.markdown("We separated the dataset to training and testing dataset, using training data to train our model then do the prediction on testing dataset, here's Logistic Regression prediction performance: ")
    st.write('')
    st.write(
        f'Running time: {(datetime.datetime.now() - start_time).seconds} s')
    st.table(pd.DataFrame(data=[round(accuracy_reg * 100.0, 2), round(precision_reg * 100.0, 2),
                                round(recall_reg*100, 2), round(roc_auc_reg*100, 2), round(f1_reg*100, 2)],
                          index=['Accuracy', 'Precision (% we predicted as Declined are truly Declined)',
                                 'Recall (% Declined have been identified)', 'ROC_AUC', 'F1'],
                          columns=['%']))
    st.write('Decision Boundary after dimension reduction:')
    decision_boundary(data)
    return model_reg


def logistic_predictor(model_reg, rows, columns, df, drop_list):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.text('This process probably takes few seconds...')
    st.write('Note: Currently, the CSV file should have **exactly the same** format with **training dataset**:', df.head(2))
    st.write(
        f'Training dataset includes **{rows}** rows and **{columns}** columns')
    st.write('')

    if uploaded_file:
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.write('-'*80)
        st.write('Uploaded data:', data.head(30))
        st.write(
            f'Uploaded data includes **{data.shape[0]}** rows and **{data.shape[1]}** columns')

        start_time = datetime.datetime.now()
        data = data.dropna(axis=0, how='all')
        data.time = [datetime.datetime.strptime(
            x, '%m/%d/%Y %H:%M') for x in data.time]
        data, _ = data_preprocessing(data, drop_list)
        data2 = data.copy()
        X = data.drop(columns=['status'])

        prediction = model_reg.predict(X)
        prediction_time = (datetime.datetime.now() - start_time).seconds
        data['status'] = ['Approved' if i ==
                          0 else 'Declined' for i in prediction]
        st.write('')
        st.write('-'*80)
        st.write('Prediction:')
        st.write(data.head(30))
        st.text(f'Running time: {prediction_time} s')
        st.write('')

        accuracy_pending = accuracy_score(data2.status, prediction)
        f1_pending = f1_score(data2.status, prediction)
        roc_auc_pending = roc_auc_score(data2.status, prediction)
        recall_pending = recall_score(data2.status, prediction)
        precision_pending = precision_score(data2.status, prediction)
        st.write('Metrics on uploaded data:')
        st.text("Note: This is only temporary since new data won't have labels")
        st.write('Accuracy:', round(100*accuracy_pending, 2), '%')
        st.write('Precision:', round(100*precision_pending, 2), '%')
        st.write('Recall:', round(100*recall_pending, 2), '%')
        st.write('ROC AUC:', round(100*roc_auc_pending, 2), '%')
        st.write('F1:', round(100*f1_pending, 2), '%')

        # Download the prediction as a CSV file
        prediction_downloader(data)


def prediction_downloader(data):
    st.write('')
    st.subheader('Want to download the prediction results?')
    csv = data.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)


def xgb_page_builder(data):
    st.sidebar.header('Hyper Parameters')
    st.sidebar.markdown('You can tune the hyper parameters by siding')
    max_depth = st.sidebar.slider('Select max_depth (default = 30)', 3, 30, 30)
    eta = st.sidebar.slider(
        'Select learning rate (divided by 10) (default = 0.1)', 0.01, 1.0, 1.0)
    min_child_weight = st.sidebar.slider(
        'Select min_child_weight (default = 0.3)', 0.1, 3.0, 0.3)
    subsample = st.sidebar.slider(
        'Select subsample (default = 0.75)', 0.5, 1.0, 0.75)
    colsample_bylevel = st.sidebar.slider(
        'Select colsample_bylevel (default = 0.5)', 0.5, 1.0, 0.5)
    colsample_bytree = st.sidebar.slider(
        'Select colsample_bytree (default = 1.0)', 0.5, 1.0, 1.0)
    params_set = [max_depth, 0.1*eta, min_child_weight,
                  subsample, colsample_bylevel, colsample_bytree]

    start_time = datetime.datetime.now()
    accuracy_xgb, f1_xgb, roc_auc_xgb, recall_xgb, precision_xgb, model_xgb = XGB_train_metrics(
        data, params_set)
    st.subheader('Model Introduction')
    st.write('')
    st.write('XGBoost - e**X**treme **G**radient **B**oosting, is an implementation of gradient boosted **decision trees** designed for speed and performance, which has recently been dominating applied machine learning. We recommend you choose this model to do the prediction.')
    st.write('See more from this blog: https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/')
    st.write('')
    st.subheader('XGB metrics on testing dataset')
    st.write('')
    st.write('')
    st.write('')
    st.markdown("We separated the dataset to training and testing dataset, using training data to train our model then do the prediction on testing dataset, here's XGB prediction performance: ")
    st.write('')
    st.write(
        f'Running time: {(datetime.datetime.now() - start_time).seconds} s')
    st.table(pd.DataFrame(data=[round(accuracy_xgb * 100.0, 2), round(precision_xgb * 100.0, 2), round(recall_xgb*100, 2), round(roc_auc_xgb*100, 2), round(f1_xgb*100, 2)],
                          index=['Accuracy', 'Precision (% we predicted as Declined are truly Declined)', 'Recall (% Declined have been identified)', 'ROC_AUC', 'F1'], columns=['%']))
    st.subheader('Feature Importance:')

    # Plot feature importance
    df_feature = pd.DataFrame.from_dict(
        model_xgb.get_booster().get_fscore(), orient='index')
    df_feature.columns = ['Feature Importance']
    feature_importance = df_feature.sort_values(
        by='Feature Importance', ascending=False).T
    fig = px.bar(feature_importance, x=feature_importance.columns,
                 y=feature_importance.T)
    fig.update_xaxes(tickangle=45, title_text='Features')
    fig.update_yaxes(title_text='Feature Importance')
    st.plotly_chart(fig)
    return model_xgb


def xgb_predictor(model_xgb, rows, columns, df, drop_list):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.text('This process probably takes few seconds...')
    st.write('Note: Currently, the CSV file should have **exactly the same** format with **training dataset**:', df.head(2))
    st.write(
        f'Training dataset includes **{rows}** rows and **{columns}** columns')
    st.write('')

    if uploaded_file:
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.write('-'*80)
        st.write('Uploaded data:', data.head(30))
        st.write(
            f'Uploaded data includes **{data.shape[0]}** rows and **{data.shape[1]}** columns')
        start_time = datetime.datetime.now()
        data = data.dropna(axis=0, how='all')
        data.time = [datetime.datetime.strptime(
            x, '%m/%d/%Y %H:%M') for x in data.time]
        data, _ = data_preprocessing(data, drop_list)
        data2 = data.copy()
        X = data.drop(columns=['status'])
        prediction = model_xgb.predict(X)
        prediction_time = (datetime.datetime.now() - start_time).seconds
        data['status'] = ['Approved' if i ==
                          0 else 'Declined' for i in prediction]

        st.write('')
        st.write('-'*80)
        st.write('Prediction:')
        st.write(data.head(30))
        st.text(f'Running time: {prediction_time} s')
        st.write('')

        accuracy_pending = accuracy_score(data2.status, prediction)
        f1_pending = f1_score(data2.status, prediction)
        roc_auc_pending = roc_auc_score(data2.status, prediction)
        recall_pending = recall_score(data2.status, prediction)
        precision_pending = precision_score(data2.status, prediction)
        st.write('Metrics on uploaded data:')
        st.text("Note: This is only temporary since new data won't have labels")
        st.write('Accuracy:', round(100*accuracy_pending, 2), '%')
        st.write('Precision:', round(100*precision_pending, 2), '%')
        st.write('Recall:', round(100*recall_pending, 2), '%')
        st.write('ROC AUC:', round(100*roc_auc_pending, 2), '%')
        st.write('F1:', round(100*f1_pending, 2), '%')

        # Download prediction as a CSV file
        prediction_downloader(data)


def main():
    """Streamlit demo web app"""

    st.sidebar.title('Menu')
    choose_model = st.sidebar.selectbox("Choose the page or model", [
                                        "Home", "Logistic Regression", "XGB"])

    # Load data
    df, rows, columns, filename = load_data()
    data, drop_list = data_preprocessing(df)

    # Provide checkbox for uploading different training dataset
    if st.checkbox('Want to use other training set?'):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        st.text(
            "Note: Don't easily change training set which may bring big influence on prediction")

        if uploaded_file:
            df, data, drop_list, filename, rows, columns = upload_different_data(
                uploaded_file)

    # Home page building
    if choose_model == "Home":
        home_page_builder(df, data, rows, columns)

    # Page for Logistic Regression
    if choose_model == "Logistic Regression":
        model_reg = logistic_page_builder(data)

        if(st.checkbox("Want to Use this model to predict on a new dataset?")):
            logistic_predictor(model_reg, rows, columns, df, drop_list)

    # Page for XGB
    if choose_model == "XGB":
        model_xgb = xgb_page_builder(data)

        if(st.checkbox("Want to Use this model to predict on a new dataset?")):
            xgb_predictor(model_xgb, rows, columns, df, drop_list)


if __name__ == "__main__":
    main()
