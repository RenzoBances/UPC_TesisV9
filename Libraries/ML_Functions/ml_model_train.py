import pandas as pd
import pickle # Object serialization
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics
import streamlit as st
from io import StringIO

def load_dataset(csv_data, opts_features, opts_target):
    features = csv_data[opts_features]
    target = csv_data[opts_target]    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1234)
    return x_train, x_test, y_train, y_test

def evaluate_model(fit_models, x_test, y_test):
    st.info('Evaluate model accuracy:', icon='📚')

    # Evaluate and Serialize Model.
    for key_algo, value_pipeline in fit_models.items():
        yhat = value_pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, yhat)*100
        st.success('Classify algorithm: {}, Accuracy: {}%'.format(key_algo, accuracy), icon='🎯')
        st.markdown("<br>", unsafe_allow_html=True)

def main_function(dataset_csv_file, model_weights, opts_features, opts_target):

    x_train, x_test, y_train, y_test = load_dataset(dataset_csv_file, opts_features, opts_target)
    
    pipelines = {
        'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
        'rc' : make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    st.info('key: {}'.format(pipelines.keys()), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)
    st.info('value: {}'.format(list(pipelines.values())[0]), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)

    fit_models = {}
    st.warning('Training Model...', icon='⚠️')
    st.markdown("<br>", unsafe_allow_html=True)

    for key_algo, value_pipeline in pipelines.items():
        model = value_pipeline.fit(x_train, y_train)
        fit_models[key_algo] = model
    st.success('Training done!', icon='🎯')
    st.markdown("<br>", unsafe_allow_html=True)

    # Using x_test data input to Ridge Classifier model to predict.
    rc_predict = fit_models['rc'].predict(x_test)

    st.info('Showing first 5 prediction values: {}'.format(rc_predict[0:5]), icon='📚')
    st.markdown("<br>", unsafe_allow_html=True)

    # Save model weights.
    st.warning('Saving Model...', icon='⚠️')
    st.markdown("<br>", unsafe_allow_html=True)

    with open(model_weights, 'wb') as f:
        pickle.dump(fit_models['rf'], f)
    st.success('Model saved!', icon='🎯')
    st.markdown("<br>", unsafe_allow_html=True)
    
    evaluate_model(fit_models, x_test, y_test)