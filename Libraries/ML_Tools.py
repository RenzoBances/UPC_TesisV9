import streamlit as st
import pandas as pd
import time

import Libraries.ML_Functions.ml_create_pose_dataset as ml_crea_pos_dt
import Libraries.ML_Functions.ml_model_train as ml_model_train
import Libraries.ML_Functions.ml_model_test as ml_model_test

def load_ml_tools():
    tab_create_pose_dataset, tab_ml_model_train, tab_ml_model_test = st.tabs(["💾CREATE POSE DATASET", "🤖 ML MODEL TRAINING", "🦾 ML MODEL TEST"])
    with tab_create_pose_dataset:
        st.markdown("**📹 SUBMIT A VIDEO**", unsafe_allow_html=True)  
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("1️⃣ SELECT EXERCISE:", unsafe_allow_html=True)
        id_exercise = st.selectbox("Choose exercise", ml_crea_pos_dt.list_exercise())
        st.markdown("<br>", unsafe_allow_html=True)
        mp4_files = st.file_uploader("Choose a MP4 file", type= ['mp4'], accept_multiple_files=True  )
        st.markdown("<br>", unsafe_allow_html=True)
        save_video_button = st.button("SAVE VIDEO")
        st.markdown("2️⃣ SAVING VIDEO", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if save_video_button:
            ml_crea_pos_dt.saving_video(id_exercise, mp4_files)
        st.markdown("------", unsafe_allow_html=True)
        st.markdown("**💾 CREATE POSE DATASET**", unsafe_allow_html=True)        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("1️⃣ SELECT EXERCISES:", unsafe_allow_html=True)
        id_exercises = st.multiselect('Choose exercise:', ['bird_dog', 'curl_up', 'forward_lunge', 'front_plank', 'push_up'])
        st.markdown("<br >", unsafe_allow_html=True)
        generate_dataset = st.button("GENERATE DATASET")
        st.markdown("<br >", unsafe_allow_html=True)
        st.markdown("------", unsafe_allow_html=True)
        st.markdown("2️⃣ PROCESSING VIDEOS  🔸  3️⃣ GENERATE CSV FILE:", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if generate_dataset:
            ml_crea_pos_dt.main_function(id_exercises)
    
    with tab_ml_model_train:
        st.markdown("**🤖 ML MODEL TRAINING**", unsafe_allow_html=True)
        st.markdown("Genera modelo entrenado de ML :", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("1️⃣ UPLOAD CSV COORDS DATASET FILE :", unsafe_allow_html=True)
        uploaded_csv_file = st.file_uploader("Choose a CSV file", type= ['csv'] )
        if uploaded_csv_file is not None:
            dataframe = pd.read_csv(uploaded_csv_file, sep=',')
            st.dataframe(dataframe)
            st.markdown("------", unsafe_allow_html=True)
            st.markdown("<br >", unsafe_allow_html=True)
            st.markdown("2️⃣ CHOOSE MODEL'S TARGET & FEATURES:", unsafe_allow_html=True)
            st.markdown("<br >", unsafe_allow_html=True)
            
            col_target, col_features = st.columns(2)
            with col_target:
                list_target = ['class']
                classes_values = dataframe[list_target[0]].unique().tolist()
                opts_target = st.multiselect("🎯 Target variable (always named 'class'):", list_target, list_target, disabled = True)
                st.markdown("<br >", unsafe_allow_html=True)
                st.text("Target class values:")
                st.code(classes_values)
                st.markdown("<br><br>", unsafe_allow_html=True)
                generate_model = st.button("GENERATE MODEL")

            with col_features:
                list_columns = list(dataframe.columns.values)
                list_columns.remove('class')
                opts_features = st.multiselect('🏹 Features variables:', list_columns,list_columns)
                st.text("{} features".format(len(opts_features)))
                st.warning("We recommend including only columns 'X', 'Y', 'Z' & 'V' (Not 'video', 'frames_per_sec', 'frame_count')", icon='⚠️')
            
            st.markdown("------", unsafe_allow_html=True)
            if(generate_model):            
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("3️⃣ MODEL PKL FILE TO BE GENERATED:", unsafe_allow_html=True)
                pkl_time_stamp = time.strftime("%Y%m%d_%H%M%S")
                path_pkl = './99. testing_resourses/outputs/'
                file_pkl = 'weights_body_language_{}.pkl'.format(pkl_time_stamp)
                path_file_pkl = path_pkl + file_pkl

                st.text_area(label='', value = '{}'.format(path_file_pkl))
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("4️⃣ TRAINING MODEL :", unsafe_allow_html=True)
                ml_model_train.main_function(dataframe, path_file_pkl, opts_features, opts_target)

    with tab_ml_model_test:
        st.markdown("**🦾 ML MODEL TEST**", unsafe_allow_html=True)
        st.markdown("Testea modelo de ML :", unsafe_allow_html=True)
        st.markdown("------", unsafe_allow_html=True)
        st.markdown("1️⃣ UPLOAD PNG IMAGE FILE :", unsafe_allow_html=True)
        uploaded_png_file = st.file_uploader("Choose a PNG file", type= ['png'] )        
        path_png = './99. testing_resourses/inputs/'
        
        if uploaded_png_file is not None:
            st.image(uploaded_png_file, caption='Model Test Image', width=400)
            uploaded_png_file_name = uploaded_png_file.name
            path_file_png = path_png + uploaded_png_file_name
            st.markdown("<br >", unsafe_allow_html=True)
            
            st.markdown("------", unsafe_allow_html=True)
            st.markdown("2️⃣ SELECT MODEL PKL FILE TO TEST:", unsafe_allow_html=True)
            list_pkl = ml_model_test.get_files_by_dir('./99. testing_resourses/outputs/', 'pkl')
            file_pkl = st.selectbox('Select model to test', list_pkl, index=0)
            st.markdown("<br >", unsafe_allow_html=True)
            
            btn_test_model = st.button("Test Model")
            
            if (btn_test_model):
                path_pkl = './99. testing_resourses/outputs/'
                path_file_pkl = path_pkl + file_pkl

                st.markdown("------", unsafe_allow_html=True)
                st.markdown("3️⃣ PREDICTING WITH MODEL PKL FILE:", unsafe_allow_html=True)
                ml_model_test.main_function(path_file_png, path_file_pkl)