import streamlit as st
import pandas as pd
import os
import datetime
from dateutil.relativedelta import relativedelta
import Libraries.reportes_plots as reportes_plots
import Libraries.utilitarios as util
import Libraries.dashboard as dashboard

current_date = datetime.datetime.today()
last_month_date = current_date - relativedelta(months = 1)

current_year = current_date.year; current_month = current_date.month; current_day = current_date.day
last_month_year = last_month_date.year; last_month = last_month_date.month; last_month_day = last_month_date.day

def get_files_by_dir(user, dir):
   c_whole_training = []
   for file in os.listdir(dir):
      if file.endswith(".csv") and user == file[16:23]:
         c_whole_training.append(file)
   return c_whole_training

def get_df_by_whole_training_data(datasets_list, d1, d2, peso_kg):
    df_calories_minute = pd.read_csv('02. trainers/exercises_calories_minute.csv', sep = '|')
    column = ''
    if (peso_kg < 50):
        column = 'less_50_Kg'    
    elif(peso_kg >= 50 and peso_kg < 59):
        column = '50_Kg'
    elif(peso_kg >= 59 and peso_kg < 70):
        column = '59_Kg'
    elif(peso_kg >= 70 and peso_kg < 82):
        column = '70_Kg'
    elif(peso_kg >= 82 and peso_kg < 93):
        column = '82_Kg'
    elif(peso_kg >= 93 and peso_kg < 100):
        column = '93_Kg'
    elif(peso_kg >= 100):
        column = '100_o_more_Kg'
    else:
        burned_kcal_factor = -1
    
    df_whole_training = pd.DataFrame()
    for i in datasets_list:
        parse_dates = ['DateTime_Start', 'DateTime_End']
        df_new = pd.read_csv(".//03. users//" + i, sep = '|', parse_dates=parse_dates)
        df_whole_training = pd.concat([df_whole_training, df_new])
    
    df_whole_training.reset_index(drop=True, inplace=True)
    df_whole_training = df_whole_training.sort_values(by=['DateTime_Start'], ascending=True)

    df_whole_training['Date_Start'] = pd.to_datetime(df_whole_training['DateTime_Start']).dt.date
    df_whole_training['Date_End'] = pd.to_datetime(df_whole_training['DateTime_End']).dt.date
    df_whole_training['Time_Seconds_diff'] = (df_whole_training['DateTime_End'] - df_whole_training['DateTime_Start']) / pd.Timedelta(seconds=1)
    df_whole_training['Time_Minutes_diff'] = (df_whole_training['DateTime_End'] - df_whole_training['DateTime_Start']) / pd.Timedelta(minutes=1)
    df_whole_training['Time_Hours_diff'] = (df_whole_training['DateTime_End'] - df_whole_training['DateTime_Start']) / pd.Timedelta(hours=1)
    df_whole_training['Kcal_factor'] = 0

    for i_1, row_1 in df_whole_training.iterrows():
        for i_2, row_2 in df_calories_minute.iterrows():
            if (row_1['id_exercise'] == row_2['id_exercise']):
                df_whole_training.at[i_1,'Kcal_factor'] = row_2[column]
    df_whole_training['Calories_burned_minutes'] = df_whole_training['Time_Minutes_diff'] * df_whole_training['Kcal_factor']
    df_whole_training['id_exercise_Kcal_factor'] = df_whole_training['id_exercise']

    df_whole_training = df_whole_training.loc[(df_whole_training['Date_Start'] >= d1) & (df_whole_training['Date_End'] <= d2)]
    
    return df_whole_training

def load_reportes(username):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(util.font_size_px("**🥇🥈🥉 YOUR WHOLE TRAINING**", 28), unsafe_allow_html=True)

    ##### SIDEBAR (START) #####
    col_start_date, col_end_date, col_time_unit = st.sidebar.columns(3)
    with col_start_date:
        d1 = st.date_input("🔜START DATE", datetime.date(last_month_year, last_month, last_month_day))
    with col_end_date:
        d2 = st.date_input("🔚END DATE", datetime.date(current_year, current_month, current_day))
    with col_time_unit:
        rd_time_unit = st.radio("⌛Charts units", ('Seconds', 'Minutes', 'Hours'), index = 1)
    st.sidebar.markdown("--------------", unsafe_allow_html=True)
    fetch_data = st.sidebar.button("FETCH DATA")
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    placeholder_fetch_data = st.sidebar.empty()
    placeholder_fetch_data.info('')
    st.sidebar.markdown("--------------", unsafe_allow_html=True)   
    #Obtiene listado de archivos csv con actividad del usuario loggeado
    user_file_list = get_files_by_dir(username, os.getcwd() + "\\03. users\\")
    with st.sidebar.expander("📁 File list"):
        st.info(user_file_list)
    st.sidebar.markdown("--------------", unsafe_allow_html=True)
    ##### SIDEBAR (END) #####

    if(fetch_data):
        if (d1 > d2):
            placeholder_fetch_data.error("START DATE cannot be greater than END DATE", icon="❌")
        else:
            df_whole_training = get_df_by_whole_training_data(user_file_list, d1, d2, st.session_state.peso)
            if len(df_whole_training) < 1:
                placeholder_fetch_data.info("No data", icon='📚')
            else:
                placeholder_fetch_data.success('{} rows found.'.format(len(df_whole_training)), icon='📅')
                ############################################                
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("Métricas del entrenamiento", 26), unsafe_allow_html=True)

                dateTime_start_str = df_whole_training['DateTime_Start'].min()
                dateTime_end_str = df_whole_training['DateTime_End'].max()
                traning_time_secs = df_whole_training['Time_Seconds_diff'].sum()
                traning_time_mins = df_whole_training['Time_Minutes_diff'].sum()
                traning_time_hours = df_whole_training['Time_Hours_diff'].sum()

                total_kcal_burned, df_calculated, col_value, df_weight = dashboard.get_burned_calories_total(df_whole_training, st.session_state.peso)

                st.text("🏁 Inicio entrenamiento :{}".format(dateTime_start_str))
                st.text("🥇 Fin entrenamiento    :{}".format(dateTime_end_str))
                st.text("🕒 Tiempo entrenamiento :{:.2f} segundos | {:.2f} minutos | {:.2f} horas".\
                        format(traning_time_secs, traning_time_mins, traning_time_hours))
                st.text("⚖️ Peso                 :{:.2f} Kg".format(st.session_state.peso))                
                st.text("🔥 Calorías quemadas    :[Ver cuadro 1] - {:.2f} calorías".format(total_kcal_burned))
                st.text("🧨 Factor de Calorías   :[Ver cuadro 2] - Múltiples ")
                st.markdown("<br />", unsafe_allow_html=True)
                st.text("Cuadro 1")
                st.dataframe(df_calculated.style.set_properties(subset=['Total_kcal_burned'], **{'background-color': 'blueviolet'}))   
                st.text("Cuadro 2")
                st.dataframe(df_weight.style.set_properties(subset=[col_value], **{'background-color': 'blueviolet'}))
                ############################################
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("🕒 Tiempo de entrenamiento:", 26), unsafe_allow_html=True)

                dict_col_time_unit = {'Seconds':'Time_Seconds_diff', 'Minutes':'Time_Minutes_diff', 'Hours':'Time_Hours_diff'}
                dict_x_axis_time_label = {'Seconds':'seconds', 'Minutes':'minutes', 'Hours':'hours'}

                img_main_1 = reportes_plots.summary_time_plot(
                    df_whole_training, dict_col_time_unit[rd_time_unit], dict_x_axis_time_label[rd_time_unit], 'stack', 0)
                img_main_2 = reportes_plots.summary_time_plot(
                    df_whole_training, dict_col_time_unit[rd_time_unit], dict_x_axis_time_label[rd_time_unit], 'group', 0)
                st.plotly_chart(img_main_1, use_container_width = True)
                st.plotly_chart(img_main_2, use_container_width = True)

                ############################################
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("🔥 Calorías quemadas:", 26), unsafe_allow_html=True)
                img_main_3 = reportes_plots.summary_time_plot(
                    df_whole_training, 'Calories_burned_minutes', 'Kcal', 'stack', 1)
                img_main_4 = reportes_plots.summary_time_plot(
                    df_whole_training, 'Calories_burned_minutes', 'Kcal', 'group', 1)
                st.plotly_chart(img_main_3, use_container_width = True)
                st.plotly_chart(img_main_4, use_container_width = True)

                ############################################
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("🏆 Similitud con Trainer por ejercicio:", 26), unsafe_allow_html=True)
                img_main_5 = reportes_plots.scatter_plot(df_whole_training)
                st.plotly_chart(img_main_5, use_container_width = True)

                ############################################
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("🧮 Predicción por regresión:", 26), unsafe_allow_html=True)
                img_main_6 = reportes_plots.regression_plot(df_whole_training, 'Calories_burned_minutes', 'Kcal')
                st.plotly_chart(img_main_6, use_container_width = True)
                img_main_7 = reportes_plots.regression_plot(df_whole_training, dict_col_time_unit[rd_time_unit],
                                                             dict_x_axis_time_label[rd_time_unit])
                st.plotly_chart(img_main_7, use_container_width = True)

                ############################################
                st.markdown("---------", unsafe_allow_html=True)
                st.markdown(util.font_size_px("💾 Raw Data:", 26), unsafe_allow_html=True)
                df_whole_training = df_whole_training.reset_index(drop=True)
                st.dataframe(df_whole_training)
                ############################################