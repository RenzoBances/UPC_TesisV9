from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Dashboard functions
def plot_articulacion_performance_by_exerc(id_articulacion, id_exercise, df_result, posfijo):
    
   plot_filter = ['id_exercise', 'DateTime_Start', 'DateTime_End', 'count_pose', 'count_pose_g', 
                  (id_articulacion + posfijo)]
   df_result_plot = df_result[plot_filter]
   df_result_plot.rename(columns = {(id_articulacion + posfijo):(id_articulacion + posfijo + '_user')}, inplace = True)

   for i, row in df_result_plot.iterrows():
      id_exercise = row['id_exercise']
      id_pose = row['count_pose']
      trainer_angle = sysang_get_aprox_pose_articul(id_exercise, id_pose, id_articulacion)[0]
      df_result_plot.at[i,(id_articulacion+posfijo+'_trainer')] = trainer_angle

   df_result_plot['DateTime_Start'] =  pd.to_datetime(df_result_plot['DateTime_Start'], format='%Y-%m-%d %H:%M:%S.%f')
   df_result_plot['DateTime_End'] =  pd.to_datetime(df_result_plot['DateTime_End'], format='%Y-%m-%d %H:%M:%S.%f')
   df_result_plot['short_time_end'] = df_result_plot['DateTime_End'].dt.strftime('%H:%M:%S')
   df_result_plot['pose_merged'] = "Pose " + df_result_plot['count_pose'].map(str) + " (" + df_result_plot['count_pose_g'].map(str) + ")"
   df_result_plot['pose_merged_time'] = df_result_plot['pose_merged'].map(str) + "<br />" + df_result_plot['short_time_end'].map(str)

   min_y = min(df_result_plot[(id_articulacion+posfijo+'_user')].min(), df_result_plot[(id_articulacion+posfijo+'_trainer')].min())
   max_y = max(df_result_plot[(id_articulacion+posfijo+'_user')].max(), df_result_plot[(id_articulacion+posfijo+'_trainer')].max())

   ds = 10
   color_trainer_line = 'rgb(9,126,226)'
   color_trainer_mark = 'rgb(8,105,189)'
   color_user_line = 'rgb(234,63,247)'
   color_user_mark = 'rgb(198,53,209)'

   x = df_result_plot["pose_merged_time"].values.tolist()
   y_trainer = df_result_plot[(id_articulacion+posfijo+'_trainer')].values.tolist()
   y_user = df_result_plot[(id_articulacion+posfijo+'_user')].values.tolist()

   correlation_trainer_user = np.corrcoef(y_trainer, y_user)[0][1]
   fig2_corr, ax = plt.subplots(figsize=(20, 1))
   fig2_corr.patch.set_alpha(0.0)
   ax.xaxis.label.set_color('white')
   ax.tick_params(axis='x', colors='white')
   fig2_corr.subplots_adjust(bottom=0.3)

   cmap = mpl.cm.coolwarm
   norm = mpl.colors.Normalize(vmin=-1, vmax=1)

   fig2_corr.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, orientation='horizontal')
   ax.plot([correlation_trainer_user]*2, [0, 1], 'w', color='#000', marker='o')
   fig2_corr
   x_rev = x[::-1]
   y_trainer_upper_ds = [x + ds for x in y_trainer]
   y_trainer_lower_ds = [x - ds for x in y_trainer]
   y_trainer_lower_ds = y_trainer_lower_ds[::-1]
   fig = go.Figure()

   #Espectro trainer
   fig.add_trace(go.Scatter(
      x = x + x_rev,
      y = y_trainer_upper_ds + y_trainer_lower_ds,
      fill = 'toself',
      fillcolor = 'rgba(9,126,226,0.2)',
      line_color = 'rgba(255,255,255,0)',
      showlegend = False,
      name = 'Trainer',
   ))

   #Línea trainer
   y_trainer_grade = list(map(str, y_trainer))
   y_trainer_grade = [s + "°" for s in y_trainer_grade]

   fig.add_trace(go.Scatter(
      x = x,
      y = y_trainer,
      line_color = color_trainer_line,
      mode = "lines+text+markers",
      textposition = 'top center',
      textfont = dict(color = '#000000'),
      marker = dict(color = color_trainer_mark, size = 8),
      name = ('trainer: ' + get_articulacion_name(id_articulacion)[0])  #'Trainer',
   ))

   #Línea User
   y_user_grade = list(map(str, y_user))
   y_user_grade = [s + "°" for s in y_user_grade]

   fig.add_trace(go.Scatter(
      x = x,
      y = y_user,
      line_color = color_user_line,
      mode = "lines+text+markers",
      textposition = 'top center',
      textfont = dict(color = '#000000'),
      marker = dict(color = color_user_mark, size = 8),
      name = ('user: ' + get_articulacion_name(id_articulacion)[0]), #'User',
   ))

   #Trainer VALUE LABEL
   for a, b in zip(x, y_trainer):
      fig.add_annotation(
         x=a, y=b,
         text = "{:.2f}°".format(b),
         ax=100, ay=80,
         showarrow=False,
         bordercolor='rgb(8,105,189)',
         bgcolor="rgb(9,126,226)",
         font=dict(color="rgb(255,255,255)")
      )

   #User VALUE LABEL
   for a, b in zip(x, y_user):
      fig.add_annotation(
         x=a, y=b,
         text = "{:.2f}°".format(b),
         ax=-100, ay=-80,
         showarrow=False,
         bordercolor='rgb(198,53,209)',
         bgcolor="rgb(234,63,247)",
         font=dict(color="rgb(255,255,255)")
      )

   #Additional plot components
   fig.update_layout(
      title_text= ((id_exercise + " - " + get_articulacion_name(id_articulacion)[0]).upper()),
      title_x=0.5,
      xaxis_title="Poses",
      yaxis_title="Ángulos (grados)",
      yaxis_range=[min_y - 16, max_y + 16],
      xaxis = dict(
         tickmode = 'array',
         tickvals = x,
         ticktext = x
      )
   )
   return fig, fig2_corr, correlation_trainer_user

def plot_arts_performance_radar(df_result, id_exercise, posfijo):
   plot_filter = ['id_exercise', 'DateTime_Start', 'DateTime_End', 'count_pose', 'count_pose_g']
   
   articuls_columns = []
   for col in df_result.columns:
      if(col[-3:] == posfijo):
         articuls_columns.append(col)
         
   plot_filter.extend(articuls_columns)
   df_result_plot = df_result[plot_filter]
   for c in articuls_columns:
      df_result_plot.rename(columns = {(c):(c + '_user')}, inplace = True)
      df_result_plot[c + '_trainer'] = pd.Series(dtype='float')

   for i, row in df_result_plot.iterrows():
      id_exercise = row['id_exercise']
      id_pose = row['count_pose']
      
      for k in articuls_columns:
         id_articulacion = k.replace(posfijo, '')
         col_user = (k + '_user')
         col_trainer = (k + '_trainer')
         
         trainer_angle = sysang_get_aprox_pose_articul(id_exercise, id_pose, id_articulacion)[0]
         df_result_plot.at[i,(id_articulacion+posfijo+'_trainer')] = trainer_angle
         
   variables = articuls_columns + articuls_columns
   variables2 = [get_articulacion_name(a)[1] for a in variables]
   
   df_result_plot['DateTime_Start'] =  pd.to_datetime(df_result_plot['DateTime_Start'], format='%Y-%m-%d %H:%M:%S.%f')
   df_result_plot['DateTime_End'] =  pd.to_datetime(df_result_plot['DateTime_End'], format='%Y-%m-%d %H:%M:%S.%f')
   df_result_plot['short_time_end'] = df_result_plot['DateTime_End'].dt.strftime('%H:%M:%S')
   df_result_plot['pose_merged'] = "Pose " + df_result_plot['count_pose'].map(str) + " (" + df_result_plot['count_pose_g'].map(str) + ")"
   df_result_plot['pose_merged_time'] = df_result_plot['pose_merged'].map(str) + " - " + df_result_plot['short_time_end'].map(str)

   chart_list = []
   for i, row in df_result_plot.iterrows():
      user_angles = []
      trainer_angles = []
      values = []
      persona_1 = []
      persona_2 = []
      
      for q in articuls_columns:
         user_angles.append(row[q+ '_user'])
         trainer_angles.append(row[q+ '_trainer'])
         persona_1.append('User')
         persona_2.append('Trainer')
         
      values = user_angles
      values.extend(trainer_angles)
      personas = persona_1
      personas.extend(persona_2)
      
      df = pd.DataFrame(dict(
         values = values, variables = variables2, personas = personas))

      fig = px.line_polar(df,
                          r = 'values',
                          theta = 'variables',
                          line_close = True,
                          color = 'personas',
                          color_discrete_map = {'Trainer': 'dodgerblue', 'User': 'gold'}
                        )
      fig.update_layout(
         title_text= ((id_exercise + " - " + row['pose_merged_time']).upper()),
         title_x=0.5,
         xaxis_title="Poses",
         yaxis_title="Ángulos (grados)"
      )
      chart_list.append(fig)

   return chart_list

def get_articulacion_name(id_articulacion):
   dict_articulacion_nombre = {
      'right_elbow_angles': 'ángulo codo derecho',
      'right_hip_angles': 'ángulo cadera derecho',
      'right_knee_angles': 'ángulo rodilla derecho',
      'right_shoulder_angles': 'ángulo hombro derecho',
      'right_hip_angles': 'ángulo cadera derecho',
      'right_knee_angles': 'ángulo rodilla derecho',
      'right_shoulder_angles': 'ángulo hombro derecho',
      'right_hip_angles': 'ángulo cadera derecho',
      'right_ankle_angles': 'ángulo tobillo derecho',
      'right_hip_angles': 'ángulo cadera derecho',
      'right_knee_angles': 'ángulo rodilla derecho',
      'left_knee_angles': 'ángulo rodilla izquierda',
      'right_shoulder_angles': 'ángulo hombro derecho',
      'right_hip_angles': 'ángulo cadera derecho',
      'right_knee_angles': 'ángulo rodilla derecho',
      'left_knee_angles': 'ángulo rodilla izquierda',
      'right_elbow_angles': 'ángulo codo derecho',
      'left_elbow_angles': 'ángulo codo izquierdo'
   }
   
   dict_articulacion_table = {
      'right_elbow_angles_pu': 'ángulo codo derecho (Push Up)',
      'right_hip_angles_pu': 'ángulo cadera derecho (Push Up)',
      'right_knee_angles_pu': 'ángulo rodilla derecho (Push Up)',
      'right_shoulder_angles_cu': 'ángulo hombro derecho (Curl Up)',
      'right_hip_angles_cu': 'ángulo cadera derecho(Curl Up)',
      'right_knee_angles_cu': 'ángulo rodilla derecho (Curl Up)',
      'right_shoulder_angles_fp': 'ángulo hombro derecho (Front Plank)',
      'right_hip_angles_fp': 'ángulo cadera derecho (Front Plank)',
      'right_ankle_angles_fp': 'ángulo tobillo derecho (Front Plank)',
      'right_hip_angles_fl': 'ángulo cadera derecho (Forward Lunge)',
      'right_knee_angles_fl': 'ángulo rodilla derecho (Forward Lunge)',
      'left_knee_angles_fl': 'ángulo rodilla izquierda (Forward Lunge)',
      'right_shoulder_angles_bd': 'ángulo hombro derecho (Bird Dog)',
      'right_hip_angles_bd': 'ángulo cadera derecho (Bird Dog)',
      'right_knee_angles_bd': 'ángulo rodilla derecho (Bird Dog)',
      'left_knee_angles_bd': 'ángulo rodilla izquierda (Bird Dog)',
      'right_elbow_angles_bd': 'ángulo codo derecho (Bird Dog)',
      'left_elbow_angles_bd': 'ángulo codo izquierdo (Bird Dog)'
   }
   return dict_articulacion_nombre.get(id_articulacion), dict_articulacion_table.get(id_articulacion)

def get_articulaciones_list(articulaciones_text): 
   dictOfStrings = {",": "|", "'": "", " ": ""}
   
   for text, replacement in dictOfStrings.items():
      articulaciones_text = articulaciones_text.replace(text, replacement)
   articulaciones_list = articulaciones_text.split('|')
   return articulaciones_list

def get_data_grouped_by_date(df_whole_training):
   st.table(df_whole_training)
   df_whole_training['Fecha'] = df_whole_training['DateTime_Start'].str.slice(0,10)
   df_grouoped_by_date = df_whole_training[['id_exercise', 'Fecha']]
   df_grouoped_by_date = df_grouoped_by_date.drop_duplicates()
   df_grouoped_by_date = df_grouoped_by_date.reset_index()
   
   del df_grouoped_by_date['index']

   return df_grouoped_by_date

def get_columns_angles(articulaciones, posfijo):   
   #######################################################################
   #Sistema: NA
   #Función: Obtiene columns_angles del dataframe resultados
   #######################################################################
   #Input:
   #  - articulaciones: str (Articulaciones, Ángulos, Unión de 3 landmarks)
   #  - posfijo:        str (posfijo del ejercicio en los campos donde se guardan los ángulos en una tabla)
   #Output:
   #  - columns_angles: list columns_angles
   #######################################################################
   
   articulaciones_list = get_articulaciones_list(articulaciones)
   columns_angles = [element + posfijo for element in articulaciones_list]
   return columns_angles

def add_results_angle_img(image_path, id_exercise, id_pose, df_results, articulaciones, posfijo):
   img = Image.open(image_path)
   I1 = ImageDraw.Draw(img)
   font = ImageFont.truetype("C:\Windows\Fonts\courbd.ttf", 34)
   columns_angles = get_columns_angles(articulaciones, posfijo)
   id_pose = 1 
   id_articulacion = 'right_hip_angles'
   trainer_angle, trainer_sd, trainer_factor_sd = sysang_get_aprox_pose_articul(id_exercise, id_pose, id_articulacion)
   

   if(id_exercise == 'push_up'):
      if(id_pose == 1):
         I1.text((514, 50), ('RIGHT_hip_ANGLE: {}°'.format('165')), fill =(0, 0, 0), font=font)
         I1.text((56, 846), ('RIGHT_ELBOW_ANGLE: {}°'.format('155')), fill =(0, 0, 0), font=font)
         I1.text((920, 826), ('RIGHT_KNEE_ANGLE: {}°'.format('170')), fill =(0, 0, 0), font=font)
   return img

def get_training_time(df_results):
   #######################################################################
   #Sistema: Visualización de Dashboard
   #Función: Obtener de tiempo de entrenamiento en minutos
   #######################################################################
   #Input:
   #  - df_results:      pandas DataFrame
   #Output:
   #  - dateTime_start_str: datetime (momento inicial)
   #  - dateTime_end_str:   datetime (momento final)
   #  - minutes:            int (minutes)
   #  - seconds:            int (seconds)
   #######################################################################

   dateTime_start_str = df_results['DateTime_Start'].iloc[0]
   dateTime_end_str =   df_results['DateTime_End'].iloc[-1]
   format = "%Y-%m-%d %H:%M:%S.%f"
   dateTime_start_datetime = datetime.strptime(dateTime_start_str, format)
   dateTime_end_datetime = datetime.strptime(dateTime_end_str, format)
   delta = dateTime_end_datetime - dateTime_start_datetime
   minutes_seconds = divmod(delta.seconds, 60)
   minutes = minutes_seconds[0]
   seconds = minutes_seconds[1]
   return dateTime_start_str, dateTime_end_str, minutes, seconds

def get_burned_calories_total(df_user, peso_kg):
   column = ''
   burned_kcal_factor = 0

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

   df_weight = pd.read_csv('02. trainers/exercises_calories_minute.csv', sep = '|')
   df_calculated = df_user.groupby(['id_exercise']).sum()
   df_calculated = df_calculated[['Time_Minutes_diff']]
   df_calculated['Kcal_min'] = 0
   df_calculated['calculation_source'] = ''
   for i_ex, row_1 in df_calculated.iterrows():
      for j, row_2 in df_weight.iterrows():
         if (i_ex == row_2['id_exercise']):
            df_calculated.at[i_ex,'Kcal_min'] = row_2[column]
            df_calculated.at[i_ex,'calculation_source'] = row_2['calculation_source']
   df_calculated['Total_kcal_burned'] = df_calculated['Kcal_min'] * df_calculated['Time_Minutes_diff']
   df_calculated = df_calculated[['Time_Minutes_diff', 'Kcal_min', 'Total_kcal_burned', 'calculation_source']]
   df_weight = df_weight[['id_exercise', 'und_symbol','less_50_Kg','50_Kg','59_Kg','70_Kg','82_Kg','93_Kg','100_o_more_Kg',
                          'calculation_source']]
   df_weight.set_index('id_exercise', inplace=True)
   df_weight.sort_index(inplace=True)
   total_kcal_burned = df_calculated['Total_kcal_burned'].sum()
   df_calculated.rename(columns = {'Time_Minutes_diff':'Time_mins'}, inplace = True)
   df_calculated.sort_index(inplace=True)
   return total_kcal_burned, df_calculated, column, df_weight

       
def get_burned_calories(minutes, id_exercise, peso_kg):
   #######################################################################
   #Sistema: Visualización de Dashboard
   #Función: Calcula calorias según minutos y peso
   #######################################################################
   #Input:
   #  - minutes:     int
   #  - id_exercise: str
   #  - peso_kg:     float
   #Output:
   #  - burned_kcal_min:    float (cálculo de Kcal quemadas x minuto según ejercicio)
   #  - burned_kcal_factor: float (factor de Kcal)
   #  - kcal_min_calc_info: str (url de donde se basó el cálculo burned_kcal_min)
   #  - df_result:          pandas DataFrame
   #  - column:             str (columna con el rango del factor burned_kcal_factor)
   #######################################################################

   df = pd.read_csv('02. trainers/exercises_calories_minute.csv', sep = '|')
   filter = (df['id_exercise']==id_exercise)
   df_result = df.loc[filter]

   df_result = df_result[['exercise_name', 'und_symbol', 'less_50_Kg', '50_Kg', '59_Kg', 
               '70_Kg', '82_Kg', '93_Kg', '100_o_more_Kg', 'calculation_source']]

   kcal_min_50_less = df_result['less_50_Kg'].iloc[0]
   kcal_min_50 = df_result['50_Kg'].iloc[0]
   kcal_min_59 = df_result['59_Kg'].iloc[0]
   kcal_min_70 = df_result['70_Kg'].iloc[0]
   kcal_min_82 = df_result['82_Kg'].iloc[0]
   kcal_min_93 = df_result['93_Kg'].iloc[0]
   kcal_min_100_o_more = df_result['100_o_more_Kg'].iloc[0]
   kcal_min_calc_info = df_result['calculation_source'].iloc[0]

   column = ''
   burned_kcal_min = 0
   burned_kcal_factor = 0

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
   burned_kcal_factor = df_result[column].iloc[0]
   burned_kcal_min = burned_kcal_factor * minutes
   return burned_kcal_min, burned_kcal_factor, kcal_min_calc_info, df_result, column

def sysang_get_aprox_pose_articul(id_exercise, id_pose, id_articulacion):

   #######################################################################
   #Sistema: Sistema de ángulos
   #Función: Obtiene el ángulo del trainer según pose y articulación
   #######################################################################
   #Input:
   #  - id_exercise:     str
   #  - id_pose:         int
   #  - id_articulacion: str (Articulacion del cuerpo)
   #Output:
   #  - trainer_angle:     float (Ángulo trainer)
   #  - trainer_sd:        float (Desviación Estándar)
   #  - trainer_factor_sd: float (Factor Desviación Estándar)
   #######################################################################

   df = pd.read_csv("02. trainers/" + id_exercise + "/costs/angulos_" + id_exercise + "_promedio.csv")
   filter = (df['pose']==id_pose) & (df['Articulacion']==id_articulacion)
   df_result = df.loc[filter]
   trainer_angle = df_result['Angulo'].iloc[0]
   trainer_sd = df_result['Desviacion_estandar'].iloc[0]
   trainer_factor_sd = df_result['Factor_dev'].iloc[0]
   return trainer_angle, trainer_sd, trainer_factor_sd

def get_result_filter_dataframe(df_results, articulaciones, posfijo):

   #######################################################################
   #Sistema: Visualización de Dashboard
   #Función: Filtra dataframe de resultado retirando las columnas sin datos
   #######################################################################
   #Input:
   #  - df_results:     pandas DataFrame
   #  - articulaciones: str (Articulaciones, Ángulos, Unión de 3 landmarks)
   #  - posfijo:        str (posfijo del ejercicio en los campos donde se guardan los ángulos en una tabla)
   #Output:
   #  - df_results:     pandas DataFrame (Dataframe filtrado)
   #######################################################################

   columns = ['DateTime_Start', 'DateTime_End', 'Class', 'Prob', 'count_pose_g', 'count_pose', 'count_rep', 'count_set', 
              'pose_trainer_cost_min', 'pose_trainer_cost_max', 'pose_user_cost', ]
   
   columns_angles = get_columns_angles(articulaciones, posfijo)

   df_results = df_results[columns + columns_angles]
   return df_results

def plot_all_aprox_gauge_chart(df_results):
   gauges_charts = []
   for i, row in df_results.iterrows():
      value = row['Prob']
      pose = "{} ({})<br \>{}".format(row['count_pose'], row['count_pose_g'], row['DateTime_Start'])
      fig = plot_aprox_gauge_chart(value, ("Aprox(%) Media Pose {}".format(pose)), 0, 16, 32, 48)
      gauges_charts.append(fig)
   return gauges_charts


def plot_aprox_gauge_chart(value, title, range_min, range_mid1, range_mid2, range_max, n_poses = None):   
   #######################################################################
   #Sistema: Visualización de Dashboard
   #Función: Plot Medidor de aproximación de 0% a 100%
   #######################################################################
   #Input:
   #  - value:       float (Costo mínimo del trainer)
   #  - title:       str (Título del plot)
   #  - range_min:   float (Inicio Intervalo bajo)
   #  - range_mid1:  float (Inicio Intervalo medio)
   #  - range_mid2:  float (Fin Intervalo medio)
   #  - range_max:   float (Fin Intervalo alto)   
   #Output:
   #  - fig: Imagen (Plot)
   #######################################################################

   w_val = 800
   title_size = 26

   if (n_poses == 3):
      w_val = 400
      title_size = 24
   elif (n_poses == 5):
      w_val= 260
      title_size = 10

   fig = go.Figure(go.Indicator(
      mode = "gauge+number+delta",
      value = value,
      domain = {'x': [0, 1], 'y': [0, 1]},
      title = {'text': title, 'font': {'size': title_size}},
      delta = {'reference': range_max, 'increasing': {'color': "RebeccaPurple"}},
      number = {'suffix': '%', 'font': {'size': 20}},
      gauge = {
         'axis': {'range': [None, range_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
         'bar': {'color': "#EA3FF7"},
         'bgcolor': "white",
         'borderwidth': 2,
         'bordercolor': "gray",
         'steps': [
            {'range': [range_min, range_mid1], 'color': "#097EE2"},
            {'range': [range_mid1, range_mid2], 'color': "#09C5DD"},
            {'range': [range_mid2, range_max], 'color': "#40E7A5"}
            ],
         'threshold': {
            'line': {'color': "cyan", 'width': 4},
            'thickness': 0.75,
            'value': range_max - 1}}))
   
   fig.layout.width = w_val
   fig.layout.height = 300 
   fig.update_layout(paper_bgcolor = "#F2F3F4", font = {'color': "darkblue", 'family': "Arial"})

   return fig

def plot_all_cost(df_results):
   costs_charts = []

   for i, row in df_results.iterrows():
      min_trainer = row['pose_trainer_cost_min']
      max_trainer = row['pose_trainer_cost_max']
      value_user = row['pose_user_cost']
      pose = "{} ({})<br \>{}".format(row['count_pose'], row['count_pose_g'], row['DateTime_Start'])
      title = "Aproximación Costos % {}".format(pose)
      fig = plot_cost(min_trainer, max_trainer, value_user, title, "#FFC107")
      costs_charts.append(fig)

   return costs_charts

def plot_cost(min_trainer, max_trainer, value_user, title, fill_color = None):
   #######################################################################
   #Sistema: Visualización de Dashboard
   #Función: Plot Costo de Usuario contra Rango de Costos del Trainer
   #######################################################################
   #Input:
   #  - min_trainer:  float (Costo mínimo del trainer)
   #  - max_trainer:  float (Costo máximo del trainer)
   #  - value_user:   float (Costo del usuario)
   #  - title:        str (Título del plot)
   #Output:
   #  - fig: Imagen (Plot)
   #######################################################################

   if fill_color== None:
      fill_color = "#EA3FF7"
   
   list_plot_cost = [min_trainer, max_trainer, value_user]
   min_list = min(list_plot_cost)
   max_list = max(list_plot_cost)
   
   fig = go.Figure(layout_xaxis_range=[min_list - 1, max_list + 1])

   fig.update_yaxes(visible=False, showticklabels=False)
   
   fig.add_vrect(x0 = min_trainer, x1 = max_trainer, 
               annotation_text="Trainer cost range: [{0:.2f} ,{1:.2f}]".format(min_trainer, max_trainer), 
               annotation_position="left top", fillcolor=fill_color, opacity=0.25, line_width=0, annotation_font_size=18,
               annotation_font_color="black")

   fig.add_vline(x = value_user, annotation_text="User cost: {0:.2f}".format(value_user), line_color="blue", 
               line_dash="dot", annotation_position="left bottom", name = "User", annotation_font_size=18,
               annotation_font_color="blue")
   
   fig.layout.width = 350
   fig.layout.height = 300
   
   fig.update_layout(
      title = dict(text = title, font=dict(size=20,color="darkblue", family="Arial"))
   )
   
   fig.update_layout({
      'plot_bgcolor': 'rgba(255, 255, 255, 1)',
      'paper_bgcolor': 'rgba(255, 255, 255, 1)',
   })
   return fig