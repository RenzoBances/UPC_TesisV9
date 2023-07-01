import plotly.express as px
import streamlit as st
from sklearn import linear_model
import pandas as pd
from datetime import timedelta
import numpy as np

def summary_time_plot(df_whole_training, y_column, y_label, barmode, n_palette):
    colorList_1 = [px.colors.qualitative.Alphabet[6], px.colors.qualitative.Alphabet[11], px.colors.qualitative.Plotly[2], px.colors.qualitative.Plotly[7], px.colors.qualitative.G10[5]]
    colorList_2 = ["#7CEA9C", '#50B2C0', "rgb(114, 78, 145)", "hsv(348, 66%, 90%)", "hsl(45, 93%, 58%)"]
    color_pallete_list = [colorList_1, colorList_2]
    total_workout_time = df_whole_training[y_column].sum()
    fig = px.bar(df_whole_training, x='Date_Start', y=y_column,
                 color='id_exercise', 
                 text_auto=True,
                 color_discrete_sequence = color_pallete_list[n_palette],
                 labels={'id_exercise':'Workout routine', 'Date_Start': 'Fecha', y_column: y_label},
                 height=400)
    fig.update_layout(
        title = 'Entrenamiento (Total: {:.2f} {}) por Fecha por Ejercicio - {}ed'.format(total_workout_time, y_label, barmode),
        xaxis_tickformat = '%d %B (%a)<br>%Y',        
        plot_bgcolor="#555",
        paper_bgcolor="#444",
        barmode = barmode,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig

def scatter_plot(df_whole_training):
    fig = px.scatter(df_whole_training, x="DateTime_Start", y="Prob", color="id_exercise", size='Kcal_factor',
                     labels={'id_exercise':'Workout routine', 'Prob':'Prob(%) Trainer', 'DateTime_Start': 'Fecha Hora'},
                     )
    fig.update_layout(
        title = 'Prob(%) similitud con Trainer por Fecha por Ejercicio',
        xaxis_tickformat = '%H~%M<br>%d %B (%a)<br>%Y',
        plot_bgcolor="#555",
        paper_bgcolor="#444",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    return fig

def regression_plot(df_whole_training, y_column, y_label):

    df_whole_training = df_whole_training[['Date_Start', y_column]]
    df_whole_training = df_whole_training.groupby(['Date_Start'], as_index=False)[y_column].sum()
    df_whole_training = df_whole_training.round(2)
    df_whole_training['index'] = df_whole_training.index

    date_min = df_whole_training['Date_Start'].min()
    date_max = df_whole_training['Date_Start'].max()
    times = len(df_whole_training)
    date_delta = (date_max-date_min).days

    x = df_whole_training['index'].values
    y = df_whole_training[y_column].values
    length = len(df_whole_training)
    x = x.reshape(length, 1)
    y = y.reshape(length, 1)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred_vals_1 = df_whole_training['index'].values.reshape(-1, 1)
    df_whole_training['Y_tendencia'] = regr.predict(y_pred_vals_1)
    df_whole_training['Y_prediccion'] = np.nan
    
    if date_delta > 0:
        y_pred_vals_2 = np.arange(df_whole_training['index'].max() + 1, df_whole_training['index'].max() + date_delta + 1).reshape(-1, 1)
        pred = regr.predict(y_pred_vals_2)

        for i in range(date_delta):
            new_date = date_max + timedelta(days = i + 1)
            row = pd.Series([new_date, np.nan, i+times, np.nan, float(pred[i])], index = df_whole_training.columns)
            df_whole_training = df_whole_training.append(row, ignore_index=True)
    
    df_whole_training.rename(columns = {y_column: y_label}, inplace = True)
    y_column = y_label
    
    fig = px.scatter(
        df_whole_training,
        x = 'Date_Start',
        y = [y_column, 'Y_tendencia', 'Y_prediccion'],
        labels={'variable':'Leyenda', 'Date_Start': 'Fecha'},
        )
    fig.update_traces(
        mode='markers+lines',
        error_y=dict(color='white', width=20, thickness=5),
        showlegend=True
    )
    fig.update_layout(
        xaxis_tickformat = '%d %B (%a)<br>%Y',
        yaxis_title = y_label,
        title='🔵Esfuerzo en {}, 🔴Tendencia & 🟢Predicción'.format(y_label),
        hovermode="x",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    annotation = "Según tendencia, se estima consumir {:.2f} {} entre {} y {}".format(
        df_whole_training['Y_prediccion'].sum(),
        y_label,
        date_max + timedelta(days = 1),
        df_whole_training['Date_Start'].max()
        )
    fig.add_annotation(dict(font=dict(color='yellow',size=15),
                            x=0,
                            y=-0.23,
                            showarrow=False,
                            text=annotation,
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    return fig