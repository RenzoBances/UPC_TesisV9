import pandas as pd
import streamlit as st
import mediapipe as mp
import cv2
import os
import numpy as np
from math import acos, degrees
import Libraries.dashboard as dashboard

def saving_video(id_exercise, mp4_files):
    video_path = './99. testing_resourses/inputs/create_pose_datasets/{}'.format(id_exercise)
    num_videos = len(os.listdir(video_path))
    for mp4_file in mp4_files:
      num_videos += 1
      if mp4_file is not None:
          video_name = id_exercise+"_trainer"+str(num_videos)+".mp4"
          save_video_path = os.path.join(video_path, video_name)
          with open(save_video_path, "wb") as f:
              f.write(mp4_file.getbuffer())
          st.success("Video saved!")
      else:
          st.warning("The video has not been loaded.")
      
def main_function(id_exercises):
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    df_puntos_final = pd.DataFrame()

    for id_exercise in id_exercises:
        path_videos_input = './99. testing_resourses/inputs/create_pose_datasets/{}/'.format(id_exercise)
        video_files = os.listdir(path_videos_input)
        video_files_list = []
        for video_file in video_files:
            video_files_list.append(video_file)

        for video in video_files_list:
            st.markdown("<br >", unsafe_allow_html=True)
            st.markdown("🩻 Processing Video: {}{}".format(path_videos_input, video), unsafe_allow_html=True)
            video = path_videos_input+video
            cap = cv2.VideoCapture(video)
            with mp_pose.Pose(static_image_mode=False) as pose:
                
                while True:   
                  ret, frame = cap.read()

                  if ret == False:
                      break
                  height, width, _ = frame.shape
                  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  results = pose.process(frame_rgb)
                  
                  resultados = []

                  for i in range(0, len(results.pose_landmarks.landmark)):
                    resultados.append(results.pose_landmarks.landmark[i].x)
                    resultados.append(results.pose_landmarks.landmark[i].y)
                    resultados.append(results.pose_landmarks.landmark[i].z)
                    resultados.append(results.pose_landmarks.landmark[i].visibility)

                  df_puntos = pd.DataFrame(np.reshape(resultados, (132, 1)).T)
                  df_puntos['class'] = id_exercise
                  df_puntos_final = pd.concat([df_puntos_final, df_puntos])
    df_puntos_final = df_puntos_final[[df_puntos_final.columns[-1]] + list(df_puntos_final.columns[:-1])]
    df_puntos_final.columns = ['class', 'x1', 'y1', 'z1', 'v1',
                                'x2', 'y2', 'z2', 'v2',
                                'x3', 'y3', 'z3', 'v3',
                                'x4', 'y4', 'z4', 'v4',
                                'x5', 'y5', 'z5', 'v5',
                                'x6', 'y6', 'z6', 'v6',
                                'x7', 'y7', 'z7', 'v7',
                                'x8', 'y8', 'z8', 'v8',
                                'x9', 'y9', 'z9', 'v9',
                                'x10', 'y10', 'z10', 'v10',
                                'x11', 'y11', 'z11', 'v11',
                                'x12', 'y12', 'z12', 'v12',
                                'x13', 'y13', 'z13', 'v13',
                                'x14', 'y14', 'z14', 'v14',
                                'x15', 'y15', 'z15', 'v15',
                                'x16', 'y16', 'z16', 'v16',
                                'x17', 'y17', 'z17', 'v17',
                                'x18', 'y18', 'z18', 'v18',
                                'x19', 'y19', 'z19', 'v19',
                                'x20', 'y20', 'z20', 'v20',
                                'x21', 'y21', 'z21', 'v21',
                                'x22', 'y22', 'z22', 'v22',
                                'x23', 'y23', 'z23', 'v23',
                                'x24', 'y24', 'z24', 'v24',
                                'x25', 'y25', 'z25', 'v25',
                                'x26', 'y26', 'z26', 'v26',
                                'x27', 'y27', 'z27', 'v27',
                                'x28', 'y28', 'z28', 'v28',
                                'x29', 'y29', 'z29', 'v29',
                                'x30', 'y30', 'z30', 'v30',
                                'x31', 'y31', 'z31', 'v31',
                                'x32', 'y32', 'z32', 'v32',
                                'x33', 'y33', 'z33', 'v33']
              
    st.markdown("<br >", unsafe_allow_html=True)
    path_png_output = './99. testing_resourses/outputs/create_pose_datasets/'
    files_csv = "{}_puntos_trainers.csv".format(id_exercise)

    st.markdown("📅 Generate CSV file: {}{}".format(path_png_output, "coords_dataset.csv"), unsafe_allow_html=True)
    st.dataframe(df_puntos_final)        
    # Duplicate records are removed
    df_puntos_final = df_puntos_final.drop_duplicates()
    df_puntos_final.to_csv("./99. testing_resourses/outputs/create_pose_datasets/coords_dataset.csv", index=False)
    st.markdown("------", unsafe_allow_html=True)
    
def list_exercise():
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    list_exercises = df_exercise['id_exercise'].unique().tolist()
    return list_exercises

def get_number_poses(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    return df_exercise['n_poses'].loc[df_exercise.index[0]]

def get_articulaciones(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    articulaciones =  df_exercise['articulaciones'].loc[df_exercise.index[0]]
    articulaciones_broken = dashboard.get_articulaciones_list(articulaciones)
    
    return articulaciones_broken

def get_landmark_values(id_exercise):
    df_exercise =pd.read_csv('./02. trainers/exercises_metadata.csv', sep = '|')
    df_exercise = df_exercise.loc[df_exercise['id_exercise'] == id_exercise]

    landmark_values = df_exercise['landmark_values'].loc[df_exercise.index[0]]
    landmark_values_broken = dashboard.get_articulaciones_list(landmark_values)

    return landmark_values_broken