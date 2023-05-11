# 1.1. PYTHON LIBRARIES
#######################
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import base64
from random import randrange
import pandas as pd
import pickle
from math import acos, degrees
import os 
from pathlib import Path
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import database as db
import threading
from gtts import gTTS
import pygame
import io

#1.2. OWN LIBRARIES
###################
import Libraries.Exercises.UpcSystemCost as UpcSystemCost
import Libraries.dashboard as dashboard
import Libraries.utilitarios as ut

#1.3. GLOBAL VARIABLES

# 2. FUNCTIONS
##############
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_exercise_metadata(id_exercise):
    df = pd.read_csv('02. trainers/exercises_metadata.csv', sep = '|', encoding='latin-1')

    st.session_state.short_name          = df.loc[df['id_exercise']==id_exercise, 'short_name'].values[0]
    st.session_state.vista               = df.loc[df['id_exercise']==id_exercise, 'vista'].values[0]
    st.session_state.articulaciones      = df.loc[df['id_exercise']==id_exercise, 'articulaciones'].values[0]
    st.session_state.posfijo             = df.loc[df['id_exercise']==id_exercise, 'posfijo'].values[0]
    st.session_state.n_poses             = df.loc[df['id_exercise']==id_exercise, 'n_poses'].values[0]
    st.session_state.n_sets_default      = df.loc[df['id_exercise']==id_exercise, 'n_sets_default'].values[0]
    st.session_state.n_reps_default      = df.loc[df['id_exercise']==id_exercise, 'n_reps_default'].values[0]
    st.session_state.n_rest_time_default = df.loc[df['id_exercise']==id_exercise, 'n_rest_time_default'].values[0]
    st.session_state.detail              = df.loc[df['id_exercise']==id_exercise, 'detail'].values[0]

#functions to use text to speech
def speak(text):
    substrs_to_compare = ['Felicitaciones', "Mantenga la posicion"]
    with io.BytesIO() as file:
        tts = gTTS(text=text, lang='es')
        tts.write_to_fp(file)
        file.seek(0)
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    if not any(text.startswith(substring) for substring in substrs_to_compare):
        time.sleep(3)

def load_user(username):
    df = db.get_user(username)
    st.session_state.nombre = df['name']
    st.session_state.edad = df['edad']
    st.session_state.peso = df['peso']
    st.session_state.talla = df['talla']
    st.session_state.imc = df['imc']
    st.session_state.perabdominal = df['perabdominal']
    st.session_state.cexweek = df['perabdominal']

def get_exercise_gif(id_exercise):
    gif_file = "02. trainers/" + id_exercise + "/images/" + id_exercise + ".gif"
    return gif_file

def font_size_px(markdown_text):
    return "<span style='font-size:26px'>" + markdown_text + "</span>"

def load_home():
    #SIDEBAR START
    load_user(username)
    st.sidebar.markdown('---')
    st.sidebar.markdown(f'''**Personal Information**''', unsafe_allow_html=True)
    st.sidebar.markdown(f'''**{int(st.session_state.edad)} años**''', unsafe_allow_html=True)
    st.sidebar.markdown(f'''**{st.session_state.peso} Kg**''', unsafe_allow_html=True)
    st.sidebar.markdown(f'''**{st.session_state.talla} cm**''', unsafe_allow_html=True)
    st.sidebar.markdown(f'''**{st.session_state.imc} de IMC**''', unsafe_allow_html=True)
    st.sidebar.markdown(f'''**{st.session_state.perabdominal} de Perimetro Abdominal**''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**POSE_LANDMARKS**<br>Una lista de puntos de referencia de la pose. Cada punto de referencia consta de lo siguiente:<br><ul><li><b>X & Y:</b> coordenadas de referencia normalizadas a [0.0, 1.0] por el ancho y la altura de la imagen, respectivamente.</li><li><b>Z:</b> Representa la profundidad del punto de referencia con la profundidad en el punto medio de las caderas como origen, y cuanto menor sea el valor, más cerca estará el punto de referencia de la cámara. La magnitud de z usa aproximadamente la misma escala que x.</li><li><b>Visibilidad:</b> un valor en [0.0, 1.0] que indica la probabilidad de que el punto de referencia sea visible (presente y no ocluido) en la imagen.</li></ul><br>",
        unsafe_allow_html=True)
    st.markdown("**MODELO DE PUNTOS DE REFERENCIA DE POSE (BlazePose GHUM 3D)**<br>El modelo de puntos de referencia en MediaPipe Pose predice la ubicación de 33 puntos de referencia de pose (consulte la figura a continuación).<br>",
        unsafe_allow_html=True)
    st.image("01. webapp_img/pose_landmarks_model.png", width=600)

def load_reportes():

    ####pruebas manu ⬇️

    ####pruebas manu ⬆️






    # Streamlit tabs
    tab1, tab2= st.tabs(["A. Today's training", "B. Your whole training"])
    dir_user = os.getcwd() + "\\03. users\\"
    b_today_training = dashboard.get_files_by_day(username, dashboard.get_datestamp_txt(), dir_user)
    c_whole_training = dashboard.get_files_by_dir(username, dir_user)

    # Today's training
    with tab1:
        image_message, aprox_indicator = st.columns(2)

        if len(b_today_training) > 0:
            with image_message:
                #st.title("Precisión del entrenamiento (%)")
                image1 = "https://i.pinimg.com/originals/d3/4c/04/d34c0453fc7a77c00b3ab2451b65d592.png"
                st.image(image1, width = 360)

            with aprox_indicator:
                df_today_training = dashboard.get_df_by_whole_training_data(b_today_training)
                df_aprox_exercise_by_date = dashboard.get_data_grouped_by_date(df_today_training)
                df_aprox_exercise_by_date = get_df_aprox_exercise_by_date(df_today_training, df_aprox_exercise_by_date)
                fig_aprox = dashboard.plot_barchar_by_date(df_aprox_exercise_by_date, "Fecha", "Aproximacion", "id_exercise", "Aproximacion", "Fecha", "Aproximación (%)", "Aproximación de ejercicios realizados por día")
                st.plotly_chart(fig_aprox)

            tr_time_indicator, cal_burned_indicator = st.columns(2)

            with tr_time_indicator:
                df_train_time_exercise_by_date = dashboard.get_df_train_time_exercise_by_date(b_today_training)
                fig_tr_time = dashboard.plot_barchar_by_date(df_train_time_exercise_by_date, "Fecha", "Training_time", "id_exercise", "Training_time", "Fecha", "Tiempo de entrenamiento (min)", "Tiempo de entrenamiento de ejercicios realizados por día")
                st.plotly_chart(fig_tr_time)  

            with cal_burned_indicator:
                df_calories_burned_by_date = dashboard.get_df_calories_burned_by_date(df_train_time_exercise_by_date)
                fig_cal = dashboard.plot_barchar_by_date(df_calories_burned_by_date, "Fecha", "Calories_burned", "id_exercise", "Calories_burned", "Fecha", "Calorias quemadas", "Calorias quemadas por ejercicios realizados por día")
                st.plotly_chart(fig_cal)
        elif len(b_today_training) == 0:
            with image_message:
                st.title("No existe información de ejercicios ejecutados el día de hoy")
    # Your whole training   
    with tab2:
        image_message1, aprox_indicator1 = st.columns(2)
        if len(c_whole_training) > 0:
            with image_message1:
                #st.title("precisión del entrenamiento (%)")
                image1 = "https://www.quotesforjoy.com/wp-content/uploads/2021/09/The-rock-quote-3.png"
                st.image(image1, width = 360)
            with aprox_indicator1:
                df_whole_training = dashboard.get_df_by_whole_training_data(c_whole_training)
                df_aprox_exercise_by_date = dashboard.get_data_grouped_by_date(df_whole_training)
                df_aprox_exercise_by_date = get_df_aprox_exercise_by_date(df_whole_training, df_aprox_exercise_by_date)
                fig_aprox = dashboard.plot_barchar_by_date(df_aprox_exercise_by_date, "Fecha", "Aproximacion", "id_exercise", "Aproximacion", "Fecha", "Aproximación (%)", "Aproximación de ejercicios realizados por día")
                st.plotly_chart(fig_aprox)

            tr_time_indicator1, cal_burned_indicator1 = st.columns(2)

            with tr_time_indicator1:
                df_train_time_exercise_by_date = dashboard.get_df_train_time_exercise_by_date(c_whole_training)
                fig_tr_time = dashboard.plot_barchar_by_date(df_train_time_exercise_by_date, "Fecha", "Training_time", "id_exercise", "Training_time", "Fecha", "Tiempo de entrenamiento (min)", "Tiempo de entrenamiento de ejercicios realizados por día")
                st.plotly_chart(fig_tr_time)  

            with cal_burned_indicator1:
                df_calories_burned_by_date = dashboard.get_df_calories_burned_by_date(df_train_time_exercise_by_date)
                fig_cal = dashboard.plot_barchar_by_date(df_calories_burned_by_date, "Fecha", "Calories_burned", "id_exercise", "Calories_burned", "Fecha", "Calorias quemadas", "Calorias quemadas por ejercicios realizados por día")
                st.plotly_chart(fig_cal)
        elif len(c_whole_training) == 0:
            with image_message1:
                st.title("No existe data histórica para el usuario")

def print_sidebar_main(id_exercise):
        
        load_exercise_metadata(id_exercise)

        #SIDEBAR START
        st.sidebar.markdown('---')
        st.sidebar.markdown(f'''**{st.session_state.short_name}**''', unsafe_allow_html=True)
        st.sidebar.image(get_exercise_gif(id_exercise))  
        vista_gif = '01. webapp_img/vista_' + st.session_state.vista + '.gif'
        with st.sidebar.expander("💡 Info"):
            st.info(st.session_state.detail)
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        st.session_state.n_sets = st.sidebar.number_input("Sets", min_value=1, max_value=10, value=st.session_state.n_sets_default)
        st.session_state.n_reps = st.sidebar.number_input("Reps", min_value=1, max_value=10, value=st.session_state.n_reps_default)
        st.session_state.seconds_rest_time = st.sidebar.number_input("Rest Time (seconds)", min_value=1, max_value=30, value=st.session_state.n_rest_time_default)
        position_image, position_text = st.sidebar.columns(2)
        with position_image:
            st.image(vista_gif, width=100)
        with position_text:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("**Vista:** " + st.session_state.vista, unsafe_allow_html=True)
            st.markdown("**N° poses:** " + str(st.session_state.n_poses), unsafe_allow_html=True)
        exercise_to_do[app_mode] = {"reps":st.session_state.n_reps,"sets":st.session_state.n_sets,"secs":st.session_state.seconds_rest_time}
        st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
        placeholder_title.title('STARTER TRAINING - '+ st.session_state.short_name)
        st.markdown('---')
        
def get_trainer_coords(id_exercise, id_trainer):
    
    df = pd.read_csv("02. trainers/" + id_exercise + "/costs/" + id_exercise + "_puntos_trainer"+str(id_trainer)+".csv")
    if id_exercise == "bird_dog":
        df = df.iloc[: , :-6]
    else:
        df = df.iloc[: , :-3]
    del df['pose']
    return df

def get_cost_pose_trainer(id_exercise, n_pose):
    if n_pose >= 1:
        df = pd.read_csv("02. trainers/" + id_exercise + "/costs/costos_" + id_exercise + "_promedio.csv")

        cost_align = df.loc[df['Pose'] == n_pose,"Costo_alineamiento"].reset_index(drop = True)[0]
        ds = df.loc[df['Pose'] == n_pose,"Desviacion_estandar_f"].reset_index(drop = True)[0]
        
        pose_trainer_cost_min = round(cost_align - ds, 2)
        pose_trainer_cost_max = round(cost_align + ds, 2)
    else:
        pose_trainer_cost_min = 0
        pose_trainer_cost_max = 0
    return pose_trainer_cost_min, pose_trainer_cost_max

def get_trainers_angles(id_exercise):
    df = pd.read_csv("02. trainers/" + id_exercise + "/costs/angulos_" + id_exercise + "_promedio.csv")
    return df

def LoadModel():
    model_weights = './04. model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model

def calculate_angleacos(a,b,c):
    # angle = degrees(acos((a**2 + c**2 - b**2) / (2 * a * c)))
    angle = degrees(acos(max(min((a**2 + c**2 - b**2) / (2 * a * c), 1), -1)))
    if angle > 0:
        angle = int(angle)
    else:
        angle = 0
    return angle

def get_angle(df, index, part):
    angle_in=df['Angulo'][(df.pose==index+1)&(df.Articulacion==part)]
    angle_in=angle_in.iloc[0]
    return angle_in

def get_desv_angle(df, index, part):
    desv_in=df['Desviacion_estandar_f'][(df.pose==index+1)&(df.Articulacion==part)]
    desv_in=desv_in.iloc[0]
    return desv_in

def get_timestap_log():
    now = time.time()
    mlsec = repr(now).split('.')[1][:3]
    timestamp1 = time.strftime("%Y-%m-%d %H:%M:%S.{}".format(mlsec))
    return timestamp1

def get_timestap_txt(id_user,id_exer):
    timestamp2 = time.strftime("%Y%m%d_%H%M%S"+"_"+id_user+"_"+id_exer)
    return timestamp2

def update_dashboard():
    
        if st.session_state.count_pose_g < st.session_state.total_poses:
            st.session_state.count_pose_g += 1
            placeholder_status.markdown(font_size_px("🏎️ TRAINING..."), unsafe_allow_html=True)
            placeholder_pose_global.metric("POSE GLOBAL DONE", str(st.session_state.count_pose_g) + " / " + str(st.session_state.total_poses), "+1 pose")

            if st.session_state.count_pose < st.session_state.n_poses:
                st.session_state.count_pose += 1
                placeholder_pose.metric("POSE DONE", str(st.session_state.count_pose) + " / "+ str(st.session_state.n_poses), "+1 pose")
        else:
            placeholder_status.markdown(font_size_px("🥇 FINISH !!!"), unsafe_allow_html=True)
            placeholder_trainer.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + "1.png")
            placeholder_pose_global.metric("POSE GLOBAL DONE", str(st.session_state.count_pose_g) + " / " + str(st.session_state.total_poses), "COMPLETED", delta_color="inverse")
            placeholder_pose.metric("POSE DONE", str(st.session_state.n_poses) + " / "+ str(st.session_state.n_poses), "COMPLETED", delta_color="inverse")
            placeholder_rep.metric("REPETITION DONE", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "COMPLETED", delta_color="inverse")
            placeholder_set.metric("SET DONE", str(st.session_state.count_set) + " / "+ str(st.session_state.n_sets), "COMPLETED", delta_color="inverse" )

def get_df_aprox_exercise_by_date(df_whole_training, df_aprox_exercise_by_date):

   aproxs_by_date = []


   for i in range(0, len(df_aprox_exercise_by_date)):
       
      df_analisis = df_whole_training[(df_whole_training['id_exercise'] == df_aprox_exercise_by_date['id_exercise'][i]) & (df_whole_training['Fecha'] == df_aprox_exercise_by_date['Fecha'][i])]
       
      if df_aprox_exercise_by_date['id_exercise'][i] == 'push_up':
       
         aproxs_by_date.append(dashboard.get_aprox_exercise(["right_elbow_angles_pu", "right_hip_angles_pu", "right_knee_angles_pu"], 
                                                            get_trainers_angles("push_up"), 
                                                            df_analisis))
           
      elif df_aprox_exercise_by_date['id_exercise'][i] == 'bird_dog':

         aproxs_by_date.append(dashboard.get_aprox_exercise(["right_shoulder_angles_bd" , "right_hip_angles_bd", "right_knee_angles_bd" , "left_knee_angles_bd", "right_elbow_angles_bd", "left_elbow_angles_bd"],
                                                            get_trainers_angles('bird_dog'), 
                                                            df_analisis))

      elif df_aprox_exercise_by_date['id_exercise'][i] == 'forward_lunge':

         aproxs_by_date.append(dashboard.get_aprox_exercise(["right_hip_angles_fl","right_knee_angles_fl", "left_knee_angles_fl"],
                                                            get_trainers_angles("forward_lunge"), 
                                                            df_analisis))

      elif df_aprox_exercise_by_date['id_exercise'][i] == 'front_plank':

         aproxs_by_date.append(dashboard.get_aprox_exercise(["right_shoulder_angles_fp", "right_hip_angles_fp", "right_ankle_angles_fp"],
                                                            get_trainers_angles("front_plank"), 
                                                            df_analisis))

      elif df_aprox_exercise_by_date['id_exercise'][i] == 'curl_up':

         aproxs_by_date.append(dashboard.get_aprox_exercise(["right_shoulder_angles_cu", "right_hip_angles_cu", "right_knee_angles_cu"],
                                                            get_trainers_angles("curl_up"), 
                                                            df_analisis))


   df_aprox_exercise_by_date['Aproximacion'] = aproxs_by_date

   return df_aprox_exercise_by_date
# 3. HTML CODE
#############
st.set_page_config(
    page_title="STARTER TRAINING - UPC",
    page_icon ="01. webapp_img/upc_logo.png",
)

# --- USER AUTHENTICATION ---
users = db.fetch_all_users()

usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,"app_training", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

img_upc = get_base64_of_bin_file('01. webapp_img/upc_logo_50x50.png')
fontProgress = get_base64_of_bin_file('01. webapp_fonts/ProgressPersonalUse-EaJdz.ttf')

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        top: 60px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    @font-face {{
        font-family: ProgressFont;
        src: url("data:image/png;base64,{fontProgress}");
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: ProgressFont;    
    }}
    .main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    .css-17qbjix {{
        font-size: 16px;
    }}
    .css-12oz5g7 {{
        padding-top: 3rem;
    }}
    .stButton{{
        text-align: center !important;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)


# 4. PYTHON CODE
#############load
pygame.mixer.init()
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if 'camera' not in st.session_state:
        st.session_state['camera'] = 0

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    placeholder_title = st.empty()
    placeholder_title.title('STARTER TRAINING')

    st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)

      # ---- SIDEBAR ----
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Bienvenido {name}")

    app_exercise = st.sidebar.selectbox('Seleccione su opcion:',
        ['INICIO','EJERCICIOS', 'REPORTES']
    )

    if app_exercise == "EJERCICIOS":
        app_mode = st.sidebar.selectbox('Seleccione su Ejerccicio:',
            ['Push Up', 'Curl Up', 'Front Plank', 'Forward Lunge', 'Bird Dog']
        )

    id_trainer = randrange(3) + 1

    exercise_to_do = {}

    if app_exercise =='INICIO':
        load_home()
    elif app_exercise =='REPORTES':

        load_reportes()
    else:
        # if app_mode =='Squats':
        #     id_exercise = 'squats'
        if app_mode =='Push Up':
            id_exercise = 'push_up'

        elif app_mode =='Curl Up':
            id_exercise = 'curl_up'

        elif app_mode =='Front Plank':
            id_exercise = 'front_plank'

        elif app_mode =='Forward Lunge':
            id_exercise = 'forward_lunge'

        elif app_mode =='Bird Dog':
            id_exercise = 'bird_dog'

        else:
            id_exercise = None

        print_sidebar_main(id_exercise)

        #MAIN-SCREEN START
        st.session_state.count_pose_g = 0
        st.session_state.count_pose   = 0
        st.session_state.count_rep    = 0
        st.session_state.count_set    = 0
        finishexercise = False

        # total_poses = Sets x Reps x N° Poses
        st.session_state.total_poses = st.session_state.n_sets * st.session_state.n_reps * st.session_state.n_poses
        exercise_control, exercise_number_set, exercise_number_rep, exercise_number_pose, exercise_number_pose_global, exercise_status  = st.columns(6)
            
        with exercise_control:
            placeholder_button_status = st.empty()
            placeholder_button_status.info('PRESS START', icon="📹")
            st.markdown("<br>", unsafe_allow_html=True)
            webcam = st.button("START / STOP")
    
        with exercise_number_set:
            placeholder_set = st.empty()
            placeholder_set.metric("SET DONE", "0 / "+ str(st.session_state.n_sets), "+1 set")

        with exercise_number_rep:
            placeholder_rep = st.empty()
            placeholder_rep.metric("REPETITION DONE", "0 / "+ str(st.session_state.n_reps), "+1 repetition")

        with exercise_number_pose:
            placeholder_pose = st.empty()
            placeholder_pose.metric("POSE DONE", "0 / "+ str(st.session_state.n_poses), "+1 pose")
        
        with exercise_number_pose_global:
            placeholder_pose_global = st.empty()
            placeholder_pose_global.metric("POSE GLOBAL DONE", "0 / " + str(st.session_state.total_poses), "+1 pose")

        with exercise_status:
            placeholder_status = st.empty()
            st.markdown("<br>", unsafe_allow_html=True)
            placeholder_status.markdown(font_size_px("⛽ READY?"), unsafe_allow_html=True)

        st.markdown('---')

        trainer, user = st.columns(2)

        with trainer:        
            st.markdown("**TRAINER**", unsafe_allow_html=True)
            placeholder_trainer = st.empty()
            placeholder_trainer.image("./01. webapp_img/trainer.png")

        with user:
            st.markdown("**USER**", unsafe_allow_html=True)
            stframe = st.empty()
            #stframe.image("./01. webapp_img/user.png")

            #Cris-DM pasar a función - inicio
            df_trainer_coords = get_trainer_coords(id_exercise, id_trainer)
            df_trainers_angles = get_trainers_angles(id_exercise)
            #Cris-DM pasar a función - fin
        
        st.markdown('---')
        placeholder_results_1 = st.empty()
        placeholder_results_2 = st.empty()
        
        with exercise_control:
            if(webcam):
                video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

                # Cámara apagada
                if st.session_state['camera'] % 2 != 0:
                    placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                    st.session_state['camera'] += 1
                    video_capture.release()
                    cv2.destroyAllWindows()
                    stframe.image("./01. webapp_img/user.png")

                # Cámara encendida
                else: 
                    placeholder_button_status.success('CAMERA ON  🔴', icon="📹")
                    st.session_state['camera'] += 1

                    st.session_state.count_set = 0
                    st.session_state.count_rep = 0
                    
                    
                    N = 5
                    placeholder_trainer.image("./01. webapp_img/warm_up.gif")
                    stframe.image("./01. webapp_img/warm_up.gif")
                    mstart = "Por favor asegurese que su dispositivo pueda ver su cuerpo completo en su pantalla"
                    speak_start_msg = threading.Thread(target=speak, args=(mstart,))
                    speak_start_msg.start()
                    time.sleep(2)
                    for secs in range(N,0,-1):
                        ss = secs%60
                        placeholder_status.markdown(font_size_px(f"🏁 START IN {ss:02d}"), unsafe_allow_html=True)
                        time.sleep(1)
                    placeholder_trainer.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + "1.png")
                    body_language_class = id_exercise

                    ############################################################
                    ##               📘 RESULT DATAFRAME (INICIO)             ##
                    ############################################################
                    df_results = ut.create_df_results()
                    st.session_state.inicio_rutina = get_timestap_log()                    
                    ############################################################
                    ##               📘 RESULT DATAFRAME (FIN)                ##
                    ############################################################


                    with user:
                        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                        up = False
                        down = False
                        mid = False
                        start = 0
                        
                        while st.session_state.count_set < st.session_state.n_sets:
                            stage = ""
                            st.session_state.count_rep = 0
                            flagTime = False
                            # Setup mediapipe instance
                            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                                cap.isOpened()
                                # while st.session_state.count_pose_g <= st.session_state.total_poses:
                                while st.session_state.count_rep < st.session_state.n_reps:
                                    
                                    ret, frame = cap.read()
                                    if ret == False:
                                        break
                                    frame = cv2.flip(frame,1)
                                    height, width, _ = frame.shape
                                    # Recolor image to RGB
                                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image.flags.writeable = False
                                
                                    # Make detection
                                    results = pose.process(image)

                                    # Recolor back to BGR
                                    image.flags.writeable = True
                                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                    
                                    # Extract landmarks
                                    # try:
                                    if results.pose_landmarks is None:
                                        cv2.putText(image, 
                                        "No se han detectado ninguno de los 33 puntos corporales",
                                        (100,250),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.5,
                                        (0, 0, 255),
                                        1, 
                                        cv2.LINE_AA)
                                        stframe.image(image,channels = 'BGR',use_column_width=True)   
                                    else:
                                        ############################################################
                                        ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (INICIO)       ##
                                        ############################################################

                                        landmarks = results.pose_landmarks.landmark
                                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                                        # Concate rows
                                        row = pose_row

                                        # Make Detections
                                        X = pd.DataFrame([row])

                                        # Load Model Clasification
                                        # body_language_class = LoadModel().predict(X)[0]
                                        
                                        body_language_prob = max(LoadModel().predict_proba(X)[0])
                                        print(f'body_language_class: {body_language_class}')
                                        # print(f'body_language_probg: {LoadModel().predict_proba(X)[0]}')
                                        print(f'body_language_prob: {body_language_prob}')
                                        body_language_prob_p = round(body_language_prob*100,2)
                                        #body_language_prob_p=round(body_language_prob_p[np.argmax(body_language_prob_p)],2)

                                        ############################################################
                                        ##         🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO (FIN)          ##
                                        ############################################################
                                        if (landmarks[11].visibility >= 0.2 and landmarks[13].visibility >= 0.2 and landmarks[15].visibility >= 0.2):
                                            right_arm_x1 = int(landmarks[11].x * width) #right_elbow_angle
                                            right_arm_x2 = int(landmarks[13].x * width)
                                            right_arm_x3 = int(landmarks[15].x * width)
                                            right_arm_y1 = int(landmarks[11].y * height)
                                            right_arm_y2 = int(landmarks[13].y * height)
                                            right_arm_y3 = int(landmarks[15].y * height)  

                                            right_arm_p1 = np.array([right_arm_x1, right_arm_y1])
                                            right_arm_p2 = np.array([right_arm_x2, right_arm_y2])
                                            right_arm_p3 = np.array([right_arm_x3, right_arm_y3])

                                            right_arm_l1 = np.linalg.norm(right_arm_p2 - right_arm_p3)
                                            right_arm_l2 = np.linalg.norm(right_arm_p1 - right_arm_p3)
                                            right_arm_l3 = np.linalg.norm(right_arm_p1 - right_arm_p2)

                                            # Calculate right_elbow_angle
                                            right_elbow_angle = calculate_angleacos(right_arm_l1, right_arm_l2, right_arm_l3)
                                            print(f'right_elbow_angle: {right_elbow_angle}')
                                        else:
                                            right_arm_x1 = 0
                                            right_arm_x2 = 0
                                            right_arm_x3 = 0
                                            right_arm_y1 = 0
                                            right_arm_y2 = 0
                                            right_arm_y3 = 0 
                                            right_elbow_angle = 0

                                        if (landmarks[12].visibility >= 0.2 and landmarks[14].visibility >= 0.2 and landmarks[16].visibility >= 0.2):
                                            left_arm_x1 = int(landmarks[12].x * width) #left_elbow_angle
                                            left_arm_x2 = int(landmarks[14].x * width)
                                            left_arm_x3 = int(landmarks[16].x * width)
                                            left_arm_y1 = int(landmarks[12].y * height)
                                            left_arm_y2 = int(landmarks[14].y * height)
                                            left_arm_y3 = int(landmarks[16].y * height)  

                                            left_arm_p1 = np.array([left_arm_x1, left_arm_y1])
                                            left_arm_p2 = np.array([left_arm_x2, left_arm_y2])
                                            left_arm_p3 = np.array([left_arm_x3, left_arm_y3])

                                            left_arm_l1 = np.linalg.norm(left_arm_p2 - left_arm_p3)
                                            left_arm_l2 = np.linalg.norm(left_arm_p1 - left_arm_p3)
                                            left_arm_l3 = np.linalg.norm(left_arm_p1 - left_arm_p2)

                                            # Calculate left_elbow_angle
                                            left_elbow_angle = calculate_angleacos(left_arm_l1, left_arm_l2, left_arm_l3)
                                            print(f'left_elbow_angle: {left_elbow_angle}')
                                        else:
                                            left_arm_x1 = 0
                                            left_arm_x2 = 0
                                            left_arm_x3 = 0
                                            left_arm_y1 = 0
                                            left_arm_y2 = 0
                                            left_arm_y3 = 0
                                            left_elbow_angle = 0

                                        if (landmarks[13].visibility >= 0.2 and landmarks[11].visibility >= 0.2 and landmarks[23].visibility >= 0.2):
                                            right_shoul_x1 = int(landmarks[13].x * width) #right_shoulder_angle
                                            right_shoul_x2 = int(landmarks[11].x * width)
                                            right_shoul_x3 = int(landmarks[23].x * width)
                                            right_shoul_y1 = int(landmarks[13].y * height)
                                            right_shoul_y2 = int(landmarks[11].y * height)
                                            right_shoul_y3 = int(landmarks[23].y * height)  

                                            right_shoul_p1 = np.array([right_shoul_x1, right_shoul_y1])
                                            right_shoul_p2 = np.array([right_shoul_x2, right_shoul_y2])
                                            right_shoul_p3 = np.array([right_shoul_x3, right_shoul_y3])

                                            right_shoul_l1 = np.linalg.norm(right_shoul_p2 - right_shoul_p3)
                                            right_shoul_l2 = np.linalg.norm(right_shoul_p1 - right_shoul_p3)
                                            right_shoul_l3 = np.linalg.norm(right_shoul_p1 - right_shoul_p2)

                                            # Calculate angle
                                            right_shoulder_angle = calculate_angleacos(right_shoul_l1, right_shoul_l2, right_shoul_l3)
                                            print(f'right_shoulder_angle: {right_shoulder_angle}')
                                        else:
                                            right_shoul_x1 = 0
                                            right_shoul_x2 = 0
                                            right_shoul_x3 = 0
                                            right_shoul_y1 = 0
                                            right_shoul_y2 = 0
                                            right_shoul_y3 = 0
                                            right_shoulder_angle = 0

                                        if (landmarks[25].visibility >= 0.2 and landmarks[27].visibility >= 0.2 and landmarks[31].visibility >= 0.2):
                                            right_ankle_x1 = int(landmarks[25].x * width) #right_ankle_angle
                                            right_ankle_x2 = int(landmarks[27].x * width)
                                            right_ankle_x3 = int(landmarks[31].x * width)
                                            right_ankle_y1 = int(landmarks[25].y * height)
                                            right_ankle_y2 = int(landmarks[27].y * height)
                                            right_ankle_y3 = int(landmarks[31].y * height)  

                                            right_ankle_p1 = np.array([right_ankle_x1, right_ankle_y1])
                                            right_ankle_p2 = np.array([right_ankle_x2, right_ankle_y2])
                                            right_ankle_p3 = np.array([right_ankle_x3, right_ankle_y3])

                                            right_ankle_l1 = np.linalg.norm(right_ankle_p2 - right_ankle_p3)
                                            right_ankle_l2 = np.linalg.norm(right_ankle_p1 - right_ankle_p3)
                                            right_ankle_l3 = np.linalg.norm(right_ankle_p1 - right_ankle_p2)

                                            # Calculate angle
                                            right_ankle_angle = calculate_angleacos(right_ankle_l1, right_ankle_l2, right_ankle_l3)
                                            print(f'right_ankle_angle: {right_ankle_angle}')
                                        else:
                                            right_ankle_x1 = 0
                                            right_ankle_x2 = 0
                                            right_ankle_x3 = 0
                                            right_ankle_y1 = 0
                                            right_ankle_y2 = 0
                                            right_ankle_y3 = 0  
                                            right_ankle_angle = 0

                                        if (landmarks[11].visibility >= 0.2 and landmarks[23].visibility >= 0.2 and landmarks[25].visibility >= 0.2):
                                            right_torso_x1 = int(landmarks[11].x * width) #right_hip_angle
                                            right_torso_x2 = int(landmarks[23].x * width)
                                            right_torso_x3 = int(landmarks[25].x * width) 
                                            right_torso_y1 = int(landmarks[11].y * height)
                                            right_torso_y2 = int(landmarks[23].y * height)
                                            right_torso_y3 = int(landmarks[25].y * height) 

                                            right_torso_p1 = np.array([right_torso_x1, right_torso_y1])
                                            right_torso_p2 = np.array([right_torso_x2, right_torso_y2])
                                            right_torso_p3 = np.array([right_torso_x3, right_torso_y3])

                                            right_torso_l1 = np.linalg.norm(right_torso_p2 - right_torso_p3)
                                            right_torso_l2 = np.linalg.norm(right_torso_p1 - right_torso_p3)
                                            right_torso_l3 = np.linalg.norm(right_torso_p1 - right_torso_p2)
                                            
                                            # Calculate right_hip_angle
                                            right_hip_angle = calculate_angleacos(right_torso_l1, right_torso_l2, right_torso_l3)
                                            print(f'right_hip_angle: {right_hip_angle}')
                                        else:
                                            right_torso_x1 = 0
                                            right_torso_x2 = 0
                                            right_torso_x3 = 0
                                            right_torso_y1 = 0
                                            right_torso_y2 = 0
                                            right_torso_y3 = 0
                                            right_hip_angle = 0

                                        if (landmarks[23].visibility >= 0.2 and landmarks[25].visibility >= 0.2 and landmarks[27].visibility >= 0.2):
                                            right_leg_x1 = int(landmarks[23].x * width) #right_knee_angle
                                            right_leg_x2 = int(landmarks[25].x * width)
                                            right_leg_x3 = int(landmarks[27].x * width) 
                                            right_leg_y1 = int(landmarks[23].y * height)
                                            right_leg_y2 = int(landmarks[25].y * height)
                                            right_leg_y3 = int(landmarks[27].y * height)

                                            right_leg_p1 = np.array([right_leg_x1, right_leg_y1])
                                            right_leg_p2 = np.array([right_leg_x2, right_leg_y2])
                                            right_leg_p3 = np.array([right_leg_x3, right_leg_y3])

                                            right_leg_l1 = np.linalg.norm(right_leg_p2 - right_leg_p3)
                                            right_leg_l2 = np.linalg.norm(right_leg_p1 - right_leg_p3)
                                            right_leg_l3 = np.linalg.norm(right_leg_p1 - right_leg_p2)

                                            # Calculate angle
                                            right_knee_angle = calculate_angleacos(right_leg_l1, right_leg_l2, right_leg_l3)
                                            print(f'right_knee_angle: {right_knee_angle}')
                                        else:
                                            right_leg_x1 = 0
                                            right_leg_x2 = 0
                                            right_leg_x3 = 0
                                            right_leg_y1 = 0
                                            right_leg_y2 = 0
                                            right_leg_y3 = 0
                                            right_knee_angle = 0
                                        
                                        if (landmarks[24].visibility >= 0.2 and landmarks[26].visibility >= 0.2 and landmarks[28].visibility >= 0.2):
                                            left_leg_x1 = int(landmarks[24].x * width) #left_knee_angle
                                            left_leg_x2 = int(landmarks[26].x * width)
                                            left_leg_x3 = int(landmarks[28].x * width) 
                                            left_leg_y1 = int(landmarks[24].y * height)
                                            left_leg_y2 = int(landmarks[26].y * height)
                                            left_leg_y3 = int(landmarks[28].y * height)

                                            left_leg_p1 = np.array([left_leg_x1, left_leg_y1])
                                            left_leg_p2 = np.array([left_leg_x2, left_leg_y2])
                                            left_leg_p3 = np.array([left_leg_x3, left_leg_y3])

                                            left_leg_l1 = np.linalg.norm(left_leg_p2 - left_leg_p3)
                                            left_leg_l2 = np.linalg.norm(left_leg_p1 - left_leg_p3)
                                            left_leg_l3 = np.linalg.norm(left_leg_p1 - left_leg_p2)

                                            # Calculate angle
                                            left_knee_angle = calculate_angleacos(left_leg_l1, left_leg_l2, left_leg_l3)
                                            print(f'left_knee_angle: {left_knee_angle}')
                                        else:
                                            left_leg_x1 = 0
                                            left_leg_x2 = 0
                                            left_leg_x3 = 0
                                            left_leg_y1 = 0
                                            left_leg_y2 = 0
                                            left_leg_y3 = 0
                                            left_knee_angle = 0

                                        ############################################################
                                        ##                💰 SISTEMA COSTOS (INICIO)              ##
                                        ############################################################

                                        pose_trainer_cost_min, pose_trainer_cost_max = get_cost_pose_trainer(id_exercise, st.session_state.count_pose+1)
                                        pose_user_cost = UpcSystemCost.get_cost_pose_user(df_trainer_coords, results, st.session_state.count_pose+1)
                                        
                                        color_validation = (255, 0, 0) #Azul - dentro del rango
                                        message_validation = "Correct Position"

                                        if (pose_user_cost >= pose_trainer_cost_min) and (pose_user_cost <= pose_trainer_cost_max):

                                            color_validation = (255, 0, 0) #Azul - dentro del rango
                                            message_validation = "Correct Position"

                                        else:
                                            color_validation = (0, 0, 255) #Rojo - fuera del rango
                                            message_validation = "Wrong Position"


                                        # #1. Esquina superior izquierda: Evaluación de costos trainer vs user
                                        cv2.rectangle(image, (700,0), (415,50), (245,117,16), -1)
                                        cv2.putText(image, 
                                                    "Current pose: "+ str(st.session_state.count_pose+1),
                                                    (435,20),
                                                    cv2.FONT_HERSHEY_DUPLEX,
                                                    0.5,
                                                    (255,255,255),
                                                    1, 
                                                    cv2.LINE_AA)
                                        cv2.putText(image,
                                                    "Range: [" + str(pose_trainer_cost_min) + " - " + str(pose_trainer_cost_max) + "]", #Rango costos
                                                    (435,40),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5,
                                                    (255,255,255),
                                                    1, 
                                                    cv2.LINE_AA)

                                        # #2. Esquina superior derecha: Posición correcta/incorrecta
                                        cv2.rectangle(image, (700,70), (415,50), (255,255,255), -1)
                                        cv2.putText(image, 
                                                     "User cost: " + str(pose_user_cost), #Costo resultante 
                                                     (465,65),
                                                     cv2.FONT_HERSHEY_SIMPLEX, 
                                                     0.5,
                                                     color_validation,
                                                     1, 
                                                     cv2.LINE_AA)

                                        ############################################################
                                        ##                💰 SISTEMA COSTOS (FIN)                 ##
                                        ############################################################


                                        
                                        ############################################################
                                        ##                📐 SISTEMA ÁNGULOS (INICIO)             ##
                                        ############################################################
                                        if (pose_user_cost < pose_trainer_cost_min) or (pose_user_cost > pose_trainer_cost_max):
                                            cost_valid = "Asegúrate de imitar la pose del ejercicio"
                                            try:
                                                if not speak_stage1.is_alive():
                                                    speak_cost.start()
                                            except:
                                                try:
                                                    if not speak_stage2.is_alive():
                                                        speak_cost.start()
                                                except:
                                                    try:
                                                        if not speak_stage3.is_alive():
                                                            speak_cost.start()
                                                    except:
                                                        try:
                                                            if not speak_stage4.is_alive():
                                                                speak_cost.start()
                                                        except:
                                                            try:
                                                                if not speak_cost.is_alive():
                                                                    speak_cost.start()
                                                            except:
                                                                speak_cost = threading.Thread(target=speak, args=(cost_valid,))
                                                                speak_cost.start() 
                                        else:
                                            #Ejercicio Pushup
                                            if body_language_class == "push_up" and body_language_prob_p > 20:
                                                print(f'body_language_prob_p: {body_language_prob_p}')
                                                print(f'start: {start}')
                                                right_elbow_angle_in= get_angle(df_trainers_angles, start, 'right_elbow_angles')
                                                print(f'right_elbow_angle_in: {right_elbow_angle_in}')
                                                right_hip_angle_in=get_angle(df_trainers_angles, start, 'right_hip_angles')
                                                print(f'right_hip_angle_in: {right_hip_angle_in}')
                                                right_knee_angle_in=get_angle(df_trainers_angles, start, 'right_knee_angles')
                                                print(f'right_knee_angle_in: {right_knee_angle_in}')
                                                desv_right_elbow_angle_in=get_desv_angle(df_trainers_angles, start, 'right_elbow_angles')#10
                                                print(f'desv_right_elbow_angle: {desv_right_elbow_angle_in}')
                                                desv_right_hip_angle_in=get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#10
                                                print(f'desv_right_hip_angle: {desv_right_hip_angle_in}')
                                                desv_right_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#10
                                                print(f'desv_right_knee_angle: {desv_right_knee_angle_in}')

                                                #SUMAR Y RESTAR UN RANGO DE 10 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                                if  up == False and\
                                                    down == False and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in-desv_right_elbow_angle_in), int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,                               #18 - float - right_shoulder_angles_cu
                                                                                    None,                               #19 - float - right_hip_angles_cu
                                                                                    None,                               #20 - float - right_knee_angles_cu
                                                                                    None,                               #21 - float - right_shoulder_angles_fp
                                                                                    None,                               #22 - float - right_hip_angles_fp
                                                                                    None,                               #23 - float - right_ankle_angles_fp
                                                                                    None,                               #24 - float - right_hip_angles_fl
                                                                                    None,                               #25 - float - right_knee_angles_fl
                                                                                    None,                               #26 - float - left_knee_angles_fl
                                                                                    None,                               #27 - float - right_shoulder_angles_bd
                                                                                    None,                               #28 - float - right_hip_angles_bd
                                                                                    None,                               #29 - float - right_knee_angles_bd
                                                                                    None,                               #30 - float - left_knee_angles_bd
                                                                                    None,                               #31 - float - right_elbow_angles_bd
                                                                                    None,                               #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_elbow_angle: {right_elbow_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Primera Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == False and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in) , int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise
                                                                                    
                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,    #18 - float - right_shoulder_angles_cu
                                                                                    None,    #19 - float - right_hip_angles_cu
                                                                                    None,    #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_elbow_angle: {right_elbow_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Segunda Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == True and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in) , int(right_elbow_angle_in + desv_right_elbow_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):                      
                                                    
                                                    print(f'right_elbow_angle: {right_elbow_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Tercera Pose')

                                                    up = False
                                                    down = False
                                                    stage = "Arriba"
                                                    start = 0
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    right_elbow_angle,                  #15 - float - right_elbow_angles_pu
                                                                                    right_hip_angle,                    #16 - float - right_hip_angles_pu
                                                                                    right_knee_angle,                   #17 - float - right_knee_angles_pu
                                                                                    None,    #18 - float - right_shoulder_angles_cu
                                                                                    None,    #19 - float - right_hip_angles_cu
                                                                                    None,    #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    st.session_state.count_rep += 1
                                                    placeholder_rep.metric("REPETITION DONE", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina    

                                                else:

                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES ************************ #
                                                    
                                                    if st.session_state.count_pose+1 == 1 or st.session_state.count_pose+1 == 3:
                                                    
                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"

                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"

                                                        else:

                                                            rec_right_elbow_angle = ""

                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):

                                                            rec_right_hip_angle = ", mantén tu cadera recta"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if (right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in):

                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"

                                                        else:
                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_elbow_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    elif st.session_state.count_pose+1 == 2:

                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"

                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"

                                                        else:
                                                            rec_right_elbow_angle = ""

                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):

                                                            rec_right_hip_angle = ", mantén tu cadera recta"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"

                                                        else:
                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_elbow_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    if final_rec != "":

                                                        if st.session_state.count_pose+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 2:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 3:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                # ************************ FIN SISTEMA DE RECOMENDACIONES ************************ #                                   
                                            #Ejerccio curlup
                                            elif body_language_class == "curl_up" and body_language_prob_p > 20:
                                                print(f'body_language_prob_p: {body_language_prob_p}')
                                                print(f'start: {start}')
                                                print(f'df_trainers_angles: {df_trainers_angles}')
                                                right_shoulder_angle_in=get_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                print(f'right_shoulder_angle_in: {right_shoulder_angle_in}')
                                                right_hip_angle_in=get_angle(df_trainers_angles, start, 'right_hip_angles')
                                                print(f'right_hip_angle_in: {right_hip_angle_in}')
                                                right_knee_angle_in=get_angle(df_trainers_angles, start, 'right_knee_angles')
                                                print(f'right_knee_angle_in: {right_knee_angle_in}')
                                                desv_right_shoulder_angle_in=get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#15
                                                print(f'desv_right_shoulder_angle_in: {desv_right_shoulder_angle_in}')
                                                desv_right_hip_angle_in=get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#15
                                                print(f'desv_right_hip_angle: {desv_right_hip_angle_in}')
                                                desv_right_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#15
                                                print(f'desv_right_knee_angle: {desv_right_knee_angle_in}')

                                                #SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                                if  up == False and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in-desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Primera Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Segunda Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == True and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in),int(right_knee_angle_in + desv_right_knee_angle_in + 1)):                      
                                                    
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Tercera Pose')

                                                    up = False
                                                    down = False
                                                    stage = "Abajo"
                                                    start = 0
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    right_shoulder_angle,               #18 - float - right_shoulder_angles_cu
                                                                                    right_hip_angle,                    #19 - float - right_hip_angles_cu
                                                                                    right_knee_angle,                   #20 - float - right_knee_angles_cu
                                                                                    None,    #21 - float - right_shoulder_angles_fp
                                                                                    None,    #22 - float - right_hip_angles_fp
                                                                                    None,    #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    #####################################s######
                                                    st.session_state.count_rep += 1
                                                    placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina

                                                else:

                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES ************************ #
                                                
                                                    if start+1 == 1 or start+1 == 3:
                                                    
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"

                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"

                                                        else:
                                                            rec_right_shoulder_angle = ""

                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona más tu cadera"

                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona menos tu cadera"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tus rodillas"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tus rodillas"

                                                        else:
                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    elif start+1 == 2:

                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"

                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"

                                                        else:

                                                            rec_right_shoulder_angle = ""

                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona más tu cadera"

                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona menos tu cadera"

                                                        else:

                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tus rodillas"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tus rodillas"

                                                        else:
                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    if final_rec != "":

                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()

                                                        elif start+1 == 2:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif start+1 == 3:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES ************************ #
                                            #Ejerccio Frontplank
                                            elif body_language_class == "front_plank" and body_language_prob_p > 25:
                                                
                                                print(f'body_language_prob_p: {body_language_prob_p}')
                                                print(f'start: {start}')
                                                print(f'df_trainers_angles: {df_trainers_angles}')
                                                right_shoulder_angle_in=get_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                print(f'right_shoulder_angle_in: {right_shoulder_angle_in}')
                                                right_hip_angle_in=get_angle(df_trainers_angles, start, 'right_hip_angles')
                                                print(f'right_hip_angle_in: {right_hip_angle_in}')
                                                right_ankle_angle_in=get_angle(df_trainers_angles, start, 'right_ankle_angles')
                                                print(f'right_ankle_angle_in: {right_ankle_angle_in}')
                                                desv_right_shoulder_angle_in=get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#15
                                                print(f'desv_right_shoulder_angle_in: {desv_right_shoulder_angle_in}')
                                                desv_right_hip_angle_in=get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#15
                                                print(f'desv_right_hip_angle: {desv_right_hip_angle_in}')
                                                desv_right_ankle_angle_in=get_desv_angle(df_trainers_angles, start, 'right_ankle_angles')#15
                                                print(f'desv_right_ankle_angle_in: {desv_right_ankle_angle_in}')

                                                #SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                                if  up == False and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in-desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                    right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                    right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_ankle_angle: {right_ankle_angle}')
                                                    print(f'Paso Primera Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == False and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    flagTime = True
                                                    
                                                elif up == True and\
                                                    down == True and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in) , int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)) and\
                                                    right_hip_angle in range(int(right_hip_angle_in - desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_ankle_angle in range(int(right_ankle_angle_in - desv_right_ankle_angle_in),int(right_ankle_angle_in + desv_right_ankle_angle_in + 1)):
                                                    
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_ankle_angle: {right_ankle_angle}')
                                                    print(f'Paso Tercera Pose')
                                                    
                                                    up = False
                                                    down = False
                                                    stage = "Abajo"
                                                    start = 0
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                    right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                    right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                    None,    #24 - float - right_hip_angles_fl
                                                                                    None,    #25 - float - right_knee_angles_fl
                                                                                    None,    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    st.session_state.count_rep += 1                                                
                                                    placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina

                                                else:

                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES ************************ #

                                                    if start+1 == 1 or start+1 == 3:
                                                    
                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"

                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"

                                                        else:
                                                            rec_right_shoulder_angle = ""

                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona más tu cadera"

                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona menos tu cadera"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if right_ankle_angle > right_ankle_angle_in+desv_right_ankle_angle_in:

                                                            rec_right_ankle_angle = ", flexiona más tu tobillo derecho"

                                                        elif right_ankle_angle < right_ankle_angle_in-desv_right_ankle_angle_in:

                                                            rec_right_ankle_angle = ", flexiona menos tu tobillo derecho"

                                                        else:
                                                            rec_right_ankle_angle = ""

                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_ankle_angle
                                                        print(final_rec)

                                                    elif start+1 == 2:

                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"

                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"

                                                        else:

                                                            rec_right_shoulder_angle = ""

                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona menos tu cadera"

                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona más tu cadera"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if right_ankle_angle > right_ankle_angle_in+desv_right_ankle_angle_in:

                                                            rec_right_ankle_angle = ", flexiona más tu tobillo derecho"

                                                        elif right_ankle_angle < right_ankle_angle_in-desv_right_ankle_angle_in:

                                                            rec_right_ankle_angle = ", flexiona menos tu tobillo derecho"

                                                        else:
                                                            rec_right_ankle_angle = ""

                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_ankle_angle
                                                        print(final_rec)

                                                    if final_rec != "":

                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()

                                                        elif start+1 == 2:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif start+1 == 3:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES ************************ # 
                                            
                                            #Ejerccio forward_lunge
                                            elif body_language_class == "forward_lunge" and body_language_prob_p > 20:

                                                print(f'body_language_prob_p: {body_language_prob_p}')
                                                print(f'start: {start}')
                                                print(f'df_trainers_angles: {df_trainers_angles}')
                                                right_hip_angle_in=get_angle(df_trainers_angles, start, 'right_hip_angles')
                                                print(f'right_hip_angle_in: {right_hip_angle_in}')
                                                right_knee_angle_in=get_angle(df_trainers_angles, start, 'right_knee_angles')
                                                print(f'right_knee_angle_in: {right_knee_angle_in}')
                                                left_knee_angle_in=get_angle(df_trainers_angles, start, 'left_knee_angles')
                                                print(f'left_knee_angle_in: {left_knee_angle_in}')
                                                desv_right_hip_angle_in=get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#25
                                                print(f'desv_right_hip_angle_in: {desv_right_hip_angle_in}')
                                                desv_right_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#25
                                                print(f'desv_right_knee_angle_in: {desv_right_knee_angle_in}')
                                                desv_left_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'left_knee_angles')#25
                                                print(f'desv_left_knee_angle_in: {desv_left_knee_angle_in}')

                                                # SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                                if  up == False and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Primera Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_knee_angle in range(int(left_knee_angle_in - desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'left_knee_angle: {left_knee_angle}')
                                                    print(f'Paso Segunda Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Tercera Pose')
                                                    up = False
                                                    down = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    st.session_state.inicio_rutina = fin_rutina
                                                    #####################################s######
                                                elif up == False and\
                                                    down == True and\
                                                    mid == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_knee_angle in range(int(left_knee_angle_in - desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)):
                                                    
                                                    mid = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage4 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage4.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ########################################### 
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'left_knee_angle: {left_knee_angle}')
                                                    print(f'Paso Cuarta Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == False and\
                                                    down == True and\
                                                    mid == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'Paso Quinta Pose')

                                                    up = False
                                                    down = False
                                                    mid = False
                                                    stage = "Arriba"
                                                    start = 0
                                                    ###########################################
                                                    update_dashboard()
                                                    speak_stage5 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage5.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    right_hip_angle,                    #24 - float - right_hip_angles_fl
                                                                                    right_knee_angle,                   #25 - float - right_knee_angles_fl
                                                                                    left_knee_angle,                    #26 - float - left_knee_angles_fl
                                                                                    None,    #27 - float - right_shoulder_angles_bd
                                                                                    None,    #28 - float - right_hip_angles_bd
                                                                                    None,    #29 - float - right_knee_angles_bd
                                                                                    None,    #30 - float - left_knee_angles_bd
                                                                                    None,    #31 - float - right_elbow_angles_bd
                                                                                    None,    #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    #####################################s######
                                                    st.session_state.count_rep += 1
                                                    placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina                                                

                                                else:

                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES ************************ #

                                                    if st.session_state.count_pose+1 == 1 or st.session_state.count_pose+1 == 3 or st.session_state.count_pose+1 == 5:

                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):

                                                            rec_right_hip_angle = "Mantén tu cadera recta"

                                                        else:

                                                            rec_right_hip_angle = ""

                                                        if (right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in) or\
                                                            (right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in):

                                                            if rec_right_hip_angle != "":

                                                                rec_right_hip_angle = "Mantén tu cadera"
                                                                rec_right_knee_angle = " y tu rodilla derecha recta"

                                                            else:

                                                                rec_right_knee_angle = "Mantén tu rodilla derecha recta"
                                                        else:

                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)
                                

                                                    elif st.session_state.count_pose+1 == 2:

                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in) or\
                                                            (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):

                                                            rec_right_hip_angle = "Mantén tu cadera recta"

                                                        else:

                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:
                                                                
                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha manteniendola hacia Abajo"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:
            
                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha manteniendola hacia Abajo"                                                    

                                                        else:

                                                            rec_right_knee_angle = ""

                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda manteniendola frente a tu cadera"

                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda manteniendola frente a tu cadera"                                                    

                                                        else:

                                                            rec_left_knee_angle = ""
                                                        
                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle+rec_left_knee_angle
                                                        print(final_rec)

                                                    elif st.session_state.count_pose+1 == 4:

                                                        if (right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in):
                                                            
                                                            rec_right_hip_angle = "Flexiona más tu cadera"

                                                        elif (right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in):

                                                            rec_right_hip_angle = "Flexiona menos tu cadera"

                                                        else:

                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha manteniendola frente a tu cadera"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha manteniendola frente a tu cadera"                                                    
                                                        
                                                        else:

                                                            rec_right_knee_angle = ""

                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda manteniendola hacia Abajo"

                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda manteniendola hacia Abajo"                                                    

                                                        else:

                                                            rec_left_knee_angle = ""

                                                        final_rec = rec_right_hip_angle+rec_right_knee_angle+rec_left_knee_angle
                                                        print(final_rec)

                                                    if final_rec != "":

                                                        if st.session_state.count_pose+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 2:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 3:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 4:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage3.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif st.session_state.count_pose+1 == 5:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage4.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES ************************ #
                                            
                                            #Ejerccio bird_dog
                                            elif body_language_class == "bird_dog" and body_language_prob_p > 20:

                                                print(f'body_language_prob_p: {body_language_prob_p}')
                                                print(f'start: {start}')
                                                right_shoulder_angle_in=get_angle(df_trainers_angles, start, 'right_shoulder_angles')
                                                print(f'right_shoulder_angle_in: {right_shoulder_angle_in}')
                                                right_hip_angle_in=get_angle(df_trainers_angles, start, 'right_hip_angles')
                                                print(f'right_hip_angle_in: {right_hip_angle_in}')
                                                right_knee_angle_in=get_angle(df_trainers_angles, start, 'right_knee_angles')
                                                print(f'right_knee_angle_in: {right_knee_angle_in}')
                                                left_knee_angle_in=get_angle(df_trainers_angles, start, 'left_knee_angles')
                                                print(f'left_knee_angle_in: {left_knee_angle_in}')
                                                right_elbow_angle_in=get_angle(df_trainers_angles, start, 'right_elbow_angles')
                                                print(f'right_elbow_angle_in: {right_elbow_angle_in}')
                                                left_elbow_angle_in=get_angle(df_trainers_angles, start, 'left_elbow_angles')
                                                print(f'left_elbow_angle_in: {left_elbow_angle_in}')
                                                
                                                desv_right_shoulder_angle_in=get_desv_angle(df_trainers_angles, start, 'right_shoulder_angles')#25
                                                print(f'desv_right_shoulder_angle_in: {desv_right_shoulder_angle_in}')
                                                desv_right_hip_angle_in=get_desv_angle(df_trainers_angles, start, 'right_hip_angles')#25
                                                print(f'desv_right_hip_angle_in: {desv_right_hip_angle_in}')
                                                desv_right_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'right_knee_angles')#25
                                                print(f'desv_right_knee_angle_in: {desv_right_knee_angle_in}')
                                                desv_left_knee_angle_in=get_desv_angle(df_trainers_angles, start, 'left_knee_angles')#25
                                                print(f'desv_left_knee_angle_in: {desv_left_knee_angle_in}')
                                                desv_right_elbow_angle_in=get_desv_angle(df_trainers_angles, start, 'right_elbow_angles')#25
                                                print(f'desv_right_elbow_angle_in: {desv_right_elbow_angle_in}')
                                                desv_left_elbow_angle_in=get_desv_angle(df_trainers_angles, start, 'left_elbow_angles')#25
                                                print(f'desv_left_elbow_angle_in: {desv_left_elbow_angle_in}')

                                                #SUMAR Y RESTAR UN RANGO DE 30 PARA EL ANGULO DE CADA POSE PARA UTILIZARLO COMO RANGO 
                                                if  up == False and\
                                                    down == False and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    
                                                    up = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage1 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage1.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set +1,      #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'Paso Primera Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == True and\
                                                    down == False and\
                                                    left_knee_angle in range(int(left_knee_angle_in-desv_left_knee_angle_in), int(left_knee_angle_in + desv_left_knee_angle_in + 1)) and\
                                                    right_elbow_angle in range(int(right_elbow_angle_in - desv_right_elbow_angle_in), int(right_elbow_angle_in + desv_right_knee_angle_in + 1)):
                                                    
                                                    down = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage2 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage2.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'left_knee_angle: {left_knee_angle}')
                                                    print(f'right_elbow_angle: {right_elbow_angle}')
                                                    print(f'Paso Segunda Pose')
                                                    st.session_state.inicio_rutina = fin_rutina                                            
                                                elif up == True and\
                                                    down == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'Paso Tercera Pose')
                                                    up = False
                                                    down = True
                                                    stage = "Abajo"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage3 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage3.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == False and\
                                                    down == True and\
                                                    mid == False and\
                                                    right_knee_angle in range(int(right_knee_angle_in-desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    left_elbow_angle in range(int(left_elbow_angle_in - desv_left_elbow_angle_in), int(left_elbow_angle_in + desv_left_elbow_angle_in + 1)):
                                                    
                                                    mid = True
                                                    stage = "Arriba"
                                                    start +=1
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage4 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage4.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16- float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ############################################ 
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'left_elbow_angle: {left_elbow_angle}')
                                                    print(f'Paso Cuarta Pose')
                                                    st.session_state.inicio_rutina = fin_rutina
                                                elif up == False and\
                                                    down == True and\
                                                    mid == True and\
                                                    right_hip_angle in range(int(right_hip_angle_in-desv_right_hip_angle_in), int(right_hip_angle_in + desv_right_hip_angle_in + 1)) and\
                                                    right_knee_angle in range(int(right_knee_angle_in - desv_right_knee_angle_in), int(right_knee_angle_in + desv_right_knee_angle_in + 1)) and\
                                                    right_shoulder_angle in range(int(right_shoulder_angle_in - desv_right_shoulder_angle_in), int(right_shoulder_angle_in + desv_right_shoulder_angle_in + 1)):
                                                    
                                                    print(f'right_hip_angle: {right_hip_angle}')
                                                    print(f'right_knee_angle: {right_knee_angle}')
                                                    print(f'right_shoulder_angle: {right_shoulder_angle}')
                                                    print(f'Paso Quinta Pose')
                                                    
                                                    up = False
                                                    down = False
                                                    mid = False
                                                    stage = "Abajo"
                                                    start = 0
                                                    ############################################
                                                    update_dashboard()
                                                    speak_stage5 = threading.Thread(target=speak, args=(stage,))
                                                    speak_stage5.start()
                                                    fin_rutina = get_timestap_log()
                                                    df_results = ut.add_row_df_results(df_results,
                                                                                    id_exercise,                        #1 - str - id_exercise

                                                                                    st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                    fin_rutina,                         #3 - str - DateTime_End

                                                                                    st.session_state.n_poses,           #4 - int - n_poses
                                                                                    st.session_state.n_sets,            #5 - int - n_sets
                                                                                    st.session_state.n_reps,            #6 - int - n_reps
                                                                                    st.session_state.total_poses,       #7 - int - total_poses
                                                                                    st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                    body_language_class,                #9 - str - Class
                                                                                    body_language_prob_p,               #10 - float - Prob
                                                                                    st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                    st.session_state.count_pose,        #12 - int - count_pose
                                                                                    st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                    st.session_state.count_set + 1,     #14 - int - count_set
                                                                                    None,   #15 - float - right_elbow_angles_pu
                                                                                    None,   #16 - float - right_hip_angles_pu
                                                                                    None,   #17 - float - right_knee_angles_pu
                                                                                    None,   #18 - float - right_shoulder_angles_cu
                                                                                    None,   #19 - float - right_hip_angles_cu
                                                                                    None,   #20 - float - right_knee_angles_cu
                                                                                    None,   #21 - float - right_shoulder_angles_fp
                                                                                    None,   #22 - float - right_hip_angles_fp
                                                                                    None,   #23 - float - right_ankle_angles_fp
                                                                                    None,   #24 - float - right_hip_angles_fl
                                                                                    None,   #25 - float - right_knee_angles_fl
                                                                                    None,   #26 - float - left_knee_angles_fl
                                                                                    right_shoulder_angle,               #27 - float - right_shoulder_angles_bd
                                                                                    right_hip_angle,                    #28 - float - right_hip_angles_bd
                                                                                    right_knee_angle,                   #29 - float - right_knee_angles_bd
                                                                                    left_knee_angle,                    #30 - float - left_knee_angles_bd
                                                                                    right_elbow_angle,                  #31 - float - right_elbow_angles_bd
                                                                                    left_elbow_angle,                   #32 - float - left_elbow_angles_bd
                                                                                    pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                    pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                    pose_user_cost                      #35 - float - pose_user_cost
                                                    )
                                                    ######################################s######
                                                    st.session_state.count_rep += 1
                                                    placeholder_rep.metric("REPETITION", str(st.session_state.count_rep) + " / "+ str(st.session_state.n_reps), "+1 rep")
                                                    st.session_state.count_pose = 0
                                                    st.session_state.inicio_rutina = fin_rutina

                                                else:

                                                    # ************************ INICIO SISTEMA DE RECOMENDACIONES ************************ #

                                                    if start+1 == 1 or start+1 == 3 or start+1 == 5 : 

                                                        if right_shoulder_angle > right_shoulder_angle_in+desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona más tu hombro derecho"

                                                        elif right_shoulder_angle < right_shoulder_angle_in-desv_right_shoulder_angle_in:

                                                            rec_right_shoulder_angle = "Flexiona menos tu hombro derecho"

                                                        else:

                                                            rec_right_shoulder_angle = ""

                                                        if right_hip_angle > right_hip_angle_in+desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona más tu cadera"

                                                        elif right_hip_angle < right_hip_angle_in-desv_right_hip_angle_in:

                                                            rec_right_hip_angle = ", flexiona menos tu cadera"

                                                        else:
                                                            rec_right_hip_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"
                                                        
                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"

                                                        else:

                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_right_shoulder_angle+rec_right_hip_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    elif start+1 == 2:
                                                        if right_elbow_angle > right_elbow_angle_in+desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona más tu codo derecho"

                                                        elif right_elbow_angle < right_elbow_angle_in-desv_right_elbow_angle_in:

                                                            rec_right_elbow_angle = "Flexiona menos tu codo derecho"

                                                        else:

                                                            rec_right_elbow_angle = ""

                                                        if left_knee_angle > left_knee_angle_in+desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona más tu rodilla izquierda"

                                                        elif left_knee_angle < left_knee_angle_in-desv_left_knee_angle_in:

                                                            rec_left_knee_angle = ", flexiona menos tu rodilla izquierda"

                                                        else:

                                                            rec_left_knee_angle = ""

                                                        final_rec = rec_right_elbow_angle+rec_left_knee_angle
                                                        print(final_rec)

                                                    elif start+1 == 4:

                                                        if left_elbow_angle > left_elbow_angle_in+desv_left_elbow_angle_in:

                                                            rec_left_elbow_angle = "Flexiona más tu codo izquierdo"

                                                        elif left_elbow_angle < left_elbow_angle_in-desv_left_elbow_angle_in:

                                                            rec_left_elbow_angle = "Flexiona menos tu codo izquierdo"

                                                        else:

                                                            rec_left_elbow_angle = ""

                                                        if right_knee_angle > right_knee_angle_in+desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona más tu rodilla derecha"

                                                        elif right_knee_angle < right_knee_angle_in-desv_right_knee_angle_in:

                                                            rec_right_knee_angle = ", flexiona menos tu rodilla derecha"

                                                        else:

                                                            rec_right_knee_angle = ""

                                                        final_rec = rec_left_elbow_angle+rec_right_knee_angle
                                                        print(final_rec)

                                                    if final_rec != "":

                                                        if start+1 == 1:
                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                speak_rec.start()

                                                        elif start+1 == 2:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage1.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif start+1 == 3:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage2.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif start+1 == 4:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage3.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                        elif start+1 == 5:

                                                            try:
                                                                if not speak_rec.is_alive():
                                                                    speak_rec.start()
                                                            except:
                                                                try:
                                                                    if not speak_stage4.is_alive():
                                                                        speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                        speak_rec.start()
                                                                except:
                                                                    speak_rec = threading.Thread(target=speak, args=(final_rec,))
                                                                    speak_rec.start()

                                                    # ************************ FIN SISTEMA DE RECOMENDACIONES ************************ #
                                            else:
                                                stage = ""
                                                start = 0
                                                up = False
                                                down = False
                                                mid = False
                                                st.session_state.count_pose_g = 0 
                                                st.session_state.count_pose = 0 
                                                print(f'Salio')
                                        #else:

                                        
                                        #Codigo para actualizar pantalla por repeticiones

                                        # Setup status box
                                        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                                        
                                        # Set data
                                        cv2.putText(image, 'SET', (15,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                        cv2.putText(image, str(st.session_state.count_set), 
                                                    (10,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                        # Rep data
                                        cv2.putText(image, 'REPS', (65,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                        cv2.putText(image, str(st.session_state.count_rep), 
                                                    (60,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                                        
                                        # Stage data
                                        cv2.putText(image, 'STAGE', (115,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                        cv2.putText(image, stage, 
                                                    (110,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                        # Class data
                                        cv2.putText(image, 'CLASS', (15,427), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                        cv2.putText(image, str(body_language_class), 
                                                    (10,467), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

                                        # Prob data
                                        cv2.putText(image, 'PROB', (150,427), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                        cv2.putText(image, str(body_language_prob_p), 
                                                    (150,467), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                    
                                        if body_language_class == "push_up": 
                                            cv2.line(image, (right_arm_x1, right_arm_y1), (right_arm_x2, right_arm_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_arm_x2, right_arm_y2), (right_arm_x3, right_arm_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_arm_x1, right_arm_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_arm_x2, right_arm_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_arm_x3, right_arm_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_elbow_angle)), (right_arm_x2 + 30, right_arm_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if body_language_class == "curl_up": 
                                            cv2.line(image, (right_shoul_x1, right_shoul_y1), (right_shoul_x2, right_shoul_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_shoul_x2, right_shoul_y2), (right_shoul_x3, right_shoul_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_shoul_x1, right_shoul_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x2, right_shoul_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x3, right_shoul_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_shoulder_angle)), (right_shoul_x2 + 30, right_shoul_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if body_language_class == "front_plank": 
                                            cv2.line(image, (right_shoul_x1, right_shoul_y1), (right_shoul_x2, right_shoul_y2), (242, 14, 14), 3)
                                            cv2.line(image, (right_shoul_x2, right_shoul_y2), (right_shoul_x3, right_shoul_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (right_shoul_x1, right_shoul_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x2, right_shoul_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_shoul_x3, right_shoul_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_shoulder_angle)), (right_shoul_x2 + 30, right_shoul_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_ankle_x1, right_ankle_y1), (right_ankle_x2, right_ankle_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_ankle_x2, right_ankle_y2), (right_ankle_x3, right_ankle_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_ankle_x1, right_ankle_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_ankle_x2, right_ankle_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_ankle_x3, right_ankle_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_ankle_angle)), (right_ankle_x2 + 30, right_ankle_y2), 1, 1.5, (128, 0, 250), 2)
                                            if start == 2 and flagTime == True:
                                                keep_pose_sec = 5 
                                                cv2.putText(image, 'WAIT FOR ' + str(keep_pose_sec) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                                mifrontplank = "Mantenga la posicion " + str(keep_pose_sec) + " segundos"
                                                speak(mifrontplank)
                                                stframe.image(image,channels = 'BGR',use_column_width=True)
                                                time.sleep(keep_pose_sec)
                                                mffrontplank = "Baje!"
                                                speak_stage2 = threading.Thread(target=speak, args=(mffrontplank,))
                                                speak_stage2.start()
                                                #############################################
                                                update_dashboard()
                                                ######################################s######
                                                fin_rutina = get_timestap_log()
                                                df_results = ut.add_row_df_results(df_results,
                                                                                id_exercise,                        #1 - str - id_exercise

                                                                                st.session_state.inicio_rutina,     #2 - str - DateTime_Start
                                                                                fin_rutina,                         #3 - str - DateTime_End

                                                                                st.session_state.n_poses,           #4 - int - n_poses
                                                                                st.session_state.n_sets,            #5 - int - n_sets
                                                                                st.session_state.n_reps,            #6 - int - n_reps
                                                                                st.session_state.total_poses,       #7 - int - total_poses
                                                                                st.session_state.seconds_rest_time, #8 - int - seconds_rest_time
                                                                                body_language_class,                #9 - str - Class
                                                                                body_language_prob_p,               #10 - float - Prob
                                                                                st.session_state.count_pose_g,      #11 - int - count_pose_g
                                                                                st.session_state.count_pose,        #12 - int - count_pose
                                                                                st.session_state.count_rep + 1,     #13 - int - count_rep
                                                                                st.session_state.count_set + 1,     #14 - int - count_set
                                                                                None,   #15 - float - right_elbow_angles_pu
                                                                                None,   #16 - float - right_hip_angles_pu
                                                                                None,   #17 - float - right_knee_angles_pu
                                                                                None,   #18 - float - right_shoulder_angles_cu
                                                                                None,   #19 - float - right_hip_angles_cu
                                                                                None,   #20 - float - right_knee_angles_cu
                                                                                right_shoulder_angle,               #21 - float - right_shoulder_angles_fp
                                                                                right_hip_angle,                    #22 - float - right_hip_angles_fp
                                                                                right_ankle_angle,                  #23 - float - right_ankle_angles_fp
                                                                                None,    #24 - float - right_hip_angles_fl
                                                                                None,    #25 - float - right_knee_angles_fl
                                                                                None,    #26 - float - left_knee_angles_fl
                                                                                None,    #27 - float - right_shoulder_angles_bd
                                                                                None,    #28 - float - right_hip_angles_bd
                                                                                None,    #29 - float - right_knee_angles_bd
                                                                                None,    #30 - float - left_knee_angles_bd
                                                                                None,    #31 - float - right_elbow_angles_bd
                                                                                None,    #32 - float - left_elbow_angles_bd
                                                                                pose_trainer_cost_min,              #33 - float - pose_trainer_cost_min
                                                                                pose_trainer_cost_max,              #34 - float - pose_trainer_cost_max
                                                                                pose_user_cost                      #35 - float - pose_user_cost
                                                )
                                                ######################################s######
                                                st.session_state.inicio_rutina = fin_rutina
                                                cv2.putText(image, '' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                                stframe.image(image,channels = 'BGR',use_column_width=True)
                                                flagTime = False
                                            else:
                                                stframe.image(image,channels = 'BGR',use_column_width=True)

                                        if body_language_class == "forward_lunge": 
                                            cv2.line(image, (left_leg_x1, left_leg_y1), (left_leg_x2, left_leg_y2), (242, 14, 14), 3)
                                            cv2.line(image, (left_leg_x2, left_leg_y2), (left_leg_x3, left_leg_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (left_leg_x1, left_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x2, left_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x3, left_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(left_knee_angle)), (left_leg_x2 + 30, left_leg_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)
                                        
                                        if body_language_class == "bird_dog": 
                                            cv2.line(image, (left_leg_x1, left_leg_y1), (left_leg_x2, left_leg_y2), (242, 14, 14), 3)
                                            cv2.line(image, (left_leg_x2, left_leg_y2), (left_leg_x3, left_leg_y3), (242, 14, 14), 3)
                                            cv2.circle(image, (left_leg_x1, left_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x2, left_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (left_leg_x3, left_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(left_knee_angle)), (left_leg_x2 + 30, left_leg_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_torso_x1, right_torso_y1), (right_torso_x2, right_torso_y2), (14, 242, 59), 3)
                                            cv2.line(image, (right_torso_x2, right_torso_y2), (right_torso_x3, right_torso_y3), (14, 242, 59), 3)
                                            cv2.circle(image, (right_torso_x1, right_torso_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x2, right_torso_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_torso_x3, right_torso_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_hip_angle)), (right_torso_x2 + 30, right_torso_y2), 1, 1.5, (128, 0, 250), 2)

                                            cv2.line(image, (right_leg_x1, right_leg_y1), (right_leg_x2, right_leg_y2), (14, 83, 242), 3)
                                            cv2.line(image, (right_leg_x2, right_leg_y2), (right_leg_x3, right_leg_y3), (14, 83, 242), 3)
                                            cv2.circle(image, (right_leg_x1, right_leg_y1), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x2, right_leg_y2), 6, (128, 0, 255),-1)
                                            cv2.circle(image, (right_leg_x3, right_leg_y3), 6, (128, 0, 255),-1)
                                            cv2.putText(image, str(int(right_knee_angle)), (right_leg_x2 + 30, right_leg_y2), 1, 1.5, (128, 0, 250), 2)
                                            stframe.image(image,channels = 'BGR',use_column_width=True)

                                st.session_state.count_set += 1
                                placeholder_set.metric("SET", str(st.session_state.count_set) + " / "+ str(st.session_state.n_sets))

                                #Codigo para actualizar pantalla por serie

                                # Setup status box
                                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                                        
                                # Set data
                                cv2.putText(image, 'SET', (15,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                cv2.putText(image, str(st.session_state.count_set), 
                                                    (10,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)

                                # Rep data
                                cv2.putText(image, 'REPS', (65,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                cv2.putText(image, str(st.session_state.count_rep), 
                                                    (60,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
                                        
                                # Stage data
                                cv2.putText(image, 'STAGE', (115,20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                                cv2.putText(image, stage, 
                                                    (110,60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA) 


                                if (st.session_state.count_set!=st.session_state.n_sets):
                                    try:
                                        cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                                        cv2.putText(image, 'REST FOR ' + str(st.session_state.seconds_rest_time) + ' s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
                                        stframe.image(image,channels = 'BGR',use_column_width=True)
                                        # cv2.waitKey(1)
                                        msucessset = "Felicitaciones, vas por buen camino"
                                        time.sleep(1)
                                        speak(msucessset)
                                        time.sleep(int(st.session_state.seconds_rest_time))
                                    except:
                                        stframe.image(image,channels = 'BGR',use_column_width=True)
                                        pass 
                        update_dashboard()                    
                        cv2.rectangle(image, (50,180), (600,300), (0,255,0), -1)
                        cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
                        stframe.image(image,channels = 'BGR',use_column_width=True)
                        msucess = "Felicitaciones, bien hecho"
                        time.sleep(1)
                        speak(msucess)
                        finishexercise = True
                        #Finalización de hilos
                        speak_start_msg.join() # Mensaje de inicio ("Por favor asegurese que su dispositivo pueda ...")
                        speak_stage1.join() # Stage 1
                        speak_stage2.join() # Stage 2
                        speak_stage3.join() # Stage 3
                        if id_exercise == "forward_lunge" or id_exercise == "bird_dog": #Se terminan los hilos de ejercicios de más de 3 poses
                            speak_stage4.join() # Stage 4
                            speak_stage5.join() # Stage 5
                        try:
                            speak_rec.join() # Recomendaciones según pose y articulaciones
                        except:
                            print("No hubieron recomendaciones")
                        time.sleep(5)          
                        cap.release()
                        cv2.destroyAllWindows()

                    placeholder_button_status.warning('CAMERA OFF ⚫', icon="📹")
                    st.session_state['camera'] += 1
                    video_capture.release()
                    cv2.destroyAllWindows()

                    st.balloons()
                    placeholder_results_1.markdown(font_size_px("RESULTADOS"), unsafe_allow_html=True)

                    #Cargar dataset de resultados
                    timestamp_show_results = get_timestap_txt(username,id_exercise)
                    df_results.to_csv(f'03. users/{timestamp_show_results}.csv', index=False)

                    #placeholder_results_2.dataframe(df_results)
                    placeholder_results_2.dataframe(dashboard.get_result_filter_dataframe(df_results, st.session_state.articulaciones, st.session_state.posfijo))

        # Recent training
        if finishexercise == True:


            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown(font_size_px("Métricas del entrenamiento"), unsafe_allow_html=True)

            dateTime_start_str, dateTime_end_str, minutes, seconds = dashboard.get_training_time(df_results)
            burned_kcal_min, burned_kcal_factor, kcal_min_calc_info, kcal_table, col_value = dashboard.\
                get_burned_calories(minutes, id_exercise, st.session_state.peso)

            st.text("🏁 Inicio entrenamiento :{}".format(dateTime_start_str))
            st.text("🥇 Fin entrenamiento    :{}".format(dateTime_end_str))
            st.text("🕒 Tiempo entrenamiento :{:.2f} minutos y {:.2f} segundos".format(minutes, seconds))
            st.text("⚖️ Peso                 :{:.2f} Kg".format(st.session_state.peso))
            st.text("🧨 Factor de Calorías   :{:.2f} KCal (Fuente: {})".format(burned_kcal_factor, kcal_min_calc_info))
            st.text("🔥 Calorías quemadas    :{:.2f} calorías".format(burned_kcal_min))
            st.dataframe(kcal_table.style.highlight_max(col_value, axis=0))

            st.markdown("<hr/>", unsafe_allow_html=True)

            system_pred_exercise, system_angles, system_cost = st.tabs(["🏃‍♀️ SISTEMA PREDICCIÓN EJERCICIO", "📐 SISTEMA ÁNGULOS", "💰 SISTEMA COSTOS"])

            if (st.session_state.n_poses == 3):
# 1.🏃‍♀️ SISTEMA CLASIFICACIÓN EJERCICIO ## 3 POSES ###                 
                with system_pred_exercise:
                    st.markdown("**🏃‍♀️ SISTEMA CLASIFICACIÓN EJERCICIO**", unsafe_allow_html=True)

                    metrica_global_plot, metrica_global_desc = st.columns(2)
                    with metrica_global_plot:
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[:, 'Prob'].mean(), 
                                                                         (("Aprox(%) Media Total {}").format(st.session_state.short_name)), 0, 16, 32, 48))
                    with metrica_global_desc:
                        st.caption("📌 *'El SISTEMA CLASIFICACIÓN EJERCICIO según Pose'* evalúa cada pose y a través de algoritmos de Machine Learning identifica a cual tiene una mayor aproximación")
                        st.caption("🎗️ Se evalua la aproximación de cada una de las {} poses".format(str(st.session_state.n_poses)))
                    
                #POSE 1
                    pose_1_spe, pose_2_spe, pose_3_spe = st.columns(3)
                    with pose_1_spe:
                        st.markdown("**Pose 1**", unsafe_allow_html=True)                        
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==1].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 1", 0, 16, 32, 48))
                #POSE 2
                    with pose_2_spe:
                        st.markdown("**Pose 2**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==2].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 2", 0, 16, 32, 48))
                #POSE 3    
                    with pose_3_spe:
                        st.markdown("**Pose 3**", unsafe_allow_html=True)
                        st.plotly_chart(dashboard.plot_aprox_gauge_chart(df_results.loc[df_results['count_pose']==3].\
                                                                        loc[:, 'Prob'].mean(), "Aprox(%) Media Pose 3", 0, 16, 32, 48))
# 2.📐 SISTEMA ÁNGULOS ## 3 POSES ### 
                with system_angles:
                    st.markdown("**📐 SISTEMA ÁNGULOS**", unsafe_allow_html=True)
                    articulaciones = dashboard.get_articulaciones_list(st.session_state.articulaciones)

                    st.markdown("Articulaciones evaluadas angularmente en este ejercicio:")
                    st.markdown("<br><br>", unsafe_allow_html=True)

                    for articulacion in articulaciones:
                        #st.markdown(articulacion)
                        #st.markdown(dashboard.get_articulacion_name(articulacion))
                        st.markdown("       🧬 __" + (dashboard.get_articulacion_name(articulacion)[0]).upper() + "__")
                        imgg = dashboard.plot_articulacion_performance_by_exerc(articulacion, id_exercise, df_results, st.session_state.posfijo)
                        st.pyplot(imgg)
                        st.markdown("<br><br>", unsafe_allow_html=True)
                    

                    st.plotly_chart(dashboard.plot_aprox_gauge_chart(aprox_exercise, "Aproximación Angular %", 0, 16, 32, 48))
                    
                    pose_1_sa, pose_2_sa, pose_3_sa = st.columns(3)
                    with pose_1_sa:
                        img_path1 = "./02. trainers/" + id_exercise + "/images/" + id_exercise + "1_results.png"
                        id_pose=1
                        img1 = dashboard.add_results_angle_img(img_path1, id_exercise, id_pose, df_results, 
                                                            st.session_state.articulaciones, st.session_state.posfijo)
                        st.image(img1)
                        
                        st.image("./02. trainers/" + id_exercise + "/images/" + id_exercise + "1_results.png")
                        
                    with pose_2_sa:
                        a=0
                    with pose_3_sa:
                        a=0
                st.markdown("<hr/>", unsafe_allow_html=True)


# 3.📐 💰 SISTEMA COSTOS ## 3 POSES ###                
                with system_cost:
                    st.markdown("**💰 SISTEMA COSTOS**", unsafe_allow_html=True)
                    st.plotly_chart(dashboard.plot_cost(
                        df_results.loc[:, 'pose_trainer_cost_min'].mean(), 
                        df_results.loc[:, 'pose_trainer_cost_max'].mean(), 
                        df_results.loc[:, 'pose_user_cost'].mean(), 
                        "Aproximación Costos %"))
                    
# 1.🏃‍♀️ SISTEMA CLASIFICACIÓN EJERCICIO ## 5 POSES ###             
            elif(st.session_state.n_poses == 5):
                a=1

            
            
            
            st.markdown("<hr/>", unsafe_allow_html=True)



            
            
            