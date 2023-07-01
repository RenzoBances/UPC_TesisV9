import streamlit as st
import Libraries.utilitarios as util

def load_home_sidebar(edad, peso, talla, imc, perabdominal, genero):
    #SIDEBAR START    
    st.sidebar.markdown('---')
    st.sidebar.markdown('❤️ __HEALTH INFORMATION__ ', unsafe_allow_html=True)
    st.sidebar.markdown("<br/>", unsafe_allow_html=True)
    st.sidebar.markdown('🟥 __Gender:__ {}'.format('Male' if genero == 'M' else 'Female'), unsafe_allow_html=True)
    st.sidebar.markdown('🟧 __Age:__ {:.0f} years old'.format(edad), unsafe_allow_html=True)
    st.sidebar.markdown('🟨 __Weight:__ {:.2f} kg'.format(peso), unsafe_allow_html=True)
    st.sidebar.markdown('🟩 __Height:__ {:.2f} cm'.format(talla), unsafe_allow_html=True)
    st.sidebar.markdown('🟦 __BMI:__ {:.2f} Kg/m<sup>2</sup> (Body Mass Index)'.format(imc), unsafe_allow_html=True)
    st.sidebar.markdown('🟪 __Abdominal circumference:__ {:.2f} cm'.format(perabdominal), unsafe_allow_html=True)
    st.sidebar.markdown('---')

def load_home(edad, peso, talla, imc, perabdominal):
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(util.font_size_px("📏 Sistemas de Medición", 26), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_system_computer_vision, col_system_pred_exercise, col_system_angles, col_system_cost = st.columns(4)
    with col_system_computer_vision:
        st.markdown(util.font_size_px("🪬 Sistema Principal Computer Vision:", 20), unsafe_allow_html=True)
        st.markdown("* [OpenCV](%s)" % 'https://opencv.org/')
        st.markdown("* [MediaPipe Pose landmark detection](%s)" % 'https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python')
        st.image("01. webapp_img/pose_landmarks_model.jpg")
    with col_system_pred_exercise:
        st.markdown(util.font_size_px("🏃‍♀️ Sistema Predicción Ejercicio:", 20), unsafe_allow_html=True)
        st.markdown("* [Sistema detección de poses 1](%s)" % 'https://aryanvij02.medium.com/push-ups-with-python-mediapipe-open-a544bd9b4351')
        st.markdown("* [Sistema detección de poses 2](%s)" % 'https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/')
    with col_system_angles:
        st.markdown(util.font_size_px("📐 Sistema Detección Ángulos:", 20), unsafe_allow_html=True)
        st.markdown("Text", unsafe_allow_html=True)
    with col_system_cost:
        st.markdown(util.font_size_px("💰 Sistema Comparación Costos Imágenes:", 20), unsafe_allow_html=True)
        st.markdown("Text", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(util.font_size_px("🏋🏼‍♂️ Workout routines", 26), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_push_up, col_curl_up, col_front_plank, col_forward_lunge, col_bird_dog = st.columns(5)
    with col_push_up:
        st.markdown(util.font_size_px("🧎🏻‍♂️ Push Up:", 20), unsafe_allow_html=True)
        st.image("02. trainers/push_up/images/push_up.gif")
    with col_curl_up:
        st.markdown(util.font_size_px("🧍🏻‍♀️ Curl Up:", 20), unsafe_allow_html=True)
        st.image("02. trainers/curl_up/images/curl_up.gif")
    with col_front_plank:
        st.markdown(util.font_size_px("🤸🏻 Front Plank:", 20), unsafe_allow_html=True)
        st.image("02. trainers/front_plank/images/front_plank.gif")
    with col_forward_lunge:
        st.markdown(util.font_size_px("🏃🏻‍♀️ Forward Lunge:", 20), unsafe_allow_html=True)
        st.image("02. trainers/forward_lunge/images/forward_lunge.gif")
    with col_bird_dog:
        st.markdown(util.font_size_px("🧎🏻 Bird Dog:", 20), unsafe_allow_html=True)
        st.image("02. trainers/bird_dog/images/bird_dog.gif")