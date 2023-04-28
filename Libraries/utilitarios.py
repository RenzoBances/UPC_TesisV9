import pandas as pd
import numpy as np

# Utilitarios  functions
def create_df_results():
    df_results = pd.DataFrame({
        'id_exercise'               : pd.Series(dtype='str'),   #1
        'DateTime_Start'            : pd.Series(dtype='str'),   #2
        'DateTime_End'              : pd.Series(dtype='str'),   #3
        'n_poses'                   : pd.Series(dtype='int'),   #4
        'n_sets'                    : pd.Series(dtype='int'),   #5
        'n_reps'                    : pd.Series(dtype='int'),   #6
        'total_poses'               : pd.Series(dtype='int'),   #7
        'seconds_rest_time'         : pd.Series(dtype='int'),   #8
        'Class'                     : pd.Series(dtype='str'),   #9
        'Prob'                      : pd.Series(dtype='float'), #10
        'count_pose_g'              : pd.Series(dtype='int'),   #11
        'count_pose'                : pd.Series(dtype='int'),   #12
        'count_rep'                 : pd.Series(dtype='int'),   #13
        'count_set'                 : pd.Series(dtype='int'),   #14
        #push_up
        'right_elbow_angles_pu'     : pd.Series(dtype='float'), #15
        'right_hip_angles_pu'       : pd.Series(dtype='float'), #16
        'right_knee_angles_pu'      : pd.Series(dtype='float'), #17
        #curl_up
        'right_shoulder_angles_cu'  : pd.Series(dtype='float'), #18
        'right_hip_angles_cu'       : pd.Series(dtype='float'), #19
        'right_knee_angles_cu'      : pd.Series(dtype='float'), #20
        #front_plank
        'right_shoulder_angles_fp'  : pd.Series(dtype='float'), #21
        'right_hip_angles_fp'       : pd.Series(dtype='float'), #22
        'right_ankle_angles_fp'     : pd.Series(dtype='float'), #23
        #forward_lunge
        'right_hip_angles_fl'       : pd.Series(dtype='float'), #24
        'right_knee_angles_fl'      : pd.Series(dtype='float'), #25
        'left_knee_angles_fl'       : pd.Series(dtype='float'), #26
        #bird_dog
        'right_shoulder_angles_bd'  : pd.Series(dtype='float'), #27
        'right_hip_angles_bd'       : pd.Series(dtype='float'), #28
        'right_knee_angles_bd'      : pd.Series(dtype='float'), #29
        'left_knee_angles_bd'       : pd.Series(dtype='float'), #30
        'right_elbow_angles_bd'     : pd.Series(dtype='float'), #31
        'left_elbow_angles_bd'      : pd.Series(dtype='float'), #32
        'pose_trainer_cost_min'     : pd.Series(dtype='float'), #33
        'pose_trainer_cost_max'     : pd.Series(dtype='float'), #34
        'pose_user_cost'            : pd.Series(dtype='float')  #35
        })
    return df_results


def add_row_df_results(df_results,
                       id_exercise,             #1
                       DateTime_Start,          #2
                       DateTime_End,            #3
                       n_poses,                 #4
                       n_sets,                  #5
                       n_reps,                  #6
                       total_poses,             #7
                       seconds_rest_time,       #8
                       Class,                   #9
                       Prob,                    #10
                       count_pose_g,            #11
                       count_pose,              #12
                       count_rep,               #13
                       count_set,               #14
                       #push_up
                       right_elbow_angles_pu,   #15
                       right_hip_angles_pu,     #16
                       right_knee_angles_pu,    #17
                       #curl_up
                       right_shoulder_angles_cu,#18
                       right_hip_angles_cu,     #19
                       right_knee_angles_cu,    #20
                       #front_plank
                       right_shoulder_angles_fp,#21
                       right_hip_angles_fp,     #22
                       right_ankle_angles_fp,   #23
                       #forward_lunge 
                       right_hip_angles_fl,     #24
                       right_knee_angles_fl,    #25
                       left_knee_angles_fl,     #26
                       #bird_dog
                       right_shoulder_angles_bd,#27
                       right_hip_angles_bd,     #28
                       right_knee_angles_bd,    #29
                       left_knee_angles_bd,     #30
                       right_elbow_angles_bd,   #31
                       left_elbow_angles_bd,    #32
                       pose_trainer_cost_min,   #33
                       pose_trainer_cost_max,   #34
                       pose_user_cost           #35
                       ):
    
    df_results.loc[len(df_results.index)] = [
        id_exercise,        #1 - str - id_exercise
        DateTime_Start,     #2 - str - DateTime_Start
        DateTime_End,       #3 - str - DateTime_End
        n_poses,            #4 - int - n_poses
        n_sets,             #5 - int - n_sets
        n_reps,             #6 - int - n_reps
        total_poses,        #7 - int - total_poses
        seconds_rest_time,  #8 - int - seconds_rest_time
        Class,              #9 - str - Class
        Prob,               #10 - float - Prob
        count_pose_g,       #11 - int - count_pose_g
        count_pose,         #12 - int - count_pose_
        count_rep,          #13 - int - count_rep
        count_set,          #14 - int - count_set
        #push_up
        right_elbow_angles_pu,   #15 - float
        right_hip_angles_pu,     #16 - float
        right_knee_angles_pu,    #17 - float
        #curl_up
        right_shoulder_angles_cu,#18 - float
        right_hip_angles_cu,     #19 - float
        right_knee_angles_cu,    #20 - float
        #front_plank
        right_shoulder_angles_fp,#21 - float
        right_hip_angles_fp,     #22 - float
        right_ankle_angles_fp,   #23 - float
        #forward_lunge 
        right_hip_angles_fl,     #24 - float
        right_knee_angles_fl,    #25 - float
        left_knee_angles_fl,     #26 - float
        #bird_dog
        right_shoulder_angles_bd,#27 - float
        right_hip_angles_bd,     #28 - float
        right_knee_angles_bd,    #29 - float
        left_knee_angles_bd,     #30 - float
        right_elbow_angles_bd,   #31 - float
        left_elbow_angles_bd,    #32 - float
        pose_trainer_cost_min,   #33 - float
        pose_trainer_cost_max,   #34 - float
        pose_user_cost           #35 - float
        ]
    return df_results