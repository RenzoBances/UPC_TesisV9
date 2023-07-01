#!/usr/bin/python

import numpy as np
import pandas as pd
from math import acos, degrees

def get_valu_angle(df, index, part):
    angle_in=df['Angulo'][(df.pose==index+1)&(df.Articulacion==part)]
    angle_in=angle_in.iloc[0]
    return angle_in

def get_desv_angle(df, index, part):
    desv_in=df['Desviacion_estandar_f'][(df.pose==index+1)&(df.Articulacion==part)]
    desv_in=desv_in.iloc[0]
    return desv_in

def calculate_angleacos(a,b,c):
    angle = degrees(acos(max(min((a**2 + c**2 - b**2) / (2 * a * c), 1), -1)))
    if angle > 0:
        angle = int(angle)
    else:
        angle = 0
    return angle