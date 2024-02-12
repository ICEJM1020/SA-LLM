""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os
import sys
from datetime import datetime, timedelta
sys.path.append(os.path.abspath('./'))

import pandas as pd
import numpy as np

from config import CONFIG


class LongMemory:

    def __init__(self, description, user_folder, base_date):
        self._user_folder = user_folder
        self._description = description

        self._base_date = base_date


    @property
    def description(self):
        return self._description
    
    @property
    def base_date(self):
        return self._base_date

    @base_date.setter
    def base_date(self, base_date):
        if isinstance(base_date, str):
            self._base_date = datetime.strptime(base_date, "%m-%d-%Y")
        else:
            self._base_date = base_date
    

class MMASH_LongMemory(LongMemory):

    def __init__(self, description, user_folder, base_date=datetime.strptime("02-01-2024", "%m-%d-%Y")):
        super().__init__(description, user_folder, base_date)

        self._b2b = pd.read_csv(os.path.join(user_folder, CONFIG['MMASH_files']['beat2beat']), index_col=0)
        self._b2b.columns  = [
                "Inter-Beat Interval in seconds",
                "day",
                "time"
            ]
        self._b2b["hour"] = self._b2b["time"].apply(lambda x : x.split(":")[0])
        self._b2b["minute"] = self._b2b["time"].apply(lambda x : x.split(":")[1])
        
        self._sensory = pd.read_csv(os.path.join(user_folder, CONFIG['MMASH_files']['sensory']), index_col=0)
        self._sensory.columns  = [
                "Newton-meter Acceleration data of the X-axis",
                "Newton-meter Acceleration data of the Y-axis",
                "Newton-meter Acceleration data of the Z-axis",
                "Steps",
                "Heart Rate",
                "Inclinometer",
                "Standing",
                "Sitting",
                "Lying",
                "Vector Magnitude",
                "day",
                "time"
            ]
        self._sensory["hour"] = self._sensory["time"].apply(lambda x : x.split(":")[0])
        self._sensory["minute"] = self._sensory["time"].apply(lambda x : x.split(":")[1])


    def summary_sensor_data(self, cur_time:datetime, cur_date:datetime):
        days = (cur_date - self._base_date).days + 1
        observation = ""

        temp_sensor = self._sensory[(self._sensory["day"]==days) & (self._sensory["hour"]==cur_time.hour) & (self._sensory["minute"]==cur_time.minute)]
        if not temp_sensor.shape[0]==0:
            describe = temp_sensor.describe()
            for fea in [
                "Newton-meter Acceleration data of the X-axis",
                "Newton-meter Acceleration data of the Y-axis",
                "Newton-meter Acceleration data of the Z-axis",
                "Heart Rate","Vector Magnitude"]:

                observation += "During this minute, sensors have recorded {} for {:d} pieces. ".format(fea, temp_sensor.shape[0])
                observation += "The mean value is {:.2f} with standard deviation {:.2f}. ".format(describe[fea]['mean'], describe[fea]['std'])
                observation += "The minimum value is {:.2f}, and maximum value is {:.2f}. ".format(describe[fea]['min'], describe[fea]['max'])
                observation += "Medium value is {:.2f}.\n".format(describe[fea]['50%'])

        temp_b2b = self._b2b[(self._b2b["day"]==days) & (self._b2b["hour"]==cur_time.hour) & (self._b2b["minute"]==cur_time.minute)]
        if not temp_b2b.shape[0]==0:
            describe = temp_b2b.describe()
            for fea in ["Inter-Beat Interval in seconds"]:
                observation += "During this minute, sensors have recorded {} for {:d} pieces. ".format(fea, temp_b2b.shape[0])
                observation += "The mean value is {:.2f} with standard deviation {:.2f}. ".format(describe[fea]['mean'], describe[fea]['std'])
                observation += "The minimum value is {:.2f}, and maximum value is {:.2f}. ".format(describe[fea]['min'], describe[fea]['max'])
                observation += "Medium value is {:.2f}.\n".format(describe[fea]['50%'])

        if not temp_sensor.shape[0]==0:
            observation += f"Totally steps are {temp_sensor['Steps'].sum()}\n"
            if not temp_sensor['Inclinometer'].sum() == temp_sensor.shape[0]:
                column_sums = temp_sensor[["Standing","Sitting","Lying"]].sum()
                column_with_max_sum = column_sums.idxmax()
                observation += f"According to the inclinometer, the current body posture is {column_with_max_sum}.\n"

        return observation if observation else "No Records."




        











