""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta

sys.path.append(os.path.abspath('./'))

from openai import OpenAI

from Scheduer.LongMemory import MMASH_LongMemory

class Scheduer:
    def __init__(self, description:str, user_folder:str, out_folder:str, datatype:str) -> None:
        self._out_folder = out_folder
        
        if datatype=="mmash":
            self.longmem = MMASH_LongMemory(description, user_folder)
        else:
            raise Exception("Scheduer datatype error, it should be one of \"mmash\"")


    def save_info(self):
        info = {}

        info["description"] = self.longmem.description

        with open(os.path.join(self._out_folder, "info.json"), "w") as f:
            json.dump(info, fp=f)

