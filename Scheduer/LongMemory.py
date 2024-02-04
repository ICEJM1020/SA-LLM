""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os
import sys
sys.path.append(os.path.abspath('./'))

import pandas as pd

from config import CONFIG


class LongMemory:

    def __init__(self, description, user_folder):
        self._user_folder = user_folder
        self._memory_tree = {
            "description" : description,
        }


    @property
    def description(self):
        return self._memory_tree["description"]



class MMASH_LongMemory(LongMemory):

    def __init__(self, description, user_folder):
        super().__init__(description, user_folder)

        self._beat2beat = pd.read_csv(os.path.join(user_folder, CONFIG['MMASH_files']['beat2beat']))
        self._sensory = pd.read_csv(os.path.join(user_folder, CONFIG['MMASH_files']['sensory']))





