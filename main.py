""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os

from config import CONFIG
from mmash import run_mmasch, evaluate_mmash, mmash_baseline

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":


    if not os.path.exists(os.path.join(CONFIG['base_dir'], "output")):
        os.mkdir(os.path.join(CONFIG['base_dir'], "output"))

    
    # run_mmasch(
    #     schedule_type = "label",
    #     user_index = [1,6,11,16,21],
    #     result_folder = "output",
    #     description_type = "few-shots",
    #     description_length = "long"
    # )

    # evaluate_mmash(
    #     # user_index="all",
    #     user_index=[1,6,11,16,21],
    #     # user_index = 8,
    #     result_folder = "output",
    # )

    mmash_baseline(
        # user_index="all",
        user_index=[1,6,11,16,21],
        result_folder = "output",
    )