""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os

from config import CONFIG
from mmash import run_mmasch



if __name__ == "__main__":


    if not os.path.exists(os.path.join(CONFIG['base_dir'], "output")):
        os.mkdir(os.path.join(CONFIG['base_dir'], "output"))

    
    run_mmasch(os.path.join(CONFIG['base_dir'], "output"))