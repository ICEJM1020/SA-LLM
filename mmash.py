""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os
import json
from datetime import datetime

import pandas as pd

from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

from config import CONFIG
from Scheduer import Scheduer

try:
    with open("Prompts/prompt_description.json", 'r') as f:
        prompts = json.load(f)['MMASH']
except:
    raise Exception("No description prompts file")

def fetch_info(user_folder):
    res = {}
    info = pd.read_csv(os.path.join(user_folder, CONFIG["MMASH_files"]["info"]), index_col=0)
    res['gender'] = "Male" if info["Gender"][0].upper() == "M" else "F"
    res['age'] = info["Age"][0]
    if res['age'] == 0: res['age'] = 27 # deal with 0 age, 27 years old is the avg age in the dataset
    res['weight'] = info["Weight"][0]
    res['height'] = info["Height"][0]

    status = pd.read_csv(os.path.join(user_folder, CONFIG["MMASH_files"]["status"]), index_col=0)
    res['meq'] = status['MEQ'][0]
    res['stai1'] = status['STAI1'][0]
    res['stai2'] = status['STAI2'][0]
    res['psqi'] = status['Pittsburgh'][0]
    res['bis'] = status['BISBAS_bis'][0]
    res['bis_drive'] = status['BISBAS_drive'][0]
    res['bis_fun'] = status['BISBAS_fun'][0]
    res['bis_reward'] = status['BISBAS_reward'][0]
    res['stress'] = status['Daily_stress'][0]

    return res


def generate_description_zero_shot(info:dict, length:int):
    prompt = PromptTemplate(
                input_variables=list(info.keys()) + ["template"],
                template=prompts["prompt"]
            )

    chain = LLMChain(
            llm=ChatOpenAI(
                    api_key=CONFIG["openai"]["api_key"],
                    organization=CONFIG["openai"]["organization"],
                    model_name='gpt-3.5-turbo',
                    temperature=0.9
                ),
            prompt=prompt
        )

    info.update({"template":prompts["template"], "length":100 if length=="long" else 50})
    description = chain.invoke(input=info)
    return description["text"]


def generate_description_few_shots(info:dict, length:str):
    example_prompt = PromptTemplate(
        input_variables=['description'],
        template=prompts["example_prompt"]
    )

    if length == "long":
        examples = prompts["example"]
    else:
        examples = prompts["example_short"]

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prompts["prefix"],
        suffix=prompts["suffix"],
        input_variables=list(info.keys()) + ["template"],
        example_separator="\n\n"
    )

    chain = LLMChain(
            llm=ChatOpenAI(
                api_key=CONFIG["openai"]["api_key"],
                organization=CONFIG["openai"]["organization"],
                model_name='gpt-3.5-turbo',
                temperature=0.9
            ),
            prompt=few_shot_prompt
        )

    info.update({"template":prompts["template"], "length":100 if length=="long" else 50})
    description = chain.invoke(input=info)
    return description["text"]


def fetch_user_folder(user_index):
    users_folder = []
    users = []
    if user_index == "all":
        for _path, dirs, _ in os.walk(CONFIG['MMASH']):
            if dirs:
                users = dirs
            else:
                users_folder.append(_path)
    elif isinstance(user_index, list):
        for user_id in user_index:
            _id = f"user_{user_id}"
            _path = os.path.join(CONFIG['MMASH'], _id)
            if os.path.exists(_path):
                users.append(f"user_{user_id}")
                users_folder.append(_path)
            else:
                print(f"Wrong user index {user_id}")
    elif isinstance(user_index, int):
        _id = f"user_{user_index}"
        _path = os.path.join(CONFIG['MMASH'], _id)
        if os.path.exists(_path):
            users.append(f"user_{user_index}")
            users_folder.append(_path)
        else:
            raise Exception(f"Wrong user index {user_id}")
    
    return users, users_folder

def run_mmasch(
        schedule_type: str = "label",
        user_index: str | list[int] | int = "all",
        result_folder: str = "output",
        description_type: str = "few-shots",
        description_length: str = "long"
    ):
    

    out_folder = os.path.join(result_folder, "mmash")
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    ##
    ## separate activity and event ?
    ##
    
    users, users_folder = fetch_user_folder(user_index=user_index)

    for user, user_folder in zip(users, users_folder):
        info = fetch_info(user_folder=user_folder)

        if description_type == "few-shots":
            description = generate_description_few_shots(info=info, length=description_length)
        elif description_type == "zero-shot":
            description = generate_description_zero_shot(info=info, length=description_length)
        else:
            raise Exception("Prompt learning type error, it should be \"zero-shot\" or \"few-shots\"")
        
        user_out_folder = os.path.join(out_folder, user)
        if not os.path.exists(user_out_folder):
            os.mkdir(user_out_folder)


        agent = Scheduer(
            description=description,
            user_folder=user_folder,
            out_folder=user_out_folder,
            schedule_type=schedule_type,
            datatype="mmash",
            labels = list(CONFIG["MMASH_label"].values()),
        )

        ############
        ## start schedule
        ############
        agent.plan(
                days=1,
                start_time="07:00",
                end_time="10:00",
                base_date = datetime.strptime("02-01-2024", '%m-%d-%Y')
            )
        # try:
        #     agent.plan(
        #         days=1,
        #         start_time="07:00",
        #         end_time="10:00",
        #         base_date = datetime.strptime("02-01-2024", '%m-%d-%Y')
        #     )
        # except:
        #     print(f"!!!!!!!! {user} Simulation Failed !!!!!!!!")

        agent.save_info()
        


if __name__ == "__main__":
    run_mmasch(
        schedule_type = "label",
        user_index = 2,
        result_folder = "output",
        description_type = "few-shots",
        description_length = "long"
    )
