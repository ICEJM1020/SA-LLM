""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import os
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import CONFIG
from Scheduer import Scheduer
from utils import *

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


def generate_label_file(user, user_out_folder, start_time:str, end_time:str, base_date:str, days:int):
    base_date_dt = datetime.strptime(base_date, '%m-%d-%Y')
    start_time_dt = datetime.strptime(f"{base_date} {start_time}", "%m-%d-%Y %H:%M")
    end_time_dt = datetime.strptime(f"{base_date} {end_time}", "%m-%d-%Y %H:%M") + timedelta(days=days)

    label = pd.read_csv(os.path.join(CONFIG["MMASH"], user+"/Activity.csv"), index_col=0)

    ################
    ## modify raw file error
    ################
    if user=="user_1":
        label.loc[30, 'End'] = "9:20"
    if user=="user_21":
        label.loc[1, 'Start'] = "10:35"
    label['End'] = label['End'].replace("24:00", "23:59")
    label['Start'] = label['Start'].replace("24:00", "00:00")

    ################
    ## delete 'saliva samples'
    ################
    label['Activity'] = label['Activity'].replace(12, np.nan)
    label = label.dropna(axis=0)

    ################
    ## generate label file
    ################
    label["start_time_dt"] = [f"{(base_date_dt + timedelta(row['Day'] - 1)).strftime('%m-%d-%Y')} {row['Start']}" for _, row in label.iterrows()]
    temp = []
    for _, row in label.iterrows():
        if datetime.strptime(row['End'], '%H:%M') < datetime.strptime(row['Start'], '%H:%M'):
            _temp_day = row['Day'] + 1
        else:
            _temp_day = row['Day']
        temp.append(f"{(base_date_dt + timedelta(_temp_day - 1)).strftime('%m-%d-%Y')} {row['End']}")
    label["end_time_dt"] = temp
    ## sort values to keep the label file so
    label = label.sort_values("start_time_dt")
    
    cur_time = start_time_dt
    _index = 0
    temp_row = label.iloc[_index]
    with open(os.path.join(user_out_folder, "label.csv"), "w") as f:
        f.write("time,activity\n")
        while True:
            if datetime.strptime(temp_row['start_time_dt'], "%m-%d-%Y %H:%M") <= cur_time < datetime.strptime(temp_row['end_time_dt'], "%m-%d-%Y %H:%M"):
                f.write(f"{cur_time.strftime('%m-%d-%Y %H:%M')},{temp_row['Activity']}\n")
            else:
                if cur_time >= datetime.strptime(temp_row['end_time_dt'], "%m-%d-%Y %H:%M") and _index<label.shape[0] - 1:
                    f.write(f"{cur_time.strftime('%m-%d-%Y %H:%M')},{temp_row['Activity']}\n")
                    while _index < label.shape[0] - 1:
                        _index += 1
                        temp_row = label.iloc[_index]
                        if cur_time <= datetime.strptime(temp_row['start_time_dt'], "%m-%d-%Y %H:%M") : break
                else:
                    f.write(f"{cur_time.strftime('%m-%d-%Y %H:%M')},Null\n")
            
            if cur_time == end_time_dt: break
            cur_time += timedelta(minutes=1)


def run_mmasch(
        schedule_type: str = "label",
        user_index: str | list[int] | int = "all",
        result_folder: str = "output",
        description_type: str = "few-shots",
        description_length: str = "long"
    ):
    ##############
    ## config
    ##############
    days = 1
    start_time = "07:00"
    end_time = "10:00"
    base_date = "02-01-2024"

    ##############
    ## fetch user data
    ##############
    out_folder = os.path.join(result_folder, "mmash")
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    users, users_folder = fetch_user_folder(user_index=user_index)

    
    for user, user_folder in zip(users, users_folder):
        print("****************************")
        print(f"** Start {user}")
        print("****************************")
        ##############
        ## generate user description
        ##############
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

        ##############
        ## create digital twin
        ##############
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
        try:
            agent.plan(
                days=days,
                start_time=start_time,
                end_time=end_time,
                base_date=datetime.strptime(base_date, '%m-%d-%Y')
            )
        except Exception as e:
            print(e)
            print(f"!!!!!!!! {user} Simulation Failed !!!!!!!!")
        except:
            print(f"!!!!!!!! {user} Simulation Failed !!!!!!!!")
        else:
            print("****************************")
            print(f"** End {user}")
            print("****************************")


def draw_prediction(user, pred_type, pred, true, user_out_folder):
    fig = plt.figure(figsize=(21,7),facecolor='white',dpi=200)
    ax = plt.subplot(1,1,1)
    ylabels = [i.split(",")[0] for i in list(CONFIG["MMASH_label"].values())]


    true.index = pd.to_datetime(true.index)
    pred.index = pd.to_datetime(pred.index)

    ax.plot(true+0.1,'|',color='#FF9843',label='Label')
    ax.plot(pred-0.1,'|',color='#3468C0',label='Pred.')

    for i in range(len(ylabels)):
        ax.hlines(y=i+0.5, xmin=true.index[0], xmax=true.index[-1], colors="grey", linestyles="dotted")

    # x tick_labels
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # set a tick every 5 minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # display the hour and minute
    # y label
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels,fontsize=12,rotation=30)

    plt.xlabel('Time',fontsize=14)
    plt.ylabel('Activity Label',fontsize=14)
    plt.title(f"{user} Prediction ({pred_type}) VS. Groundtruth", fontsize=18)
    plt.legend(fontsize=14)

    plt.savefig(os.path.join(user_out_folder, f"Prediction_{pred_type}.png"), dpi=200)


def evaluate(user, pred_type, pred, true, user_out_folder):
    assert len(pred.shape)==1, "Prediction dataframe error, should be (N, )"
    assert len(true.shape)==1, "Label dataframe error, should be (N, )"
    ## split query 
    query = []
    _temp_query = {}
    for idx, activity in enumerate(pred.value_counts().index):
        _temp_query[activity] = ""
        if (idx%49==0) and (not idx==0):
            query.append(_temp_query)
            _temp_query = {}
    if _temp_query: query.append(_temp_query)

    ## create activity to catalogue dict
    activity_dict = {}
    for sub_query in query:
        sub_dict = catalog_activity(sub_query, catalogue=list(CONFIG["MMASH_label"].values()), retry_times=3)
        activity_dict.update(sub_dict)
    activity_dict.update({"Sleeping":"sleeping", "sleeping":"sleeping", "sleep":"sleeping", "Sleep":"sleeping"})
    ## if there is list in dict, choose first
    for i in activity_dict.keys():
        if isinstance(activity_dict[i],list):
            if activity_dict[i]:
                activity_dict[i] = activity_dict[i][0]
            else:
                activity_dict[i] = "sleeping"

    ## map activity to catalogue
    pred = pred.map(activity_dict)
    pred = pred.replace("unknown", "sleeping")
    pred = pred.replace("Unknown", "sleeping")
    pred = pred.astype(str)

    ## map catalogue to integer
    invert_catalog = {v: k for k, v in CONFIG["MMASH_label"].items()}
    pred = pred.map(invert_catalog)
    pred = pred.replace(np.nan, 0)
    pred.astype(int)

    ## draw prediction
    draw_prediction(user, pred_type, pred.astype(np.float16), true.astype(np.float16), user_out_folder)

    ## use intersection of no-nan data in pred and true to evaluate
    indicse = true[true.notna()].index
    indicse = list(set(indicse) & set(pred.index))
    if len(indicse)==0: return "No intersection"

    y_true = true.loc[indicse].astype(np.float16).astype(np.int8)
    y_pred = pred.loc[indicse].astype(np.float16).astype(np.int8)

    report = classification_report(y_true=y_true, y_pred=y_pred)
    return report, activity_dict


def evaluate_mmash(
        user_index: str | list[int] | int = "all",
        result_folder: str = "output",
    ):
    ##############
    ## config
    ##############
    days = 1
    start_time = "07:00"
    end_time = "10:00"
    base_date = "02-01-2024"
    out_folder = os.path.join(result_folder, "mmash")

    users, _ = fetch_user_folder(user_index=user_index)

    for user in users:
        print(f"************* Evaluation Start {user} *************")
        user_out_folder = os.path.join(out_folder, user)
        if not os.path.exists(user_out_folder):
            continue
        ############
        ## evaluate
        ############
        generate_label_file(
            user=user,
            user_out_folder=user_out_folder,
            start_time=start_time,
            end_time=end_time,
            base_date=base_date,
            days=days
        )

        pred = pd.read_csv(os.path.join(user_out_folder, "activity.csv"), index_col=0)
        pred = pred.replace("Null", np.nan)
        pred = pred.replace("None", np.nan)
        pred_plan = pred["planning_activity"]
        pred_act = pred["activity"]
        true = pd.read_csv(os.path.join(user_out_folder, "label.csv"), index_col=0)
        true = true.replace("Null", np.nan)
        true_act = true['activity'].astype(np.float16)

        try:
            with open(os.path.join(user_out_folder, "eva.txt"), "w") as f:
                f.write("\n\n============Planning Activity============\n\n")
                report, activity_dict = evaluate(user=user, pred_type="Planning", pred=pred_plan, true=true_act, user_out_folder=user_out_folder)
                f.write(report)
                f.write("\n\n")
                f.write(json.dumps(activity_dict, indent=4))
                f.write("\n\n============Recognition Activity============\n\n")
                report, activity_dict = evaluate(user=user, pred_type="Recognition", pred=pred_act, true=true_act, user_out_folder=user_out_folder)
                f.write(report)
                f.write("\n\n")
                f.write(json.dumps(activity_dict, indent=4))
        except Exception as e:
            print(e)
            print(f"!!!! Failed {user} !!!!")

        print(f"************* Evaluation End {user} *************")

