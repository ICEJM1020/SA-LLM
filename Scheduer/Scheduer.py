""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-04
""" 

import sys
import os
import json
from datetime import datetime, timedelta

from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

sys.path.append(os.path.abspath('./'))
from config import CONFIG
from Scheduer.LongMemory import MMASH_LongMemory
from Scheduer.ShortMemory import ShortMemory
from Scheduer.utils import *

class Scheduer:
    def __init__(
            self, 
            description:str, 
            user_folder:str, 
            out_folder:str, 
            datatype:str,
            schedule_type:str = "label",
            activities_by_labels:bool = True,
            labels:list[str] = [],
            retry_times = 3,
            verbose = False,
        ):
        ########
        # Memory
        ########
        try:
            with open("Prompts/prompt_schedule.json", "r") as f:
                self._schedule_prompts = json.load(fp=f)
            with open("Prompts/prompt_decompose.json", "r") as f:
                self._decompose_prompts = json.load(fp=f)
            with open("Prompts/prompt_utils.json", "r") as f:
                self._utils_prompts = json.load(fp=f)
            with open("Prompts/prompt_activity.json", "r") as f:
                self._activity_prompts = json.load(fp=f)
        except:
            raise Exception("No schedule prompts file")

        ########
        # free schedule or use label
        ########
        if schedule_type == "label":
            if not labels:
                raise Exception("schedule_type \"label\" need to set the parameter \"labels\", which is a list of of possible activities.")
            else:
                self.free = False
                self.labels = labels
        else:
            self.free = True
        ## if use labels to generate activities
        self.activities_by_labels = activities_by_labels
    
        ########
        # Memory
        ########
        self.short_memory = ShortMemory()
        
        if datatype=="mmash":
            self.long_memory = MMASH_LongMemory(description, user_folder)
        else:
            raise Exception("Scheduer datatype error, it should be one of \"mmash\"")
        
        ########
        # utils
        ########
        self._output_cache = ""
        self._output_cache_length = 500
        self._out_folder = out_folder
        self._out_activity_file = os.path.join(out_folder, "activity.csv")
        with open(self._out_activity_file, "w") as f:
            f.write("time,activity,planning_activity,event,sensor_summary\n")
        self._retry_times = retry_times
        self._verbose = verbose


    def plan(
            self, 
            days:int=1, 
            start_time:str="00:00", 
            end_time:str="23:59",
            base_date:datetime=datetime.strptime("02-01-2024", '%m-%d-%Y')
            ):
        try:
            start_time_dt = datetime.strptime(start_time, "%H:%M")
            del(start_time_dt)
        except:
            print(f"start_time format error {start_time} (should be HH:MM). Set to 00:00")

        self.short_memory.cur_date = base_date
        self.long_memory.base_date = base_date
        self.short_memory.cur_time = start_time
        ## create end_time to end the simulation in that time
        self.end_time = base_date + timedelta(days=days, hours=int(end_time.split(":")[0]), minutes=int(end_time.split(":")[1]))
        for _ in range(days + 1):
            
            _schedule = self._create_range_schedule(
                start_date=self.short_memory.cur_date,
                start_time=self.short_memory.cur_time
            )
            self.short_memory.schedule =_schedule.dump_dict()
            if CONFIG["debug"]: print(self.short_memory.schedule)

            response = self._run_schedule()
            self.save_info()

            if isinstance(response, bool):
                return True
            elif isinstance(response, str):
                self.short_memory.cur_date = self.short_memory.cur_date_dt + timedelta(days=1)
                self.short_memory.cur_time = response
            else:
                return False
    

    #### re-try wrapper
    def _create_range_schedule(self,start_date,start_time) -> Schedule:
        for try_idx in range(self._retry_times):
            try:
                _schedule = self._create_range_schedule_chat(
                    start_date=start_date,
                    start_time=start_time,
                )
            except:
                if try_idx + 1 == self._retry_times:
                    raise Exception(f"Event schedule generation failed {self._retry_times} times")
                else:
                    continue
            else:
                return _schedule


    def _create_range_schedule_chat(
                self,
                start_date,
                start_time,
                llm_temperature=1.0,
            ):
        
        ########
        ## Generate schedule examples
        ########
        # We need to add .replace("{", "{{").replace("}", "}}") after serialising as JSON
        schedule_examples = []
        for idx, entry in enumerate(self._schedule_prompts['schedule_examples']):
            schedule_examples.append({
                    "description": entry["description"], 
                    "start_time": entry["start_time"],
                    "schedule":[]
                })
            for event_entry in entry["schedule"]:
                schedule_examples[idx]["schedule"].append(
                        ScheduleEntry.model_validate(event_entry).model_dump_json().replace("{", "{{").replace("}", "}}")
                        # json.dumps(event_entry).replace("{", "{{").replace("}", "}}")
                    )
        example_prompt = PromptTemplate(
            input_variables=["description", "start_time", "schedule"],
            template=self._schedule_prompts["schedule_example_prompt"]
        )

        ########
        ## schedule few-shots examples
        #######
        schedule_parser = PydanticOutputParser(pydantic_object=Schedule)

        prompt = FewShotPromptTemplate(
            examples=schedule_examples,
            example_prompt=example_prompt,
            prefix=self._schedule_prompts["prefix"],
            suffix=self._schedule_prompts["suffix"],
            input_variables=['description', 'start_date', 'start_time'],
            partial_variables={"format_instructions": schedule_parser.get_format_instructions()},
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                    api_key=CONFIG["openai"]["api_key"],
                    organization=CONFIG["openai"]["organization"],
                    model_name='gpt-3.5-turbo-16k',
                    temperature=llm_temperature,
                    verbose=self._verbose,
                ),
                prompt=prompt,
            )
        
        results = chain.invoke(input={
            'description':self.long_memory.description,
            'start_date':start_date,
            'start_time':start_time,
            "event_examples":label_list_to_str(self._schedule_prompts["event_examples"])
        })
        response = results['text'].replace("24:00", "23:59")
        return schedule_parser.parse(response)


    def _run_schedule(self):
        self.save_info()
        
        ## decompose the first event
        self._decompose()
        while True:
            
            ## Recognize if the planning activity is reasonable
            reg_success, reg_activity, sensor_summary = self._recognize_activity()
            self.short_memory.sensor_summary = sensor_summary
            ## If success, record planning activity to short memory
            if reg_success:
                self.short_memory.cur_activity = self.short_memory.planning_activity
            ## If not success, record recognized activity to short memory, then re-decompose this event
            else:
                self.short_memory.cur_activity = reg_activity
                self._re_decompose()

            ## save to local file
            self.save_activity()
            if CONFIG["debug"]: print(f"[{self.short_memory.cur_time}] {self.short_memory.cur_event['event']}-{self.short_memory.cur_activity} ")

            ###############
            ## Update time and check end
            ###############
            self.short_memory.cur_time = self.short_memory.cur_time_dt + timedelta(minutes=1)
            if self.short_memory.cur_time_dt == datetime.strptime("00:00", "%H:%M"):
                self.short_memory.cur_date = self.short_memory.cur_date_dt + timedelta(days=1)

            if self.short_memory.check_new_event():
                self._decompose()

            if self.short_memory.check_end_schedule():
                return self.short_memory.cur_time
            
            if self.short_memory.date_time_dt == self.end_time:
                return -1
            
    def _decompose(self):
        self.short_memory.cur_activity_set = self._generate_activity()
        decompose = self._decompose_task().dump_list()
        self.short_memory.cur_decompose = decompose
        self.short_memory.origin_decompose = decompose
        if CONFIG["debug"]: print(decompose)

    def _re_decompose(self):
        decompose = self._decompose_task(re_decompose=True).dump_list()
        if decompose:
            self.short_memory.cur_decompose = decompose
        if CONFIG["debug"]: print(decompose)


    def _decompose_task(self, re_decompose=False) -> Decompose:
        for try_idx in range(self._retry_times):
            try:
                _decompose = self._decompose_task_chat(re_decompose)
                assert len(_decompose.decompose) > 0
            except:
                if try_idx + 1 == self._retry_times:
                    raise Exception(f"Event decompose failed {self.short_memory.cur_event_str} {self._retry_times} times")
                else:
                    continue
            else:
                return _decompose

    def _decompose_task_chat(self, re_decompose, llm_temperature=1.0):
        ########
        ## Generate decompose examples
        ########self.activities_by_labels
        decompose_examples = []
        for idx, entry in enumerate(self._decompose_prompts['example']):
            decompose_examples.append({
                    "event": entry["event"], 
                    "cur_activity" : entry["cur_activity"],
                    "options" : entry["options"] if self.activities_by_labels else "",
                    "decompose":[]
                })
            for event_entry in entry["decompose"]:
                decompose_examples[idx]["decompose"].append(
                        DecomposeEntry.model_validate(event_entry).model_dump_json().replace("{", "{{").replace("}", "}}")
                        # json.dumps(event_entry).replace("{", "{{").replace("}", "}}")
                    )
        example_prompt = PromptTemplate(
            input_variables=["event", "cur_activity", "options", "decompose"],
            template=self._decompose_prompts["example_prompt"]
        )

        ########
        ## event decompose few-shots examples
        #######
        decompose_parser = PydanticOutputParser(pydantic_object=Decompose)
        
        prompt = FewShotPromptTemplate(
            examples=decompose_examples,
            example_prompt=example_prompt,
            prefix=self._decompose_prompts["re_prefix"] if re_decompose else self._decompose_prompts["prefix"],
            suffix=self._decompose_prompts["suffix"],
            input_variables=['description', 'past_activity_summary','cur_activity', 'cur_event', 'cur_time','activity_option'],
            partial_variables={"format_instructions": decompose_parser.get_format_instructions()},
        )

        chain = LLMChain(
            llm=ChatOpenAI(
                    api_key=CONFIG["openai"]["api_key"],
                    organization=CONFIG["openai"]["organization"],
                    model_name='gpt-3.5-turbo-16k',
                    temperature=llm_temperature,
                    verbose=self._verbose,
                ),
                prompt=prompt,
            )
        
        results = chain.invoke(input={
            'description':self.long_memory.description,
            'cur_time':self.short_memory.cur_time,
            'cur_activity':self.short_memory.cur_activity,
            'cur_event':self.short_memory.cur_event_str,
            'past_activity_summary':self._summary_activity(),
            'activity_option': label_list_to_str(self.short_memory.cur_activity_set) if self.activities_by_labels else label_list_to_str(self.labels),
        })

        response = results['text'].replace("24:00", "23:59")
        return decompose_parser.parse(response)
    

    def _generate_activity(self) -> list:
        for try_idx in range(self._retry_times):
            try:
                _activity_list = self._generate_activity_chat()
            except:
                if try_idx + 1 == self._retry_times:
                    raise Exception(f"Generate activity list failed {self.short_memory.cur_event_str} {self._retry_times} times")
                else:
                    continue
            else:
                return _activity_list

    def _generate_activity_chat(self):
        activity_list_parser = PydanticOutputParser(pydantic_object=ActivitySet)

        human_prompt = HumanMessagePromptTemplate.from_template(self._utils_prompts["generate_activities_list"])
        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
        
        request = chat_prompt.format_prompt(
            event = self.short_memory.cur_event_str,
            example = self._utils_prompts["generate_activities_list_example"],
            catalogues = label_list_to_str(self.labels),
            format_instructions = activity_list_parser.get_format_instructions()
        ).to_messages()

        model = ChatOpenAI(
                api_key=CONFIG["openai"]["api_key"],
                organization=CONFIG["openai"]["organization"],
                model_name='gpt-3.5-turbo',
                temperature=1.5,
                verbose=self._verbose
            )
        results = model.invoke(request)

        return activity_list_parser.parse(results.content).activity


    def _summary_activity(self):
        human_prompt = HumanMessagePromptTemplate.from_template(self._utils_prompts["activity_summary"])
        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

        records = self.short_memory.fetch_records(num_items=30)
        records_str = ""
        for record in records:
            records_str += f"[{record['time']}] Event[{record['schedule_event']}] Activity[{record['activity']}] : {record['sensor_summary']}\n"

        request = chat_prompt.format_prompt(records = records_str).to_messages()

        model = ChatOpenAI(
                api_key=CONFIG["openai"]["api_key"],
                organization=CONFIG["openai"]["organization"],
                model_name='gpt-3.5-turbo',
                temperature=0.5,
                verbose=self._verbose
            )
        results = model.invoke(request)

        return results.content


    def _recognize_activity(self):
        for try_idx in range(self._retry_times):
            try:
                reg_res = self._recognize_activity_chat()
            except:
                if try_idx + 1 == self._retry_times:
                    return True, None, "Null"
                else:
                    continue
            else:
                return reg_res
    
    def _recognize_activity_chat(self):
        
        human_prompt = HumanMessagePromptTemplate.from_template(self._activity_prompts["prompt"])
        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
        
        request = chat_prompt.format_prompt(
            description=self.long_memory.description,
            cur_time=self.short_memory.cur_time,
            past_activity_summary=self._summary_activity(),
            observation=self.long_memory.summary_sensor_data(cur_time=self.short_memory.cur_time_dt, cur_date=self.short_memory.cur_date_dt),
            cur_activity=self.short_memory.planning_activity,
            options=label_list_to_str(self.short_memory.cur_activity_set)
        ).to_messages()

        model = ChatOpenAI(
                api_key=CONFIG["openai"]["api_key"],
                organization=CONFIG["openai"]["organization"],
                model_name='gpt-3.5-turbo',
                temperature=0.5,
                verbose=self._verbose
            )
        results = model.invoke(request)

        return parse_reg_activity(results.content)


    def save_info(self):
        info = {}

        info["description"] = self.long_memory.description
        info["schedule"] = self.short_memory.schedule

        with open(os.path.join(self._out_folder, "info.json"), "w") as f:
            json.dump(info, fp=f)
    
    def save_activity(self):
        # "time, activity, event, feature_summary\n"
        self._output_cache += self.short_memory.csv_record()

        if len(self._output_cache) >= self._output_cache_length:
            with open(self._out_activity_file, "a+") as f:
                f.write(self._output_cache)
            self._output_cache = ""
