""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-06
""" 

from datetime import datetime, timedelta
import json

from Scheduer.utils import *



class RecordsQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def push(self, item):
        if len(self.queue) == self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def pop(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None
    
    def get_current(self):
        return self.queue[-1]
        
    def length(self):
        return len(self.queue)
        
    def fetch(self, size):
        return self.queue[-size:]


class ShortMemory:

    def __init__(self, cache_length:int=60) -> None:
        self._cur_date = ""
        self._cur_time = ""
        self._schedule = {}

        self._cur_event = {}
        self._cur_event_index = 0
        self._max_event = 25

        self._origin_decompose = []
        self._origin_decompose_index = 0
        self._cur_decompose = []
        self._cur_decompose_index = 0

        self._cur_activity = "Sleeping"
        self._activity_set = []

        self._sensor_summary = ""

        self.memory_cache = RecordsQueue(cache_length)


    @property
    def cur_date(self):
        return self._cur_date
    
    @property
    def cur_date_dt(self):
        return datetime.strptime(self._cur_date, "%m-%d-%Y")
    
    @cur_date.setter
    def cur_date(self, cur_date):
        if isinstance(cur_date, str):
            self._cur_date = cur_date
        elif isinstance(cur_date, datetime):
            self._cur_date = datetime.strftime(cur_date, "%m-%d-%Y")
        else:
            raise Exception("Current date type error, need to be \"str\" or \"datetime\"")
        
    @property
    def cur_time(self):
        return self._cur_time
    
    @property
    def cur_time_dt(self):
        return datetime.strptime(self._cur_time, "%H:%M")
    
    @cur_time.setter
    def cur_time(self, cur_time):
        if isinstance(cur_time, str):
            self._cur_time = cur_time
        elif isinstance(cur_time, datetime):
            self._cur_time = datetime.strftime(cur_time, "%H:%M")
        else:
            raise Exception("Current time type error, need to be \"str\" or \"datetime\"")
        
    @property
    def date_time_dt(self):
        return datetime.strptime(self.date_time, "%m-%d-%Y %H:%M")

    @property
    def date_time(self):
        return f"{self._cur_date} {self._cur_time}"
    
    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule:Schedule):
        self._schedule = schedule
        self._cur_event = schedule[0]
        self._cur_event_index = 0
        self._max_event = len(schedule.keys())

    @property
    def cur_event(self):
        return self._cur_event
    
    @property
    def cur_event_str(self):
        try:
            res = f"To do {self._cur_event['event']}, from {self._cur_event['start_time']} to {self._cur_event['end_time']}"
        except:
            res="Null"
        return res

    @property
    def cur_decompose(self):
        return self._cur_decompose
    
    @cur_decompose.setter
    def cur_decompose(self, decompose):
        self._cur_decompose = decompose
        self._cur_decompose_index = 0

    @property
    def cur_activity(self):
        return self._cur_activity
    
    @cur_activity.setter
    def cur_activity(self, activity):
        self._cur_activity = activity
        self.memory_cache.push({
            "time" : self.date_time,
            "schedule_event" : self._cur_event["event"],
            "activity" : activity,
            "sensor_summary" : self._sensor_summary
        })

    @property
    def last_event(self):
        if self._cur_event_index == 0:
            return "sleep until now"
        else:
            return json.dumps(self._schedule[self._cur_event_index - 1])
        
    @property
    def next_event(self):
        if self._cur_event_index == self._max_event - 1:
            return "Undecided"
        else:
            return json.dumps(self._schedule[self._cur_event_index + 1])
        
    @property
    def planning_activity(self):
        _decompose_entry = self._cur_decompose[self._cur_decompose_index]
        start_time = datetime.strptime(_decompose_entry["start_time"], "%m-%d-%Y %H:%M")
        end_time = datetime.strptime(_decompose_entry["end_time"], "%m-%d-%Y %H:%M")

        if start_time <= self.date_time_dt < end_time:
            return _decompose_entry['activity']
        else:
            if not self._cur_decompose_index == len(self._cur_decompose) - 1:
                self._cur_decompose_index += 1
            
            return self._cur_decompose[self._cur_decompose_index]['activity']
        
    @property
    def cur_activity_set(self):
        return self._activity_set
    
    @cur_activity_set.setter
    def cur_activity_set(self, activity_set):
        self._activity_set = activity_set

    @property
    def origin_decompose(self):
        return self._origin_decompose
    
    @origin_decompose.setter
    def origin_decompose(self, decompose):
        self._origin_decompose = decompose
        self._origin_decompose_index = 0

    @property
    def old_planning_activity(self):
        _decompose_entry = self._origin_decompose[self._origin_decompose_index]
        start_time = datetime.strptime(_decompose_entry["start_time"], "%m-%d-%Y %H:%M")
        end_time = datetime.strptime(_decompose_entry["end_time"], "%m-%d-%Y %H:%M")

        if start_time <= self.date_time_dt < end_time:
            return _decompose_entry['activity']
        else:
            if not self._origin_decompose_index == len(self._origin_decompose) - 1:
                self._origin_decompose_index += 1
            return self._origin_decompose[self._origin_decompose_index]['activity']

    @property
    def sensor_summary(self):
        return self._sensor_summary
    
    @sensor_summary.setter
    def sensor_summary(self, summary):
        self._sensor_summary = summary


    def csv_record(self):
        entry = self.memory_cache.get_current()
        return f"{entry['time']},\"{entry['activity']}\",\"{self.old_planning_activity}\",\"{entry['schedule_event']}\",\"{entry['sensor_summary']}\"\n"


    def check_new_event(self):
        if self.date_time_dt == datetime.strptime(self._cur_event["end_time"], "%m-%d-%Y %H:%M"):
            self._cur_event_index += 1
            try:
                self._cur_event = self._schedule[self._cur_event_index]
            except:
                return False
            return True
        return False
    

    def check_end_schedule(self):
        return self._cur_event_index == self._max_event
    

    def fetch_records(self, num_items):
        return self.memory_cache.fetch(num_items)
        
    

    
        
    

