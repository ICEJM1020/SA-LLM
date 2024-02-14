""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-02-13
""" 
import json

import pandas as pd
import numpy as np

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

from config import CONFIG



def catalog_activity(query, catalogue, retry_times=3) -> dict:
    for try_idx in range(retry_times):
        try:
            res = catalog_activity_chat(query, catalogue)
        except:
            if try_idx + 1 == retry_times:
                raise Exception("Catalog activity error")
            else:
                continue
        else:
            return res
        
def catalog_activity_chat(query, catalogue):
    human_prompt = HumanMessagePromptTemplate.from_template("Fill the following JSON format query, catalog each activity (key value) into following vague catalogues:\n{catalogue}\n\nJSON query\n{query}\n\nReturn your answer and keep the JSON format.")
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    
    request = chat_prompt.format_prompt(
        catalogue = catalogue,
        query = json.dumps(query)
    ).to_messages()

    model = ChatOpenAI(
            api_key=CONFIG["openai"]["api_key"],
            organization=CONFIG["openai"]["organization"],
            model_name='gpt-3.5-turbo',
            temperature=1.0
        )
    results = model.invoke(request)

    _dict = json.loads(results.content)

    return _dict