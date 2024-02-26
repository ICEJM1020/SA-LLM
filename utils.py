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
            temperature=0.3
        )
    results = model.invoke(request)

    _dict = json.loads(results.content)

    return _dict


def levenshtein_distance(seq1, seq2):
    """
    Compute the Levenshtein Distance between two sequences.
    Inputs:
        seq1: First sequence (list or numpy array)
        seq2: Second sequence (list or numpy array)
    Output:
        Computed Levenshtein Distance (int)
    """
    # Initialize matrix of zeros
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Populate the matrix based on Levenshtein distance algorithm
    for i in range(1, rows):
        distance_matrix[i][0] = i
    for j in range(1, cols):
        distance_matrix[0][j] = j
        
    for i in range(1, rows):
        for j in range(1, cols):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(distance_matrix[i-1][j] + 1,       # Deletion
                                        distance_matrix[i][j-1] + 1,       # Insertion
                                        distance_matrix[i-1][j-1] + cost)  # Substitution
            
    return distance_matrix[-1][-1]


def levenshtein_similarity(true, pred):
    ld = levenshtein_distance(true, pred)
    similarity = 1 - (ld / len(true))
    return similarity


def cosine_similarity(true, pred):
    """
    Compute the Cosine Similarity between two sequences.
    Inputs:
        true: true sequence (list or numpy array)
        pred: prediction sequence (list or numpy array)
    Output:
        Computed Cosine Similarity (float)
    """
    # Convert sequences to numpy arrays if they aren't already
    seq1 = np.array(true)
    seq2 = np.array(pred)
    
    # Compute the dot product between the two sequences
    dot_product = np.dot(seq1, seq2)
    
    # Compute the norm (magnitude) of each sequence
    norm_seq1 = np.linalg.norm(seq1)
    norm_seq2 = np.linalg.norm(seq2)
    
    # Compute the cosine similarity
    # Avoid division by zero
    if norm_seq1 == 0 or norm_seq2 == 0:
        return 0
    cosine_similarity = dot_product / (norm_seq1 * norm_seq2)
    
    return cosine_similarity




