import pandas as pd
from collections import defaultdict
import numpy as np
import pickle

def process_data(user_df):
  
  """
  
  Input: Dataframe containing only datapoints of one user

  Output: List of preference pairs in the format [(prompt, preferred, nonpreferred), ...]
  
  """

  data = []

  for conversation in user_df["conversation_history"]:
    turns = conversation[-1]["turn"] + 1 # grab the last turn and since it's zero indexed add 1
    for i in range(turns):
      d = np.array([])

      # Loop to find the prompt
      for j in range(len(conversation)):
        if conversation[j]["turn"] != i:
          continue

        if conversation[j]["role"] == "user":
          d = np.append(d, conversation[j]["content"])
          break

      # Find the preferred response
      for j in range(len(conversation)):
        if conversation[j]["turn"] != i:
          continue

        if conversation[j]["role"] == "model" and conversation[j]["if_chosen"] == True:
          d = np.append(d, conversation[j]["content"])
          break

      # Find the first rejected response
      for j in range(len(conversation)):
        if conversation[j]["turn"] != i:
          continue

        if conversation[j]["role"] == "model" and conversation[j]["if_chosen"] == False:
          d = np.append(d, conversation[j]["content"])
          break

      if len(d) == 3:
        data.append(d)

  data = np.array(data, dtype=object)
  return data

def get_top_users(df):

    """
    
    Input: Dataframe of Prism dataset

    Output: List containing the user_ids with the most conversation turns

    """

    counts = defaultdict(int)
    for _, dp in df.iterrows():
        counts[dp['user_id']] += dp['conversation_turns']

    preference_counts = list(counts.items())
    preference_counts.sort(key=lambda x: x[1])

    top_users = [user_id for user_id, _ in preference_counts[-10:]]
    return top_users

df = pd.read_json("hf://datasets/HannahRoseKirk/prism-alignment/conversations.jsonl",
                   lines=True)

top_users = get_top_users(df)

data = defaultdict(list)

for user_id in top_users:
    data[user_id] = process_data(df[df['user_id'] == user_id])

with open("prism_data.pkl", "wb") as f:
    pickle.dump(data, f)