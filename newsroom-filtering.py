"""
prompt based approach to filter out political news articles from the newsroom dataset. We didn't end up using it.
"""
import requests
import json
from datasets import load_dataset
from typing_extensions import Optional
import os
from collections import defaultdict
from tqdm import tqdm
from enum import Enum

MAX_NUM_TO_CHECK_TRAIN = 1000
MAX_NUM_TO_CHECK = {"train": MAX_NUM_TO_CHECK_TRAIN,
                    "validation": round(MAX_NUM_TO_CHECK_TRAIN * 0.001),
                    "test": round(MAX_NUM_TO_CHECK_TRAIN * 0.002)}
SEED = 45
START = 0

class Parts(Enum):
    URL = 1
    SUMMARY = 2
    CONTENT = 3

USE = Parts.CONTENT

PROMPT_URL = "You are an expert news classifier. You are given the following URL: [{}] for a news article."
PROMPT_SUMMARY = ("You are an expert news classifier. You are given the following summary of a news article."
                  "\nSummary: \n{}\n")
PROMPT_CONTENT = ("You are an expert news classifier. You are given the following content of a news article."
                  "\nContent: \n{}\n")

PROMPT_INSTR = """
Your task is to decide whether the article is about political news or non-political news.

Political news includes topics such as:
	•	government actions, policies, or legislation
	•	elections, political parties, or political leaders
	•	international relations and diplomacy
	•	political protests or movements
	•	public administration and governance

Non-political news includes:
	•	sports, technology, science, health, education, entertainment
	•	local crime, business news unrelated to policy
	•	lifestyle, human interest stories

Please answer only YES (if the article is political) or NO (if it is not).
Do not explain your answer.
"""

class OpenRouter:
    def __init__(self, model_name: str, key: str, role:str="user", site_url: str = "", site_name: str = ""):
        self.model_name = model_name
        self.key = key
        self.role = role
        self.site_url = site_url
        self.site_name = site_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_response(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": self.role,
                    "content": prompt
                }
            ],
            "provider": {
                "sort": "throughput"
            },
            "max_tokens": 1
        }


        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")



class PoliticalNewsFilter:
    def __init__(self, router: OpenRouter):
        self.router = router

    def run(self, url: str, summary: Optional[str] = None, content: Optional[str] = None) -> Optional[bool]:
        if USE == Parts.URL:
            prompt = f"{PROMPT_URL.format(url)} {PROMPT_INSTR}"
        elif USE == Parts.SUMMARY:
           prompt = f"{PROMPT_SUMMARY.format(summary)} {PROMPT_INSTR}"
        elif USE == Parts.CONTENT:
            prompt = f"{PROMPT_CONTENT.format(content)} {PROMPT_INSTR}"
        else:
            raise RuntimeError("prompt not defined")
        reply = self.router.get_response(prompt)
        if reply.lower() in ["yes", "no"]:
            return reply == "yes"
        return None


dataset = load_dataset("sagnikrayc/newsroom") # this is private, so you will not be able to reproduce the code exactly
model_config = {
    "model_name": "meta-llama/llama-4-scout",
    "key": open(os.path.expanduser("~/.openrouter-keys/llm-bias-shift.key")).read().strip() # again, private
}
router = OpenRouter(**model_config)
news_filter = PoliticalNewsFilter(router=router)
political = defaultdict(list)
non_political = defaultdict(list)
errors = defaultdict(list)

for k in dataset:
    split = dataset[k].shuffle(SEED)
    for i in tqdm(range(START, MAX_NUM_TO_CHECK[k]), desc=f"filtering: {k}"):
        datum = split[i]
        result = news_filter.run(datum['url'], summary=datum['summary'], content=datum["text"])
        if result is None:
            errors[k].append(datum)
        elif result:
            political[k].append(datum)
        else:
            non_political[k].append(datum)
    print("="*30)

os.makedirs("political", exist_ok=True)
os.makedirs("non-political", exist_ok=True)
os.makedirs("errors", exist_ok=True)

print("political:", [f"{k}: {len(v)}" for k,v in political.items()])
for k in dataset:
    json.dump(political[k], open(f"political/{k}.json", "w"), indent=2)

print("non political:", [f"{k}: {len(v)}" for k,v in non_political.items()])
for k in dataset:
    json.dump(political[k], open(f"non-political/{k}.json", "w"), indent=2)

print("errors:", [f"{k}: {len(v)}" for k,v in errors.items()])
for k in dataset:
    json.dump(political[k], open(f"errors/{k}.json", "w"), indent=2)
