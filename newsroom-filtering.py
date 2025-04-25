import requests
import json
from datasets import load_dataset
from typing_extensions import Optional
import os
from collections import defaultdict
from tqdm import tqdm

MAX_NUM_TO_CHECK_TRAIN = 5
MAX_NUM_TO_CHECK = {"train": MAX_NUM_TO_CHECK_TRAIN,
                    "validation": round(MAX_NUM_TO_CHECK_TRAIN * 0.1),
                    "test": round(MAX_NUM_TO_CHECK_TRAIN * 0.2)}
SEED = 42


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

    def run(self, url: str, summary: Optional[str] = None) -> Optional[bool]:
        prompt_base = f"you are given the following url: {url} for a news article" if summary is not None else \
            f"you are given the following url: [{url}] and summary: [{summary}] for a news article"

        prompt = (f"{prompt_base} . the article can cover political or other type of news (sports/technology/society). "
                           f"do you think the article covers political news? please answer in YES or NO. "
                           f"Do not provide any explanation")
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
    for i in tqdm(range(MAX_NUM_TO_CHECK[k]), desc=f"filtering: {k}"):
        datum = split[i]
        result = news_filter.run(datum['url'])
        if result is None:
            errors[k].append(datum)
        elif result:
            political[k].append(datum)
        else:
            non_political[k].append(datum)
    print("="*30)

print("political:", [f"{k}: {len(v)}" for k,v in political.items()])
print("non political:", [f"{k}: {len(v)}" for k,v in non_political.items()])
print("errors:", [f"{k}: {len(v)}" for k,v in errors.items()])
