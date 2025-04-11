import requests
import json
import re
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="urls.yaml")
parser.add_argument("--key", default="sapplyvalues")
args = parser.parse_args()

config = yaml.safe_load(open(args.config))[args.key]
url = config['url']
standard_options = config['standard_options']
save_loc = config['save_loc']

res = requests.get(url)
js_text = res.text

pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'
matches = re.findall(pattern, js_text)
print(f"Found {len(matches)} objects.")
output = []
for idx, match in enumerate(matches):
    item = json.loads(match)
    output.append({
        "idx": item.get('id', idx),
        "statement": item["question"],
        "options": standard_options
    })

    # Save to file
with open(save_loc, "w") as wf:
    json.dump(output, wf, indent=2)
    print(f"Saved {len(output)} questions to {save_loc}")
