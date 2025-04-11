import requests
import json
import re

# Download the questionslong.js file
url = "https://10groups.github.io/questionslong.js"
res = requests.get(url)
js_text = res.text

# Extract the JavaScript array from the text
pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'
matches = re.findall(pattern, js_text)
print(f"Found {len(matches)} objects.")
output = []
standard_options = [
    "Strongly Agree", "Agree", "Partially Agree",
    "Neutral/Unsure/Ambivalent",
    "Partially Disagree", "Disagree", "Strongly Disagree"
]

for idx, match in enumerate(matches):
    item = json.loads(match)
    output.append({
        "idx": item['id'],
        "statement": item["question"],
        "options": standard_options
    })

    # Save to file
with open("../data/10groups.json", "w") as f:
    json.dump(output, f, indent=2)
    print(f"Saved {len(output)} questions to ../data/10groups.json")
