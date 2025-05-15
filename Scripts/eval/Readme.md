### To execute these pyhton scripts you need to pass model name and dataset name as arguments, note: datasets are different for each script:

model list can be found here: https://docs.google.com/spreadsheets/d/1bT3aud85-DFrRi29A8r2W9yT7ljHda3mLXgOMLkN2Hk/edit?gid=2114419155#gid=2114419155

Classification datasets = [newsarticles,imdb]

QA datasets = [canadianQA, openR1]

Conversational datasets = [finetome, pol-convo]

Summarization datasets = [newsroom, scisumm]

---------- 

example comand:

python eval-conversation.py --model_name SadiaK/Phi4-FT-Pol-convo --dataset_name pol-convo
