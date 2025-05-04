### To execute these pyhton scripts you need to pass model name and dataset name as arguments, note: datasets are different for each script:

model_n = [llama3,mistral,falcon, phi4,gemma]

Classification datasets = [newsarticles,imdb]

QA datasets = [canadianQA, openR1]

Conversational datasets = [finetome, pol-convo]

Summarization datasets = [ newsroom, scisumm]

---------- 

example comand:

python finetuning-LLM-QA --model_n falcon --dataset_n canadianQA
