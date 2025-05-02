### To execute these pyhton scripts you need to pass model name and dataset name as arguments:
model_n = [llama3,mistral,falcon]
Classification datasets = [newsarticles,imdb]
QA datasets = [canadianQA, openR1]
example comand:
python finetuning-LLM-QA --model_n falcon --dataset_n canadianQA
