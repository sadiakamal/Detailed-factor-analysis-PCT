### To execute these pyhton scripts you need to pass model name and dataset name as arguments, note: datasets are different for each script:

model_n = [llama3,mistral,falcon]

Classification datasets = [newsarticles,imdb]

QA datasets = [canadianQA, openR1]

Conversational datasets = [finetome, pol-convo]

Summarization datasets = [newsroom, scisumm]

---------- 

To execute run the following:

python train-conversation.py --model_n mistral --dataset_n pol-convo
