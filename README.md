# GPT-2 fine-tuned for conversation aims

This repository contains a pipeline with GPT-2 model fine-tuned using LoRA and a console application 
that allows to generate answers to questions. 

You can find model [here](https://drive.google.com/drive/folders/1XU3zndn_9hLCr0JyqLJl3ruTmVfC6uPf?usp=sharing).

To run console application download the model, place the folder in the project, 
open a terminal in your project folder and use the following command: 
`python3 inference.py openai-community/gpt2 -l model_with_lora_finetuning`. 

After the model has loaded you can write your prompt and send it to the model using `Enter` key. 
To leave write an empty message to model. 