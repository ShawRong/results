
import torch
import spacy
from myselfcheckgpt.modeling_selfcheck import SelfCheckNgram
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datetime import datetime 

def list_files_in_directory(directory):
    files = os.listdir(directory)
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]
 

def load_file_jsonl(file :str):
    data = []
    try:
        with open(file, mode='r', encoding='utf-8') as file:
            for line in file:
                # Parse each line as a JSON object
                json_obj = json.loads(line)
                data.append(json_obj)
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format. {e}")
    except Exception as e:
        print(f"Error: {e}")

    return data

# some model that is ready
not_ok = [
    './openchat-3.5-0106-gemma'
]
ok = [
    './Yi-1.5-9B-Chat',
    './llama-7b-finnish-instruct-v0.2',
    './Llama-3-Instruct-Neurona-8b-v2',
    './Qwen2-7B-Instruct',
    './SeaLLM-7B-v2.5',
    './Arcee-Spark',
    './CroissantLLMChat-v0.1',
    './internlm2-chat-7b', #need trust-remote-mode
    './bloom-6b4-clp-german-oasst-v0.1',
    './ProjectIndus',
    './occiglot-7b-de-en-instruct',
    './occiglot-7b-eu5-instruct',
    './modello-italia-9b',
    './OpenHathi-7B-Hi-v0.1-Base',
    './falcon-7b-instruct',
    './Pythia-Chat-Base-7B'
]

def main():
    #some setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    selfcheck_ngram = SelfCheckNgram(n=1) # n=1 means Unigram, n=2 means Bigram, etc.
    nlp = spacy.load("en_core_web_sm")

    #some input
    file_path = './new_input' 

    files = list_files_in_directory(file_path)
    info = []
    for file in files:
        str_lst = file.split('_')
        tag = str_lst[0]
        model_names = str_lst[1:]
        index = model_names[-1].rfind('.')
        model_names[-1] = model_names[-1][:index]
        info.append({'tag': tag, 'model_name': '_'.join(model_names)})
    print(f"info example: {info[0]}")
        
    input_file_name_lst = [] 
    for info_entity in info:
        tag = info_entity['tag'] 
        model_name = info_entity['model_name']
        input_file_name = "./new_input/" + tag + '_' + model_name +".jsonl"
        input_file_name_lst.append(input_file_name)
    print(f"input_file_name_lst: {input_file_name_lst}")
    
    #get files by input_file_name
    for input_file_name in input_file_name_lst:
        data = load_file_jsonl(input_file_name)
        print(f"data example: model_input:{data[0]['model_input']}, model_output_text:{data[0]['model_output_text']}, model_id {data[0]['model_id']}")

        model_name = data[0]['model_id']
        index = model_name.find('/')
        model_name = './' + model_name[index+1:]
        print(f"model_name to load {model_name}")

        #we get only some model ready now
        if not (model_name in ok):
            print(f'this model {model_name} is not ready now.')
            continue

        #tokenizer and model init
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        #torch.float16 here!
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(device)

        #define the ask
        def ask(message_to_ask, model):
            #question here
            message = message_to_ask

            config = dict(top_k=50, top_p=0.90, temperature=0.3)
            prompt = message + '\n'
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            # eos_token_id=tokenizer.encode('\n'),
            # pad_token_id=tokenizer.encode('\n')[0],
            **config,
            )

            response_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response_text = response_text.replace(prompt, "") # response repeats the input in the begining
            response_token_ids = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
            # response_embeddings = outputs.sequences[0].to("cpu").tolist()[len(inputs.input_ids[0]):]
            #response_tokens = tokenizer.convert_ids_to_tokens(response_token_ids)
            #response_logits = [l.to("cpu").tolist() for l in outputs.logits]

            return response_text

        #ask('Hello, how are you going?', model)
        #  
        #begin asking
        to_save = []
        for item in data:
            question = item['model_input']
            answer = item['model_output_text']
            sample_lst = []
            for i in range(3):
                sample = ask(question, model)
                sample_lst.append(sample)

            sample_withquestion = []
            for sample in sample_lst:
                sample_withquestion.append(question + '\n' + sample)
            passage = question + '\n' + answer
            #print(sample_withquestion)

            to_eval = answer.strip() 
            doc = nlp(to_eval)
            sentences = [sent.text.strip() for sent in doc.sents] # spacy sentence tokenization
            print(f"sentences: {sentences}, passage: {passage}, sample_lst: { sample_lst}")
            sent_scores_ngram = selfcheck_ngram.predict(
                sentences = sentences,   
                passage = passage,
                sampled_passages = sample_lst,
            )
            #print(sent_scores_ngram)
            lst_logprob = sent_scores_ngram['sent_level']['lst_neg_logprob']
            lst_logprob = np.abs([item for sublist in lst_logprob for item in sublist])

            doc = nlp(to_eval)
            sentences = [sent.text.strip() for sent in doc.sents] # spacy sentence tokenization
            token_lst = []
            for sent in doc.sents:
                for token in nlp(sent.text.strip()):
                    token_lst.append(token)
            print(f"token_lst: {token_lst}, lst_logprob: { lst_logprob}")
            df = pd.DataFrame({'tokens': token_lst, 'logprob': lst_logprob})
            to_drop = df[df['logprob'] == df['logprob'].max()]['tokens'].tolist()
            to_save.append(to_drop)

        #filtering
        filtered_list = [[word for word in item if not word.is_stop] for item in to_save ]
        print(f"not filtered: {to_save}, filtered:{filtered_list}")

        #output
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        token_lst_lst = []
        for save in filtered_list:
            token_lst = list(token.text for token in save)
            token_lst_lst.append(token_lst)
        print(token_lst_lst)

        try:
            model_mark = input_file_name.replace('/', '_')
            save_file_name = f'./output_tmp_2/output_{model_mark}_{timestamp}.json'
            with open(save_file_name, 'w') as f:
                json.dump(token_lst_lst, f, indent=4)
                print(f"saved file to {save_file_name}")
        except Exception as e:
            print("发生错误:", e)
        del model
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()