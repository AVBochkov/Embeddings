import random
import time
from datetime import timedelta
from datetime import datetime

import os
import numpy as np
import pandas as pd

import sacrebleu

import torch
import torch.nn as nn

from datasets import load_dataset

def load_opus_samples(source_lang, target_lang, _lang_lang='en-ru',n_questions=100):
    ds_opus_dataset_id = 'Helsinki-NLP/opus-100'
    ds_opus = load_dataset(ds_opus_dataset_id, _lang_lang, split='validation')
    samples = []
    for entry in ds_opus:
        samples.append({
            'question': entry['translation'][source_lang],
            'answer':entry['translation'][target_lang],
        })
    random.shuffle(samples)
    return samples[:n_questions]

def load_squad_v2_samples(n_questions=100):
    ds_squad2_dataset_id = 'squad_v2'
    ds_squad2 = load_dataset(ds_squad2_dataset_id, split='validation')
    samples = []
    for entry in ds_squad2:
        answer = ""
        if 'answers' in entry and 'text' in entry['answers']:
            if entry['answers']['text']:
                answer = entry['answers']['text'][0]
            else:
                answer = "no answer"

        if answer != "no answer" and answer != "":
            if len(answer) <= 8:
                samples.append({
                    'question': entry['question'],
                    'passage': entry['context'],
                    'answer':answer,
                })
    random.shuffle(samples)
    return samples[:n_questions]

def load_mmlu_samples(subjects, n_questions=100):
    samples = []
    
    if n_questions != -1:
        max_n = n_questions
    else:
        max_n = 0
        
    for subject in subjects:
        ds = load_dataset("lukaemon/mmlu", subject, split="test")
        
        if n_questions == -1:
            max_n = max_n + len(ds)

        for entry in ds:
            samples.append({
                'question': entry['input'],
                'choices': [entry['A'], entry['B'], entry['C'], entry['D']],
                'answer': entry['target'],  # Уже A/B/C/D
            })
            
    random.shuffle(samples)
    return samples[:max_n]

def load_boolq_samples(n_questions=100):
    boolq_dataset_id = 'boolq'
    ds_boolq = load_dataset(boolq_dataset_id, split='validation')
    samples = []
    for entry in ds_boolq:
        samples.append({
            'question': entry['question'],
            'passage': entry['passage'],
            'answer': entry['answer'],  # True/False (bool)
        })
    random.shuffle(samples)
    return samples[:n_questions]

def load_commonsenseqa_samples(n_questions=100):
    commonsense_dataset_id = 'commonsense_qa'
    ds_commonsense = load_dataset(commonsense_dataset_id, split='validation')
    samples = []
    for entry in ds_commonsense:
        choices = entry['choices']
        samples.append({
            'question': entry['question'],
            'choices': [choices['text'][i] for i in range(len(choices['text']))],
            'labels': choices['label'],  # ['A', 'B', 'C', etc.]
            'answer': entry['answerKey'],  # например 'B'
        })
    random.shuffle(samples)
    return samples[:n_questions]

def load_ai2arc_e_samples(n_questions=100):
    ai2_arc_dataset_id = 'ai2_arc'
    ds_ai2arc = load_dataset(ai2_arc_dataset_id, 'ARC-Easy', split='validation')
    samples = []
    for entry in ds_ai2arc:
        choices = entry['choices']
        samples.append({
            'question': entry['question'],
            'choices': [choices['text'][i] for i in range(len(choices['text']))],
            'labels': choices['label'],
            'answer': entry['answerKey'],
        })
    random.shuffle(samples)
    return samples[:n_questions]
    
def load_ai2arc_c_samples(n_questions=100):
    ai2_arc_dataset_id = 'ai2_arc'
    ds_ai2arc = load_dataset(ai2_arc_dataset_id, 'ARC-Challenge', split='validation')
    samples = []
    for entry in ds_ai2arc:
        choices = entry['choices']
        samples.append({
            'question': entry['question'],
            'choices': [choices['text'][i] for i in range(len(choices['text']))],
            'labels': choices['label'],
            'answer': entry['answerKey'],
        })

    random.shuffle(samples)
    return samples[:n_questions]

def format_prompt_opus(entry, source_lang, target_lang):
    question = entry['question']
    prompt = f"Translate from {source_lang.replace('en','English').replace('ru','Russian').replace('zh','Chinese')} to {target_lang.replace('en','English').replace('ru','Russian').replace('zh','Chinese')}:\n{question}\nAnswer:"
    correct_answer = entry['answer']
    return prompt, correct_answer

def evaluate_model_on_opus(p_model, p_tokenizer, p_samples, source_lang, target_lang, max_new_tokens=100, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0.0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_opus(entry, source_lang, target_lang)

        # Tokenize and infer
        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = p_model.generate(input_ids, min(len(correct_answer), max_new_tokens))[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()
        idx = gen_letter.find('</S>')
        gen_letter = gen_letter[:idx] if idx != -1 else gen_letter

        #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer.upper())

        references = [correct_answer.upper()]
        
        predictions = [gen_letter.upper()]
        
        bleu = sacrebleu.corpus_bleu(predictions, [references])

        correct += bleu.score

    acc = correct / total
    return acc

def format_prompt_squad_v2(entry):
    context = entry['passage']
    question = entry['question']
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    correct_answer = entry['answer']
    return prompt, correct_answer

def evaluate_model_on_squad_v2(p_model, p_tokenizer, p_samples, max_new_tokens=100, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_squad_v2(entry)

        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = p_model.generate(input_ids, min(len(correct_answer), max_new_tokens))[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()

        if gen_letter.upper() == correct_answer.upper() or correct_answer.upper() in gen_letter.upper():
            correct += 1
            #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer.upper())

    acc = correct / total
    return acc

def format_prompt_mmlu(entry):
    letters = ['A', 'B', 'C', 'D']
    options = " ".join([f"{letter}) {text}" for letter, text in zip(letters, entry['choices'])])
    prompt = f"Question: {entry['question']}\nOptions: {options}\nAnswer:"
    correct_answer = entry['answer']
    return prompt, correct_answer
    
def evaluate_model_on_mmlu(p_model, p_tokenizer, p_samples, max_new_tokens=1, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_mmlu(entry)

        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)#.unsqueeze(0)
        output_ids = p_model.generate(input_ids, max_new_tokens)[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()[:1]

        #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer)

        if gen_letter.upper() == correct_answer.upper():
            correct += 1

    acc = correct / total
    return acc

def format_prompt_boolq(entry):
    context = entry['passage']
    question = entry['question']
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer (True/False):"
    correct_answer = 'T' if entry['answer'] else 'F'
    return prompt, correct_answer

def evaluate_model_on_boolq(p_model, p_tokenizer, p_samples, max_new_tokens=1, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_boolq(entry)

        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = p_model.generate(input_ids, max_new_tokens)[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()[:1]

        #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer)

        if gen_letter.upper() == correct_answer.upper():
            correct += 1

    acc = correct / total
    return acc

def format_prompt_ai2arc(entry):
    options = "\n".join([f"{label}) {text}" for label, text in zip(entry['labels'], entry['choices'])])
    prompt = f"Question: {entry['question']}\nOptions:\n{options}\nAnswer:"
    correct_answer = entry['answer']
    return prompt, correct_answer

def evaluate_model_on_ai2_arc(p_model, p_tokenizer, p_samples, max_new_tokens=1, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_ai2arc(entry)

        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = p_model.generate(input_ids, max_new_tokens)[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()[:1]
        
        #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer)
        
        if gen_letter.upper() == correct_answer.upper():
            correct += 1

    acc = correct / total
    return acc

def format_prompt_commonsenseqa(entry):
    options = "\n".join([f"{label}) {text}" for label, text in zip(entry['labels'], entry['choices'])])
    prompt = f"Question: {entry['question']}\nOptions:\n{options}\nAnswer:"
    correct_answer = entry['answer']
    return prompt, correct_answer

def evaluate_model_on_commonsense_qa(p_model, p_tokenizer, p_samples, max_new_tokens=1, device='cuda'):
    p_model.eval()

    samples = p_samples

    correct = 0
    total = len(samples)

    for entry in samples:
        prompt_text, correct_answer = format_prompt_commonsenseqa(entry)

        input_ids = p_tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = p_model.generate(input_ids, max_new_tokens)[0].tolist()
        output_text = p_tokenizer.decode(output_ids).replace('<s>','')

        gen_letter = output_text[len(prompt_text):].strip().upper()[:1]

        #print(prompt_text, 'gen_letter: ', gen_letter, ' correct_answer: ', correct_answer)

        if gen_letter.upper() == correct_answer.upper():
            correct += 1

    acc = correct / total
    return acc

def save_list_to_file(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(f"{item}\n" for item in data_list)

def load_list_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
        
def evaluate_with_stats(model, tokenizer, benchmark, p_samples, subjects, device='cuda', n_runs=100, p_source_lang='', p_target_lang=''):
    accuracies = []

    if benchmark == 'MMLU' and len(subjects) == 1:
        _samples = load_mmlu_samples(subjects, n_questions=-1)
    else:
        _samples = p_samples
    
    for _ in range(n_runs):
        if benchmark == 'MMLU':
            accuracy = evaluate_model_on_mmlu(
                p_model=model, 
                p_tokenizer=tokenizer, 
                p_samples=_samples, 
                max_new_tokens=1, 
                device=device
            )

        if benchmark == 'ARC-e' or benchmark == 'ARC-c':
            accuracy = evaluate_model_on_ai2_arc(
                p_model=model, 
                p_tokenizer=tokenizer, 
                p_samples=_samples, 
                max_new_tokens=2, 
                device=device
            )
            
        if benchmark == 'C-SENSE':
            accuracy = evaluate_model_on_commonsense_qa(
                p_model=model, 
                p_tokenizer=tokenizer, 
                p_samples=_samples, 
                max_new_tokens=2, 
                device=device
            )
            
        if benchmark == 'SQUAD':
            accuracy = evaluate_model_on_squad_v2(
                p_model=model, 
                p_tokenizer=tokenizer, 
                p_samples=_samples, 
                max_new_tokens=100, 
                device=device
            )
            
        if benchmark == 'BLEU':
            accuracy = evaluate_model_on_opus(
                p_model=model, 
                p_tokenizer=tokenizer, 
                p_samples=_samples, 
                source_lang=p_source_lang, 
                target_lang=p_target_lang, 
                max_new_tokens=100, 
                device=device
            )

        if benchmark == 'BLEU':
            accuracies.append(accuracy)
        else:
            accuracies.append(accuracy * 100)
    
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    ci95 = 1.96 * std / np.sqrt(n_runs)  
    
    return mean, std, ci95