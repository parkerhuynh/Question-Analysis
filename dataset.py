import json
from torch.utils.data import Dataset, DataLoader
import torch
import os
import pickle
import en_core_web_lg, random, re, json
import numpy as np
import random
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}
def LSTM_tokenize(stat_ques_list, args):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }
    
    spacy_tool = None
    pretrained_emb = []
    if args.use_glove:
        spacy_tool = en_core_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if args.use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    return token_to_ix, pretrained_emb

def rnn_proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break
    return ques_ix
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', #'-',
                '>', '<', '@', '`', ',', '?', '!']
def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
            or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText
comma_strip = re.compile("(\d)(\,)(\d)")
manual_map = { 'none': '0',
                'zero': '0',
                'one': '1',
                'two': '2',
                'three': '3',
                'four': '4',
                'five': '5',
                'six': '6',
                'seven': '7',
                'eight': '8',
                'nine': '9',
                'ten': '10'}
articles = ['a', 'an', 'the']

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def prep_ans(answer):
    
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

class QuestionDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ans_type_to_idx = {
            'yes/no': 0,
            'color': 1,
            'object': 2,
            'number': 3,
            'location': 4,
            'action': 5,
            'other': 6,
            'human': 7,
            'sport': 8
            }
        self.idx_to_ans_type = {
            0: 'yes/no',
            1: 'color',
            2: 'object',
            3: 'number',
            4: 'location',
            5: 'action',
            6: 'other',
            7: 'human',
            8: 'sport'
            }
        self.question_type_dict = self.load_question_type()
        self.questions = self.load_questions()
        
        self.annotations = self.load_annotations()
        self.output_dim = len(self.ans_type_to_idx.keys())
        random.shuffle(self.annotations)

    def __len__(self):
        if self.args.debug:
            return 500
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        que = self.questions[ann["id"]]
        question_id = ann["id"]
        question = que["question"]
        
        label = que["question_type"]
        
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.args.max_ques_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'question_id': question_id,
            'question_text': question,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def load_questions(self):
        if self.split == "train":
            question_path = self.args.train_question
        else:
            question_path = self.args.val_question
        with open(question_path, 'r') as file:
            questions = json.load(file)["questions"]
        processed_questions = {}
        for question in questions:
            question_type_str = self.question_type_dict[question["question"]]
            question_type_idx = self.ans_type_to_idx[question_type_str]
            question["question_type"] = question_type_idx
            processed_questions[question["id"]] = question
        return processed_questions
    
    def load_annotations(self):
        if self.split == "train":
            annotation_path = self.args.train_annotation
        else:
            annotation_path = self.args.val_annotation
        with open(annotation_path, 'r') as file:
            annotation_list = json.load(file)["annotations"]
        processed_annotations = []
        for ann in annotation_list:
            ans_count = 0
            for judge in ann["judgements"].values():
                if judge["answer"] == 1:
                    ans_count += 1
            if ans_count >= 2 or ann["overall_scores"]["question"] < 0.5:
                processed_annotations.append(ann)
        return processed_annotations
    
    def load_question_type(self):
        question_type_dict = {}
        with open('/home/ndhuynh/github/Question-Analysis/train_question_type_gpt_v6.json', 'r') as file:
            for line in file:
                question_object = json.loads(line)
                question_str = question_object["question"]
                question_type = question_object["question_type"]
                question_type = question_type_processing(question_type)
                if question_str not in question_type_dict.keys():
                    question_type_dict[question_str] = question_type
        file.close()
        
        with open('/home/ndhuynh/github/Question-Analysis/val_question_type_gpt_v6.json', 'r') as file:
            for line in file:
                question_object = json.loads(line)
                question_str = question_object["question"]
                if question_str not in question_type_dict:
                    question_type = question_object["question_type"]
                    question_type = question_type_processing(question_type)
                    question_type_dict[question_str] = question_type
        file.close()
        return question_type_dict
        
def question_type_processing(question_type):
    question_type = question_type.strip()
    question_type = question_type.split(':')[-1]
    question_type = question_type.lower()
    question_type = re.sub(r'\([^)]*\)', '', question_type)
    question_type = question_type.replace("'", "")
    question_type = question_type.replace(".", "")
    question_type = question_type.replace("`", "")
    question_type = question_type.replace(";", ",")
    question_type = question_type.replace('"', "")
    question_type = question_type.replace("question type: ", "")
    question_type = question_type.replace("question: ", "")
    question_type = question_type.replace("’", "")
    question_type = question_type.replace("[", "")
    question_type = question_type.replace("]", "")
    question_type = question_type.replace("-", "")
    question_type = question_type.replace(" or ", ", ")
    question_type = question_type.replace("+", ", ")
    question_type = question_type.replace(" , ", ", ")
    question_type = re.sub('\s+', ' ', question_type).strip()
    # 
    for word in ["clothing", "object", "material", "shape", "food", "transportation", "pattern", "letter", "drink"]:
        if word in question_type:
            return "object"
    for word in ["age", "animal", "body parts", "gender", "body part", "person", "emotion"]:
        if word in question_type:
            return "human"
    for word in ["weather", "season", "time", "unknown", "any", "not", "event", "geometry", "music", "description", "flavor", "sound", "taste"]:        
        if word in question_type:
            return "other"
    for word in ["activity", "action"]:        
        if word in question_type:
            return "action"
    for word in ["direction"]:        
        if word in question_type:
            return "location"
    for word in ["yes/no"]:        
        if word in question_type:
            return "yes/no"
    for word in ["number"]:        
        if word in question_type:
            return "number"
    question_type = question_type.split(",")
    question_type =  [item.strip() for item in question_type][0]
    return question_type