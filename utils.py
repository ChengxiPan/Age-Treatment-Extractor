import re
from openai import OpenAI
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
import config
import nltk
nltk.download('punkt')

checkpoint_path = config.CHECKPOINT_PATH
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
id2label = config.ID_TO_LABEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


'''
function: age_extractor
description: Extracts the age from a string use Regular Expressions
input: 
    text (str) - the text to extract the age from
output: 
    age (float) - the age extracted from the text
example:
    xxx-year-old -> xxx
    xxx-month-old -> xxx/12
    xxx years old -> xxx
    xxx-year old -> xxx
'''
def re_age_extractor(text) -> float:
    year_match = re.search(r'(\d{1,3})-year-old', text)
    month_match = re.search(r'(\d{1,3})-month-old', text)
    years_match = re.search(r'(\d{1,3})\s+years?\s+old', text)
    year_old_match = re.search(r'(\d{1,3})-year\s+old', text)
    
    if year_match:
        return int(year_match.group(1))  
    elif month_match:
        return int(month_match.group(1)) / 12 
    elif years_match:
        return int(years_match.group(1))
    elif year_old_match:
        return int(year_old_match.group(1))
    return None  # Return None if no match is found or text is not a string


'''
function: gpt_treatment_extractor
description: Extracts treatment from a medical transcription using GPT
input: 
    text (str) - the text to extract the treatment from
output: 
    treatment (str) - the treatment extracted from the text
example:
    Transcription: "The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures"
    Treatment Type: "(intensity-modulated) radiation therapy"
'''
client = OpenAI()
# Define the function
def gpt_treatment_extractor(transcription, model="gpt-3.5-turbo") -> str:
    messages = [
        {"role": "system", "content": "You are a medical professional. Please extract the primary treatment type from the medical transcription."},
        {"role": "user", "content": f"Extract the primary treatment type from the medical transcription. If unspecified, return 'no specific':\n\n"
            f"Transcription: \"{transcription}\"\n\n"
            f"Treatment Type (e.g., 'radiation therapy'):"}
    ]

    response = client.chat.completions.create(
        model=model,  
        messages=messages,
        max_tokens=50,
        temperature=0.0
    )
    
    treatment = response.choices[0].message.content
    print(treatment)
    return treatment


'''
function: re_treatment_extractor
description: Extracts treatment from a medical transcription using Regular Expressions
input: 
    text (str) - the text to extract the treatment from
output: 
    treatment (list) - the treatment extracted from the text
example:
    Transcription: "The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures"
    Treatment Type: "(intensity-modulated) radiation therapy"
PS: TIME IS LIMITED, SO THIS FUNCTION NEED TO BE IMPROVED
'''
import nltk
nltk.download('punkt')

def re_treatment_extractor(text):
    # 处理空值或非字符串类型
    if not isinstance(text, str):
        return []  # 如果不是字符串，返回空列表，表示没有治疗信息

    sentences = nltk.sent_tokenize(text)  # 句子分割

    # 定义正则表达式模式，仅提取治疗名称
    treatment_patterns = [
        r'receive\s+([\w\s\-]+therapy)',            # 捕捉 "receive [treatment] therapy"
        r'undergo\s+([\w\s\-]+therapy)',            # 捕捉 "undergo [treatment] therapy"
        r'treated with\s+([\w\s\-]+therapy)',       # 捕捉 "treated with [treatment] therapy"
        r'administered\s+([\w\s\-]+therapy)',       # 捕捉 "administered [treatment] therapy"
        r'scheduled for\s+([\w\s\-]+therapy)',      # 捕捉 "scheduled for [treatment] therapy"
        r'planned for\s+([\w\s\-]+therapy)',        # 捕捉 "planned for [treatment] therapy"
        r'([\w\s\-]+therapy)',                      # 捕捉 "[treatment] therapy"
        r'([\w\s\-]+surgery)',                      # 捕捉 "[treatment] surgery"
        r'([\w\s\-]+chemotherapy)',                 # 捕捉 "[treatment] chemotherapy"
        r'([\w\s\-]+immunotherapy)',                # 捕捉 "[treatment] immunotherapy"
        r'([\w\s\-]+medication)',                   # 捕捉 "[treatment] medication"
        r'([\w\s\-]+targeted therapy)'              # 捕捉 "[treatment] targeted therapy"
    ]

    treatments = []

    for sentence in sentences:
        for pattern in treatment_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                treatments.append(match.group(1).strip())  # 只提取治疗名称

    return treatments



'''
function: NER
description: Extracts treatment from a medical transcription using Named Entity Recognition
input: 
    text (str) - the text to extract the treatment from
output: 
    treatment (list) - the treatment extracted from the text
example:
    Transcription: "The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures"
    Treatment Type: "(intensity-modulated) radiation therapy"
PS: TIME IS LIMITED, SO THIS FUNCTION NEED TO BE IMPROVED
'''
def NER(text):
    if not isinstance(text, str):
        return []
    # tokenization
    inputs = tokenizer(text.split(), return_tensors="pt", truncation=True, is_split_into_words=True)
    # to_GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = outputs.logits.argmax(dim=2)
    predicted_labels = [id2label[prediction.item()] for prediction in predictions[0].cpu().numpy()]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    
    # filter B-TREATMENT
    result = [(token, label) for token, label in zip(tokens, predicted_labels) if label == 'B-TREATMENT']
    
    return result




'''
function: Bert_NER
description: Extracts named entities from a text using a pre-trained BERT model
!!!!!!!!!NOT APPLICABLE!!!!!!!!!!!
'''
def Bert_NER(example):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained model and tokenizer, and move the model to the device
    model_name = "dslim/bert-base-NER"
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the NER pipeline and move it to the correct device
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Perform NER on the input text
    ner_results = nlp(example)

    return ner_results
