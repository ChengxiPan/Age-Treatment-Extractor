# Medical-Transcriptions-Processing

### Task

Task1: Extract the age information, there might be variations, you need to cover these cases

Task2: Extract the treatment the patient received from the "transcription" column. For instance, this sentence "The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures." shows the patient get radiation therapy as treatment.

### Tutorial

For quick use:

* Firstly, export your gpt_api_key in command line.
* Secondly, run the following command

```bash
python3 --ageExtractor re_age_extractor 
		--treatmentExtractor gpt_treatment_extractor
		--gptModel gpt-3.5-turbo
```

Augments:

* ageExtractor: This project provides 1 function to extract age information from the original data, which is Regular Expression.

  ```bash
  python3 main.py --ageExtractor re_age_extractor
  ```



* treatmentExtractor: This project provides 3 functions to extract treatment information from the original data, GPT-API, Regular Expression and bio-BERT-NER. However, the last 2 methods don't perform well. GPT-Extractor is recommended.

  ```bash
  python3 main.py --treatmentExtractor gpt_treatment_extractor
  ```

  Once the gpt_treatment_extractor is chosen, `--gptModel`should be provided. You can choose whatever version of GPT-Model you want. OpenAI provides the following interfaces, but please carefully choose and count the price before you running your code because it may contribute to high cost.

  ![image-20240909151716086](https://raw.githubusercontent.com/Sweet196/Picgo-images/main/problems/202409091531020.png)

### Explanations

#### Task1：Extract Age Information

Example: 

```yaml
Input: A 23-year-old white female presents with complaint of allergies.
Output: 23
```

Method: Regular Expressions

<div style="border-left: 8px solid red; padding: 10px; background-color: #f8d7da;">
  ⚠️ Why Regular Expression?: 
    <p></p>
    The format of age information is fixed. In this situation RE Method can take into account both speed and accuracy at the same time. GPT can guarantee the accuracy and perform even better in more complex situations, but it may be expensive and slow in processing.
</div>


Age format is fixed in the following 4 types:

- Xxx-year-old
- Xxx-month-old
- Xxx years old
- Xxx-year old

```python
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
```



#### Task2: Extract Treatment Information

Example: 

```yaml
Input: The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures.
Output: Intensity-modulated radiation therapy
```

Methods: 

1. gpt_treatment_extractor
2. re_treatment_extractor
3. NER (Name Entity Recognition)

##### gpt_treatment_extractor

ChatGPT excels at handling and generating natural language. It can understand context, answer questions, engage in creative writing, translate text, and perform a variety of other tasks. Obviously it is easy to ChatGPT to complete the task.

```yaml
Prompt: Extract the primary treatment type from the medical transcription. If unspecified, return 'no specific'
Question: The patient will receive intensity-modulated radiation therapy in order to deliver high-dose treatment to sensitive structures.
Answer: Intensity-modulated radiation therapy
```

```python
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
```

##### Regular Expressions

It is noticed that the majority of treatments have a fixed suffix. In this situation Regular Expressions will work, but time is limited and I don't have enough time to develop a rounded rule. The following block only shows some examples to extract treatments with RE:

```python
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
def re_treatment_extractor(text):
    if not isinstance(text, str):
        return []

    sentences = nltk.sent_tokenize(text)

    treatment_patterns = [
        r'receive\s+([\w\s\-]+therapy)',            # "receive [treatment] therapy"
        r'undergo\s+([\w\s\-]+therapy)',            # "undergo [treatment] therapy"
        r'treated with\s+([\w\s\-]+therapy)',       # "treated with [treatment] therapy"
        r'administered\s+([\w\s\-]+therapy)',       # "administered [treatment] therapy"
        r'scheduled for\s+([\w\s\-]+therapy)',      # "scheduled for [treatment] therapy"
        r'planned for\s+([\w\s\-]+therapy)',        # "planned for [treatment] therapy"
        r'([\w\s\-]+therapy)',                      # "[treatment] therapy"
        r'([\w\s\-]+surgery)',                      # "[treatment] surgery"
        r'([\w\s\-]+chemotherapy)',                 # "[treatment] chemotherapy"
        r'([\w\s\-]+immunotherapy)',                # "[treatment] immunotherapy"
        r'([\w\s\-]+medication)',                   # "[treatment] medication"
        r'([\w\s\-]+targeted therapy)'              # "[treatment] targeted therapy"
    ]

    treatments = []

    for sentence in sentences:
        for pattern in treatment_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                treatments.append(match.group(1).strip())

    return treatments

```

#### bioBERT+NER( Name Entity Recognition)

BioBERT (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining) is a pre-trained language model designed specifically for biomedical text. It is based on the BERT architecture but has been further trained on a large corpus of biomedical literature. I decide to apply Name Entity Recognition task on the dataset and it is supposed to perform well.

In the part of generating bio_data, I use `transcription` column as input and the transcription extracted by `gpt-3.5-turbo` as label. Finally a bio_data.csv is generated:

```csv
token,label
[CLS],I-TREATMENT
subjective,I-TREATMENT
:,I-TREATMENT
",",I-TREATMENT
this,O
23,O
-,O
year,O
-,O
old,O
white,O
female,O
```

Tag "B" indicates that the character is the beginning of a word or can represent a single-character word. "I" indicates that the character is in the middle of a word. "O" indicates that the character is not part of any word. The "O" tag is not used in part-of-speech tagging tasks, but it is meaningful in named entity recognition tasks.

However it doesn't work, and I suppose it's because the sample size is too small.

```python
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
```

 ### Evaluation Matrix

| Task              | Method                  | Time Speed  | Time Cost             |
| ----------------- | ----------------------- | ----------- | --------------------- |
| Age Extract       | Regular Expressions     | 7868it/s    | 0.66s                 |
| Treatment Extract | gpt-3.5-turbo           | 1.87it/s    | 2670s                 |
|                   | **gpt-4o-mini**         | **2.5it/s** | (not test completely) |
|                   | gpt-4o                  | 2.27it/s    | (not test completely) |
|                   | Regular Expression      | 114it/s     | 44s                   |
|                   | Name Entity Recognition | 39.46it/s   | 126s                  |

When applying the Name Entity recognition method, I choose the whole dataset as the train_data and no data for evaluation. If the dataset is splited, more parameters can be recorded(e.g. F1 score).
