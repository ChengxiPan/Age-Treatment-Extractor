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
  ⚠️ **Why Regular Expression? **: 
    <p></p>
    The format of age information is fixed. In this situation RE Method can take into account both speed and accuracy at the same time. GPT can guarantee the accuracy and perform even better in more complex situations, but it may be expensive and slow in processing.
</div>

Age format is fixed in the following 4 types:

- Xxx-year-old
- Xxx-month-old
- Xxx years old
- Xxx-year old

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



