---
license: openrail
language:
- az
metrics:
- accuracy
datasets:
- tibb.az
tags:
- classification
- medical
---

# Azerbaijani Medical Forum Question Classification

With the rapid increase of the internet, patients are increasingly use it for health information and support. However, given the large number of queries, and limited number of experts as well as not knowing which doctor to tell your complaint to, a significant fraction of the questions remains unanswered. Also, when patients apply online to the hospital, automatic direction to the appropriate doctor according to their disease is very important.

Automatic question classifiers can overcome this issue by directing questions to specific experts according to their topic preferences to get quick and better responses.
In this project, I aim to classify Azerbaijani health forum questions with BERT multilingual base model (uncased). BERT is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion.

For medical question classification, it requires high-quality datasets to train a deep-learning approach in a supervised way. Currently, there is no public dataset for Azerbaijani medical classification, and the datasets of other fields are not applicable to the medical QA system. To solve this problem, I scraped a m.tibb.az website using Python where 27k questions in 19 medical branch have been asked by users and answered by medical experts. 
I will also provide dataset which can be used in Azerbaijani medical QA and related fields. 

# How to use

Here is how to use this model.

__Firstly, you need to build a dictionary with medical branch names and their numbers, because target is encoded and model output will be a number.__

```python
branch_dict = {0: 'Endoskopist', 1: 'Nevropatoloq',2: 'Dermato veneroloq',3: 'Qastroenteroloq',
 4: 'Psixoloq', 5: 'Pediatr', 6: 'Proktoloq', 7: 'Endokrinoloq',
 8: 'Psixoterapevt', 9: 'Allerqoloq', 10: 'Oftalmoloq', 11: 'Kardioloq', 12: 'Uroloq',
 13: 'Plastik cərrah', 14: 'Cərrah-proktoloq', 15: 'Ümumi cərrah',
 16: 'Hepatoloq', 17: 'LOR həkimi', 18: 'Ginekoloq'}
```

__Secondly, we will use a simple Python function in order to convert model result to branch name.__

```python
def result_helper_funct(model_result):
    
    result = model_result[0][0]
    if result in branch_dict.keys(): 
        return branch_dict[result]
```

__Then, we need to install simpletransformers library__

```python
!pip install simpletransformers
```
__After succesfully installing, use pre-trained model.__

```python
from simpletransformers.classification import ClassificationModel
model = ClassificationModel("bert", "nijatzeynalov/azerbaijani-medical-question-classification", use_cuda=False)
```

__At the next step, we just write down the text we want to classify and use our helper function.__

```python
sample_text = 'salam menim qulagimda agri var'
result = model.predict([sample_text])

result_helper_funct(result)
```
__Code result:__

```python
'LOR həkimi'
```
__Let's try another example.__

```python
sample_text = 'üzümdə səpgi var'
result = model.predict([sample_text])

result_helper_funct(result)
```
__Code result:__

```python
'Allerqoloq'
```
