# HarmoniCA - Harmonizing Clinical Assessments

## Background

Diversity in the design of clinical assessment instruments creates fundamental incompatibilities when attempting to use them in a retrospective collaborative research setting (retrospective multi-site consortia, machine learning analyses, federated learning settings etc.). Previous harmonization approaches for questionnaire data have primarily relied on psychometric linking methods such as Item Response Theory (IRT) or Principal Component Analysis (PCA). While valuable, these methods require overlapping response data between questionnaires, which is often unavailable in retrospective multi-site analyses.

## Functionality of the tool

This tool offers mapping of individual questionnaire items from multiple instruments to pre-defined symptom dimensions. Dimension scores are subsequently transformed to allow comparability across different clinical instruments.

![img](docs/image.png)

## How to use


**Step 1:** Clone this repo: 

`git clone git@github.com:julia-pfarr/HarmoniCA.git`

**Step 2:** Install requirements:

`python -m venv /your-path/harmonica`
`pip install -r requirements.txt` 

**Step 3:** Run harmonization:

``` 
python harmonize.py -i <items.csv> 
``` 

Your `items.csv` needs to look like this:
```
construct,questionnaire,item_id,item_text
depression,CES-D,CES-D_01,I was bothered by things that usually don’t bother me.
depression,CES-D,CES-D_02,I did not feel like eating; my appetite was poor.
anxiety,DASS,DASS_02,I was aware of dryness of my mouth.,
anxiety,DASS,DASS_04,"I experienced breathing difficulty",
...
``` 

Models for each construct are pulled from [Huggingface](https://hf.co/collections/julia-pfarr/harmonica) during the harmonization process, so make sure to have an internet connection and enough local space (~1.5GB per model/construct).

**Step 4:** Open a PR to contribute new harmonized questionnaires

The `harmonized_inventory.csv` get's updated automatically. We appreciate a Pull Request on this repo with your updated `harmonized_inventory.csv` so that we can have an ever growing inventory! :-) 

You can try everything first with the `test-items.csv` from this repo!

## The research behind this tool

Symptom dimensions were chosen based on the convergence of evidence across original scale publications, validation studies, expert recommendations, diagnostic manuals, and neuroimaging applications, along with practical considerations regarding dimension homogeneity and sample characteristics (see our [OSF project](https://osf.io/caxzb/overview) for the full literature review and consensus pipeline).

Clinicians and researchers assigned items to dimensions through a structured survey. Through semantic similarity analysis using different embedding models and computation of embeddings for both questionnaire items and dimension descriptions, the best performing embedding model for each construct was chosen (maximum cosine similarity between item and to dimension description embedding). 

The best performing model for each construct was fine-tuned with the probability distribution of the expert mappings using contrastive learning.

## Questionnaires mapped by experts

| Abbreviation | Full Name | Construct |
|---|---|---|
| BDI | Beck's Depression Inventory I | Depression |
| HDRS/HAM-D | Hamilton Depression Rating Scale | Depression |
| GDS | Geriatric Depression Scale | Depression |
| MADRS | Montgomery-Åsberg Depression Rating Scale | Depression |
| PHQ-9 | Patient Health Questionnaire-9 | Depression |
| SDS | Zung Self-Rating Depression Scale | Depression |
| DSI | Depressive Symptom Inventory | Depression |
| MFQ | Mood and Feelings Questionnaire | Depression |
| BAI | Beck's Anxiety Inventory | Anxiety |
| HARS/HAM-A | Hamilton Anxiety Rating Scale | Anxiety |
| STAI | State-Trait Anxiety Inventory | Anxiety |
| PAS | Parkinson Anxiety Scale | Anxiety |
| GAD-7 | Generalized Anxiety Disorder Scale | Anxiety |
| PSWQ | Penn State Worry Questionnaire | Anxiety |
| SMGAD | Severity Measure for Generalized Anxiety Disorder | Anxiety |
| ASensI | Anxiety Sensitivity Index | Anxiety |
| AES | Apathy Evaluation Scale | Apathy |
| SAS | Starkstein's Apathy Scale | Apathy |
| AMI | Apathy and Motivation Index | Apathy |
| LARS | Lille Apathy Rating Scale | Apathy|
| DAS | Dimensional Apathy Scale | Apathy |
| SHAPS | Snaith-Hamilton Pleasure Scales | Apathy |
| TEPS | Temporal Experience of Pleasure Scale | Apathy |
| QUIP-C | Questionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease | Impulse Control Disorders |
| QUIP-RS | Quetionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease - Rating Scale | Impulse Control Disorders | 
| ESS | Epworth Sleepiness Scale | Sleep |
| PDSS | Parkinson's Disease Sleep Scale | Sleep |
| PSQI | Pittsburgh Sleep Quality Index | Sleep |
| RBDSQ | REM Sleep Behavior Disorder Screening Questionnaire | Sleep|
| ISI | Insomnia Severity Index | Sleep |
| SCOPA-sleep | Scales for Outcomes in Parkinson's Disease - sleep | Sleep |
| AIS | Athens Insomnia Scale | Sleep |
| MSQ | Mayo Sleep Questionnaire | Sleep |
| SDQ | Sleep Disorders Questionnaire | Sleep |
| PPRS | Parkinson's Psychosis Rating Scale | Psychosis |
| SCOPA-PC | Scales for Outcomes in Parkinson's Disease - Psychiatric Complications | Psychosis |
| SAPS-PD | Scale for the Assesment of Positive Symptoms - Parkinson's Disease | Psychosis |
| UMD-PDHQ | University of Miami Parkinson's Disease Hallucinations Questionnaire | Psychosis |
| CAPE-P15 | Community Assessment of Psychic Experiences, Positive scale, 15-item version | Psychosis |
| PQ-B | Prodromal Questionnaire, Brief version | Psychosis |
| HADS | Hopsital Anxiety and Depression Scale | Multi-category |
| SCL-90 | Symptom Checklist-90 | Multi-category |
| ASBPD | Ardouin Scale of Behavior in Parkinson's Disease | Multi-category |
| NPI | Neuropsychiatric Inventory | Multi-category |
| MDS-UPDRS I | Unified Parkinson's Disease Rating Scale - Part I | Multi-category |
| NMSS | Non-Motor Symptoms Scale for Parkinson's Disease | Multi-category | 
| NMSQuest | Non-Motor Symptoms Questionnaire | Multi-category |

## Results

Cross validation results:

| Construct| Folds | Mean accuracy| SD |
|-----------|----------|-----------|----------|
| Depression | 14 | 86.9% | ±11.9% |
| Anxiety | 13 | 93.7% | ±10.7% |
| Sleep | 17 | 93.4% | ±12.1% |
| Apathy | 11 | 77.9% | ±25.0% |

## Final models

All fine-tuned models can be found on [Huggingface](https://hf.co/collections/julia-pfarr/harmonica)
