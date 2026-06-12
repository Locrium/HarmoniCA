# HarmoniCA - Harmonizing Clinical Assessments

## Background

Diversity in the design of clinical assessment instruments creates fundamental incompatibilities when attempting to use them in a retrospective collaborative research setting (retrospective multi-site consortia, machine learning analyses, federated learning settings etc.). Previous harmonization approaches for questionnaire data have primarily relied on psychometric linking methods such as Item Response Theory (IRT) or Principal Component Analysis (PCA). While valuable, these methods require overlapping response data between questionnaires, which is often unavailable in retrospective multi-site analyses.

## Functionality of the tool

This tool offers mapping of individual questionnaire items from multiple instruments to pre-defined symptom dimensions. Dimension scores are subsequently transformed to allow comparability across different clinical instruments.

![img](docs/image.png)

## Research

Symptom dimensions were chosen based on the convergence of evidence across original scale publications, validation studies, expert recommendations, diagnostic manuals, and neuroimaging applications, along with practical considerations regarding dimension homogeneity and sample characteristics. Clinicians and researchers assigned items to dimensions through a structured survey. Through semantic similarity analysis using different embedding models and computation of embeddings for both questionnaire items and dimension descriptions, the best performing embedding model for each construct was chosen based on maximum cosine similarity between item and to dimension description embeddings (mirroring the expert task). The best performing model for each construct was fine-tuned with the probability distribution of the expert mappings using contrastive learning.

## Available Questionnaires

The table below lists all 48 questionnaires currently supported by the tool, organized by the symptom construct each one maps to: Depression (9), Anxiety (9), Apathy (7), Psychosis (6), Impulse Control Disorders (2), Sleep (9), and Multi-category (6) for questionnaires spanning several constructs.

| Abbreviation | Full Name | Construct |
|---|---|---|
| BDI | Beck Depression Inventory | Depression |
| HDRS/HAM-D | Hamilton Depression Rating Scale | Depression |
| HADS-D | Hospital Anxiety and Depression Scale – Depression subscale | Depression |
| GDS | Geriatric Depression Scale | Depression |
| MADRS | Montgomery-Åsberg Depression Rating Scale | Depression |
| PHQ-9 | Patient Health Questionnaire-9 | Depression |
| SDS | Zung Self-Rating Depression Scale | Depression |
| DSI | Depressive Symptom Inventory | Depression |
| MFQ | Mood and Feelings Questionnaire | Depression |
| BAI | Beck Anxiety Inventory | Anxiety |
| HARS/HAM-A | Hamilton Anxiety Rating Scale | Anxiety |
| HADS-A | Hospital Anxiety and Depression Scale – Anxiety subscale | Anxiety |
| STAI | State-Trait Anxiety Inventory | Anxiety |
| PAS | Parkinson Anxiety Scale | Anxiety |
| GAD-7 | Generalized Anxiety Disorder Scale | Anxiety |
| PSWQ | Penn State Worry Questionnaire | Anxiety |
| SMGAD | Severity Measure for Generalized Anxiety Disorder | Anxiety |
| ASensI | Anxiety Sensitivity Index | Anxiety |
| AES | Apathy Evaluation Scale | Apathy |
| SAS | Starkstein's Apathy Scale | Apathy |
| AMI | Apathy and Motivation Index | Apathy |
| LARS | Lille Apathy Rating Scale | Apathy |
| DAS | Dimensional Apathy Scale | Apathy |
| SHAPS | Snaith-Hamilton Pleasure Scales | Apathy |
| TEPS | Temporal Experience of Pleasure Scale | Apathy |
| PPRS | Parkinson's Psychosis Rating Scale | Psychosis |
| SCOPA-PC | Scales for Outcomes in Parkinson's Disease – Psychiatric Complications | Psychosis |
| SAPS-PD | Scale for the Assessment of Positive Symptoms – Parkinson's Disease | Psychosis |
| UM-PDHQ | University of Miami Parkinson's Disease Hallucinations Questionnaire | Psychosis |
| CAPE-P15 | Community Assessment of Psychic Experiences, Positive scale, 15-item version | Psychosis |
| PQ-B | Prodromal Questionnaire, Brief version | Psychosis |
| QUIP-RS | Questionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease, Rating Scale | Impulse Control Disorders |
| QUIP-C | Questionnaire for Impulsive-Compulsive Disorders in Parkinson's Disease | Impulse Control Disorders |
| ESS | Epworth Sleepiness Scale | Sleep |
| PDSS | Parkinson's Disease Sleep Scale | Sleep |
| PSQI | Pittsburgh Sleep Quality Index | Sleep |
| RBDSQ | REM Sleep Behavior Disorder Screening Questionnaire | Sleep |
| ISI | Insomnia Severity Index | Sleep |
| SCOPA-SLEEP | Scales for Outcomes in Parkinson's Disease – Sleep | Sleep |
| AIS | Athens Insomnia Scale | Sleep |
| MSQ | Mayo Sleep Questionnaire | Sleep |
| SDQ | Sleep Disorders Questionnaire | Sleep |
| NPI | Neuropsychiatric Inventory | Multi-category |
| MDS-UPDRS Part I | Unified Parkinson's Disease Rating Scale, Part I (non-motor) | Multi-category |
| NMSS | Non-Motor Symptoms Scale for Parkinson's Disease | Multi-category |
| NMSQuest | Non-Motor Symptoms Questionnaire | Multi-category |
| SCL-90 | Symptom Checklist-90 | Multi-category |
| ASBPD | Ardouin Scale of Behavior in Parkinson's Disease | Multi-category |

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
