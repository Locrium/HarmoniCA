"""
HarmoniCA configuration: supported constructs, dimension descriptions,
and best-model selection per construct.
"""

CONSTRUCTS = ['depression', 'apathy', 'psychosis', 'anxiety', 'sleep', 'impulse_control']

# Dimension descriptions used for labelling and (optionally) base-model similarity
DIMENSION_DESCRIPTIONS = {
    'depression': {
        1: {
            'label': 'Mood & Affective Symptoms',
            'description': 'dysphoria, sadness, unhappiness, feeling low, feeling blue, hopelessness',
        },
        2: {
            'label': 'Cognitive & Self-Perception',
            'description': 'self-attitudes, predictions of the future, narratives of the past, guilt, worthlessness, failure, self-criticism, pessimism',
        },
        3: {
            'label': 'Somatic & Vegetative Symptoms',
            'description': 'physiological manifestations or consequences of depression, neurophysiological, autonomic, sleep problems, appetite changes, fatigue, weight changes, physical symptoms',
        },
        4: {
            'label': 'Activity & Interest Deficit',
            'description': 'performance, functioning, pleasure, psychomotor signs, anhedonia, loss of interest, reduced activity, difficulty doing things',
        },
        5: {
            'label': 'Anxiety & Distress',
            'description': 'irritability, withdrawal, unrest, agitation, worry, tension, nervousness',
        },
    },
    'anxiety': {
        1: {
            'label': 'Somatic Anxiety',
            'description': 'physiological manifestations or consequences of anxiety, neurophysiological, autonomic, vegetative symptoms, physical symptoms of anxiety like racing heart, sweating, trembling, shortness of breath',
        },
        2: {
            'label': 'Cognitive Anxiety',
            'description': 'distressing thoughts, panic, obsessive thoughts, predictions of the future, worry, fear, apprehension, anticipation of negative events',
        },
    },
    'psychosis': {
        1: {
            'label': 'Hallucinations',
            'description': 'hallucinations, illusions, misidentification, seeing hearing feeling or smelling things that are not there, perceptual disturbances',
        },
        2: {
            'label': 'Delusions',
            'description': 'delusions, false sense of presence, paranoid beliefs, persecution, jealousy, false beliefs that are firmly held despite evidence',
        },
    },
    'apathy': {
        1: {
            'label': 'Behavioral-Cognitive Apathy',
            'description': 'goal-directed cognition and behavior, interest in the new, curiosity, planning, thinking about goals and future, effort, dependency, initiative, getting things done, taking action, productivity',
        },
        2: {
            'label': 'Affective Apathy',
            'description': 'emotional responsivity, emotional blunting, reduced emotional reactions, indifference to positive or negative events',
        },
    },
    'impulse_control': {
        1: {
            'label': 'Pathological Gambling',
            'description': 'thoughts occupied by gambling, difficulty controlling thoughts about gambling, time spent on gambling, financial investment in gambling',
        },
        2: {
            'label': 'Hypersexuality',
            'description': 'thoughts occupied by sex, difficulty controlling thoughts about sex, time spent on sexual activities, financial investment in sexual activities',
        },
        3: {
            'label': 'Compulsive Buying',
            'description': 'thoughts occupied by buying, difficulty controlling thoughts about buying, time spent on buying things, financial investment in buying',
        },
        4: {
            'label': 'Compulsive Eating',
            'description': 'thoughts occupied by eating, difficulty controlling thoughts about eating, time spent on eating, overeating',
        },
        5: {
            'label': 'Punding-hobbyism',
            'description': 'display of stereotyped, repetitive behaviors, related or unrelated to hobbies, repetitive purposeless activities',
        },
        6: {
            'label': 'Dopamine Dysregulation Syndrome',
            'description': 'compulsive use of dopamine medications despite adequate motor benefits and the annoying consequences',
        },
    },
    'sleep': {
        1: {
            'label': 'Daytime Sleepiness and Alertness',
            'description': 'excessive sleepiness during the day, difficulty staying awake, reduced awareness or vigilance, drowsiness',
        },
        2: {
            'label': 'Nocturnal Sleep Disturbances',
            'description': 'anything that impairs the continuity and quality of nighttime sleep, difficulty falling asleep, waking during night, insomnia',
        },
        3: {
            'label': 'REM Sleep Behavior and Dreaming',
            'description': 'acting out dreams, vivid or disturbing dreams, movements during sleep, talking in sleep',
        },
        4: {
            'label': 'Sleep-Disordered Breathing',
            'description': 'snoring, pauses in breathing during sleep, sleep apnea, breathing difficulties at night',
        },
    },
}

# HuggingFace repository IDs — one per construct.
# Fill in your repo IDs after creating the repos on HuggingFace Hub.
# Format: 'your-org/harmonica-{construct}'
HF_REPOS = {
    'depression':     "julia-pfarr/HarmoniCA_depression",  # e.g. 'your-org/harmonica-depression'
    'apathy':         "julia-pfarr/HarmoniCA_apathy",
    'psychosis':      "julia-pfarr/HarmoniCA_psychosis",
    'anxiety':        "julia-pfarr/HarmoniCA_anxiety",
    'sleep':          "julia-pfarr/HarmoniCA_sleep",
    'impulse_control':"julia-pfarr/HarmoniCA_impulse-control",
}

# Best model type per construct (validated against round-2 expert survey)
# 'ft'       : fine-tuned contrastive/prototype model, no blending
# 'ft_knn'   : fine-tuned model + kNN blend with inventory pool
# 'base_knn' : base sentence-transformer + kNN against inventory pool
BEST_MODEL = {
    'depression':     'ft_knn',    # fine-tuned + kNN (alpha=0.5, k=5); round-2 κ=0.746
    'apathy':         'base_knn',  # base (BAAI/bge-large-en-v1.5) + kNN; round-1 κ=0.68
    'psychosis':      'ft',        # fine-tuned only; round-2 κ=0.700
    'anxiety':        'ft',        # fine-tuned only; round-2 κ=0.877
    'sleep':          'ft',        # fine-tuned only; round-2 κ=0.468
    'impulse_control':'ft',        # fine-tuned only (prototype model); round-2 κ=0.884
}

# kNN settings used during validation
KNN_K = 5
KNN_ALPHA = {
    'depression': 0.5,  # equal blend of fine-tuned probs and kNN probs
}

# Base model names for constructs that use base+kNN
BASE_MODEL_NAMES = {
    'apathy': 'BAAI/bge-large-en-v1.5',
}
