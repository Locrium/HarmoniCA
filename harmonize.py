"""
HarmoniCA — command-line entry point.

Usage examples
--------------
# Assign items from a CSV file:
python harmonize.py --items items.csv

# items.csv must have columns: construct, questionnaire, item_id, item_text
# Additional columns are ignored. Multiple questionnaires and constructs are supported.

# Programmatic use:
from harmonica import HarmoniCA

hca = HarmoniCA(models_dir='models', inventory_path='inventory/harmonized_inventory.csv')
result = hca.harmonize(
    questionnaire='PHQ-9',
    construct='depression',
    items=[
        {'item_id': 'PHQ9_01', 'item_text': 'Little interest or pleasure in doing things'},
        {'item_id': 'PHQ9_02', 'item_text': 'Feeling down, depressed, or hopeless'},
    ]
)
for a in result['assignments']:
    print(f"{a['item_id']}: Dim {a['dimension']} — {a['dimension_label']} (conf={a['confidence']:.2f})")
"""

import argparse
import json
import pandas as pd
from pathlib import Path

from harmonica import HarmoniCA

DEFAULT_MODELS_DIR    = Path(__file__).parent / 'models'
DEFAULT_INVENTORY     = Path(__file__).parent / 'inventory' / 'harmonized_inventory.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Assign questionnaire items to construct dimensions using HarmoniCA.'
    )
    parser.add_argument('--items', '-i', required=True,
                        help='CSV file with columns: construct, questionnaire, item_id, item_text')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: print to stdout)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore inventory and always run the model')
    parser.add_argument('--models-dir', default=str(DEFAULT_MODELS_DIR))
    parser.add_argument('--inventory', default=str(DEFAULT_INVENTORY))

    args = parser.parse_args()

    # Load items
    items_df = pd.read_csv(args.items)
    required = {'construct', 'questionnaire', 'item_id', 'item_text'}
    missing = required - set(items_df.columns)
    if missing:
        raise ValueError(f"Items CSV is missing required columns: {missing}")

    # Run per (construct, questionnaire) group
    hca = HarmoniCA(models_dir=args.models_dir, inventory_path=args.inventory)
    all_results = []
    for (construct, questionnaire), group in items_df.groupby(['construct', 'questionnaire']):
        items = group[['item_id', 'item_text']].to_dict('records')
        result = hca.harmonize(
            questionnaire=questionnaire,
            construct=construct,
            items=items,
            force_rerun=args.force_rerun,
        )
        group_df = pd.DataFrame(result['assignments'])
        group_df.insert(0, 'questionnaire', questionnaire)
        group_df.insert(1, 'construct', construct)
        group_df['source'] = result['source']
        all_results.append(group_df)

    out_df = pd.concat(all_results, ignore_index=True)

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(out_df[['questionnaire', 'construct', 'item_id', 'dimension', 'dimension_label', 'confidence']].to_string(index=False))



if __name__ == '__main__':
    main()
