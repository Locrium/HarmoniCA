"""
HarmoniCA — command-line entry point.

Usage examples
--------------
# Assign items from a CSV file:
python harmonize.py --questionnaire PHQ-9 --construct depression --items items.csv

# items.csv format (two columns, no header required):
#   item_id, item_text
#   PHQ9_01, "Little interest or pleasure in doing things"
#   PHQ9_02, "Feeling down, depressed, or hopeless"
#   ...

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
    parser.add_argument('--questionnaire', '-q', required=True,
                        help='Questionnaire name (e.g. PHQ-9)')
    parser.add_argument('--construct', '-c', required=True,
                        help='Construct (depression / apathy / psychosis / anxiety / sleep / impulse_control)')
    parser.add_argument('--items', '-i', required=True,
                        help='CSV file with columns item_id, item_text')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: print to stdout)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Ignore inventory and always run the model')
    parser.add_argument('--models-dir', default=str(DEFAULT_MODELS_DIR))
    parser.add_argument('--inventory', default=str(DEFAULT_INVENTORY))

    args = parser.parse_args()

    # Load items
    items_df = pd.read_csv(args.items, header=None if _has_no_header(args.items) else 'infer')
    if items_df.shape[1] >= 2:
        items_df.columns = ['item_id', 'item_text'] + list(items_df.columns[2:])
    items = items_df[['item_id', 'item_text']].to_dict('records')

    # Run
    hca    = HarmoniCA(models_dir=args.models_dir, inventory_path=args.inventory)
    result = hca.harmonize(
        questionnaire=args.questionnaire,
        construct=args.construct,
        items=items,
        force_rerun=args.force_rerun,
    )

    out_df = pd.DataFrame(result['assignments'])
    out_df.insert(0, 'questionnaire', args.questionnaire)
    out_df.insert(1, 'construct',     args.construct)
    out_df['source'] = result['source']

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(f"\nSource: {result['source']}")
        print(out_df[['item_id', 'dimension', 'dimension_label', 'confidence']].to_string(index=False))


def _has_no_header(path: str) -> bool:
    with open(path) as f:
        first = f.readline().split(',')[0].strip()
    return first.lower() not in ('item_id', 'id', 'item')


if __name__ == '__main__':
    main()
