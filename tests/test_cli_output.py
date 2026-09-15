"""
Tests for harmonize.py's CLI output behavior: a results CSV must always be
written — to `<items>_harmonized.csv` next to the input file by default, or
to `--output <path>` if given — and it must contain only the questionnaires
present in the input items CSV.

`HarmoniCA.harmonize` is mocked so these run without any ML dependencies or
network access.
"""
import sys
import pandas as pd
import pytest
from unittest.mock import patch

from harmonica.harmonica import HarmoniCA
from harmonica import harmonize as hz

INVENTORY_COLUMNS = [
    'item_id', 'item_text', 'construct', 'questionnaire',
    'dimension', 'dimension_label', 'confidence',
    'source', 'date_added', 'model_version',
]


def _fake_harmonize(self, questionnaire, construct, items, force_rerun=False):
    return {
        'assignments': [
            {
                'item_id': it['item_id'],
                'item_text': it['item_text'],
                'dimension': 1,
                'dimension_label': 'Test Dim',
                'confidence': 0.9,
            }
            for it in items
        ],
        'source': 'model',
        'questionnaire': questionnaire,
        'construct': construct,
    }


@pytest.fixture
def items_csv(tmp_path):
    path = tmp_path / 'my_items.csv'
    pd.DataFrame([
        {'construct': 'depression', 'questionnaire': 'PHQ-9', 'item_id': 'PHQ-9_01', 'item_text': 'Little interest'},
        {'construct': 'anxiety', 'questionnaire': 'GAD-7', 'item_id': 'GAD7_01', 'item_text': 'Feeling nervous'},
    ]).to_csv(path, index=False)
    return path


@pytest.fixture
def empty_inventory(tmp_path):
    path = tmp_path / 'inventory.csv'
    pd.DataFrame(columns=INVENTORY_COLUMNS).to_csv(path, index=False)
    return path


def _run_cli(argv):
    with patch.object(sys, 'argv', argv), patch.object(HarmoniCA, 'harmonize', _fake_harmonize):
        hz.main()


def test_default_output_path_written_next_to_items(items_csv, empty_inventory, tmp_path, capsys):
    _run_cli(['harmonize.py', '--items', str(items_csv), '--inventory', str(empty_inventory)])

    expected_output = tmp_path / 'my_items_harmonized.csv'
    assert expected_output.exists()

    captured = capsys.readouterr()
    assert 'Saved to' in captured.out


def test_explicit_output_path_is_respected(items_csv, empty_inventory, tmp_path):
    custom_output = tmp_path / 'custom_results.csv'
    _run_cli([
        'harmonize.py', '--items', str(items_csv),
        '--inventory', str(empty_inventory), '--output', str(custom_output),
    ])

    assert custom_output.exists()
    assert not (tmp_path / 'my_items_harmonized.csv').exists()


def test_output_only_contains_requested_questionnaires(items_csv, empty_inventory, tmp_path):
    _run_cli(['harmonize.py', '--items', str(items_csv), '--inventory', str(empty_inventory)])

    out_df = pd.read_csv(tmp_path / 'my_items_harmonized.csv')
    assert set(out_df['questionnaire']) == {'PHQ-9', 'GAD-7'}
    assert set(out_df['item_id']) == {'PHQ-9_01', 'GAD7_01'}
    assert len(out_df) == 2  # nothing beyond what was requested


def test_output_excludes_questionnaires_not_in_items_csv(items_csv, empty_inventory, tmp_path):
    """A questionnaire that was never requested must never appear in the output,
    even if HarmoniCA's internals happened to know about it."""
    _run_cli(['harmonize.py', '--items', str(items_csv), '--inventory', str(empty_inventory)])

    out_df = pd.read_csv(tmp_path / 'my_items_harmonized.csv')
    assert 'BDI-I' not in set(out_df['questionnaire'])
