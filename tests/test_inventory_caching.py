"""
Tests for HarmoniCA's inventory caching behavior in `harmonize()`.

These mock out `_run_model` so they run fast and without any ML dependencies
(torch, sentence-transformers) or network access — they only exercise the
inventory lookup / merge logic in `harmonica/harmonica/__init__.py`.
"""
import pandas as pd
import pytest
from unittest.mock import patch

from harmonica.harmonica import HarmoniCA

INVENTORY_COLUMNS = [
    'item_id', 'item_text', 'construct', 'questionnaire',
    'dimension', 'dimension_label', 'confidence',
    'source', 'date_added', 'model_version',
]


def _make_inventory(path, rows):
    pd.DataFrame(rows, columns=INVENTORY_COLUMNS).to_csv(path, index=False)


def _fake_run_model(calls):
    """Records which item_ids were sent to the model and returns dimension=2 for each."""
    def _run(self, construct, items):
        calls.append([it['item_id'] for it in items])
        return [
            {
                'item_id': it['item_id'],
                'item_text': it['item_text'],
                'dimension': 2,
                'dimension_label': 'NewDim',
                'confidence': 0.99,
            }
            for it in items
        ]
    return _run


@pytest.fixture
def hca(tmp_path):
    inventory_path = tmp_path / 'inventory.csv'
    _make_inventory(inventory_path, [
        {
            'item_id': 'PHQ-9_01',
            'item_text': 'Little interest or pleasure in doing things.',
            'construct': 'depression',
            'questionnaire': 'PHQ-9',
            'dimension': 4,
            'dimension_label': 'Activity & interest deficits',
            'confidence': 0.9,
            'source': 'expert_consensus',
            'date_added': '2026-01-01',
            'model_version': 'ft',
        },
    ])
    return HarmoniCA(models_dir=str(tmp_path / 'models'), inventory_path=str(inventory_path))


def test_full_inventory_match_skips_model(hca):
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ-9_01', 'item_text': 'Little interest or pleasure in doing things.'}],
        )

    assert result['source'] == 'inventory'
    assert calls == []
    assert result['assignments'][0]['dimension'] == 4


def test_partial_match_runs_model_only_for_missing_items(hca):
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[
                {'item_id': 'PHQ-9_01', 'item_text': 'Little interest or pleasure in doing things.'},
                {'item_id': 'PHQ-9_02', 'item_text': 'Feeling down, depressed, or hopeless.'},
            ],
        )

    assert result['source'] == 'mixed'
    assert calls == [['PHQ-9_02']]  # only the missing item was sent to the model

    by_id = {a['item_id']: a for a in result['assignments']}
    assert by_id['PHQ-9_01']['dimension'] == 4  # served from inventory
    assert by_id['PHQ-9_02']['dimension'] == 2  # freshly predicted

    # input order is preserved
    assert [a['item_id'] for a in result['assignments']] == ['PHQ-9_01', 'PHQ-9_02']

    # inventory now holds exactly one row per item — no duplicate for PHQ-9_01
    updated = pd.read_csv(hca.inventory_path)
    assert (updated['item_id'] == 'PHQ-9_01').sum() == 1
    assert (updated['item_id'] == 'PHQ-9_02').sum() == 1


def test_no_match_runs_model_for_all_items(hca):
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='GAD-7',
            construct='anxiety',
            items=[{'item_id': 'GAD7_01', 'item_text': 'Feeling nervous, anxious, or on edge.'}],
        )

    assert result['source'] == 'model'
    assert calls == [['GAD7_01']]


def test_force_rerun_bypasses_inventory(hca):
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ-9_01', 'item_text': 'Little interest or pleasure in doing things.'}],
            force_rerun=True,
        )

    assert result['source'] == 'model'
    assert calls == [['PHQ-9_01']]


def test_repeated_call_after_partial_fill_is_pure_inventory_hit(hca):
    calls = []
    items = [
        {'item_id': 'PHQ-9_01', 'item_text': 'Little interest or pleasure in doing things.'},
        {'item_id': 'PHQ-9_02', 'item_text': 'Feeling down, depressed, or hopeless.'},
    ]
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        hca.harmonize(questionnaire='PHQ-9', construct='depression', items=items)
        result2 = hca.harmonize(questionnaire='PHQ-9', construct='depression', items=items)

    assert result2['source'] == 'inventory'
    assert calls == [['PHQ-9_02']]  # model was only ever called once, during the first pass
