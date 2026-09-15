"""
Tests for detecting the same questionnaire item under a different item_id
coding (e.g. 'PHQ9_1' vs the inventory's 'PHQ-9_01') via normalized item-text
matching, and resolving it through the `confirm_match` callback.
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
    def _run(self, construct, items):
        calls.append([it['item_id'] for it in items])
        return [
            {
                'item_id': it['item_id'],
                'item_text': it['item_text'],
                'dimension': 99,
                'dimension_label': 'NewDim',
                'confidence': 0.5,
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


def test_text_match_with_different_id_is_detected(hca):
    """A differently-coded item_id with matching text should trigger confirm_match."""
    calls = []
    seen = []

    def confirm(user_item, inv_row):
        seen.append((user_item['item_id'], inv_row['item_id']))
        return True  # user confirms it's the same item

    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_1', 'item_text': 'Little interest or pleasure in doing things.'}],
            confirm_match=confirm,
        )

    assert seen == [('PHQ9_1', 'PHQ-9_01')]
    assert calls == []  # model was never run — reused the inventory assignment
    assert result['source'] == 'inventory'
    assert result['assignments'][0]['item_id'] == 'PHQ9_1'  # user's own id is preserved
    assert result['assignments'][0]['dimension'] == 4        # dimension reused from inventory


def test_text_match_declined_runs_model_as_new_item(hca):
    calls = []

    def confirm(user_item, inv_row):
        return False  # user says it's actually a different item

    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_1', 'item_text': 'Little interest or pleasure in doing things.'}],
            confirm_match=confirm,
        )

    assert calls == [['PHQ9_1']]  # ran through the model like any new item
    assert result['source'] == 'model'
    assert result['assignments'][0]['dimension'] == 99


def test_normalization_ignores_case_whitespace_and_punctuation(hca):
    calls = []
    confirmations = []

    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_1', 'item_text': '  LITTLE INTEREST OR PLEASURE IN DOING THINGS  '}],
            confirm_match=lambda u, r: confirmations.append(1) or True,
        )

    assert confirmations == [1]
    assert calls == []
    assert result['assignments'][0]['dimension'] == 4


def test_no_text_match_never_calls_confirm(hca):
    """A genuinely new item (no text match) should go straight to the model
    without invoking confirm_match at all."""
    calls = []
    confirm_calls = []

    def confirm(user_item, inv_row):
        confirm_calls.append(1)
        return True

    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_2', 'item_text': 'Feeling down, depressed, or hopeless.'}],
            confirm_match=confirm,
        )

    assert confirm_calls == []
    assert calls == [['PHQ9_2']]
    assert result['source'] == 'model'


def test_default_confirm_match_prompts_interactively(hca):
    """Without an explicit confirm_match, the default should prompt via input()."""
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)), \
         patch('builtins.input', return_value='y') as mock_input:
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_1', 'item_text': 'Little interest or pleasure in doing things.'}],
        )

    mock_input.assert_called_once()
    assert calls == []
    assert result['source'] == 'inventory'


def test_default_confirm_match_declines_on_no(hca):
    calls = []
    with patch.object(HarmoniCA, '_run_model', _fake_run_model(calls)), \
         patch('builtins.input', return_value='n'):
        result = hca.harmonize(
            questionnaire='PHQ-9',
            construct='depression',
            items=[{'item_id': 'PHQ9_1', 'item_text': 'Little interest or pleasure in doing things.'}],
        )

    assert calls == [['PHQ9_1']]
    assert result['source'] == 'model'
