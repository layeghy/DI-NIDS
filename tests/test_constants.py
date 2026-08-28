from dinids.constants import FEATURE_COLUMNS, FLOW_COLUMNS, IDENTIFIER_COLUMNS


def test_feature_counts_match_research_inputs() -> None:
    assert len(FLOW_COLUMNS) == 43
    assert len(IDENTIFIER_COLUMNS) == 4
    assert len(FEATURE_COLUMNS) == 39
    assert not set(FEATURE_COLUMNS).intersection(IDENTIFIER_COLUMNS)
