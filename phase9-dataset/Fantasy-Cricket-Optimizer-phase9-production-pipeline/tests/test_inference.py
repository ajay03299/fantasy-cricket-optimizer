"""Tests for the FantasyPredictor inference engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.inference.inference_engine import FantasyPredictor


@pytest.fixture(scope="module")
def predictor():
    return FantasyPredictor()


def test_predictor_loads(predictor):
    assert len(predictor.all_players) > 0
    assert len(predictor.all_teams) > 0


def test_fuzzy_player_match(predictor):
    # "V Kohli" should be in the database
    resolved = predictor.resolve_player_name("V Kohli")
    assert resolved is not None
    # Common typo / full name should still resolve
    fuzzy = predictor.resolve_player_name("Virat Kohli")
    assert fuzzy is not None


def test_list_available_players(predictor):
    matches = predictor.list_available_players("Kohli", limit=5)
    assert len(matches) > 0
    assert any("Kohli" in m for m in matches)


def test_get_team_recent_xi(predictor):
    teams = predictor.all_teams
    if teams:
        xi = predictor.get_team_recent_xi(teams[0])
        assert len(xi) > 0
        assert len(xi) <= 11


def test_invalid_xi_size(predictor):
    if not predictor.models_loaded:
        pytest.skip("models not loaded")
    with pytest.raises(ValueError, match="11 players"):
        predictor.predict_match({
            'match_date': '2026-04-15',
            'venue': 'Wankhede',
            'team1': predictor.all_teams[0],
            'team2': predictor.all_teams[1],
            'team1_xi': ['p1', 'p2'],  # too few
            'team2_xi': ['p1'] * 11,
        })


def test_unknown_team(predictor):
    if not predictor.models_loaded:
        pytest.skip("models not loaded")
    with pytest.raises(ValueError, match="Unknown team"):
        predictor.predict_match({
            'match_date': '2026-04-15',
            'venue': 'Wankhede',
            'team1': 'Fake Team',
            'team2': predictor.all_teams[0],
            'team1_xi': ['p'] * 11,
            'team2_xi': ['p'] * 11,
        })
