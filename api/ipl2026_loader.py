"""
Aggregate IPL 2026 ball-by-ball CSV into per-player match stats and expected playing XIs.
Used for recent form / hot-tag calibration; H2H remains on the historical dataframe in main.py.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Short codes as in IPL_2026/ipl_2026_deliveries.csv batting_team / bowling_team
SHORT_TO_SCHEDULE_NAME = {
    "RCB": "Royal Challengers Bengaluru",
    "SRH": "Sunrisers Hyderabad",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "RR": "Rajasthan Royals",
    "CSK": "Chennai Super Kings",
    "PBKS": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "DC": "Delhi Capitals",
}

SCHEDULE_NAME_TO_SHORT = {v: k for k, v in SHORT_TO_SCHEDULE_NAME.items()}


def schedule_team_to_short(name: str) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip()
    if n in SCHEDULE_NAME_TO_SHORT:
        return SCHEDULE_NAME_TO_SHORT[n]
    n2 = n.replace("Bengaluru", "Bangalore")
    for full, short in SCHEDULE_NAME_TO_SHORT.items():
        if full.replace("Bengaluru", "Bangalore") == n2:
            return short
    return None


def _dedupe_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    subset = ["match_id", "innings", "over", "striker", "bowler"]
    if not all(c in df.columns for c in subset):
        return df
    return df.drop_duplicates(subset=subset, keep="first")


def _simplified_match_fantasy_proxy(row: pd.Series) -> float:
    """Correlated with Dream11-style points; only used for relative form ranking."""
    runs = float(row.get("runs_bat") or 0)
    wkts = float(row.get("wickets") or 0)
    balls_bat = float(row.get("balls_bat") or 0)
    balls_bowl = float(row.get("balls_bowl") or 0)
    rc = float(row.get("runs_conceded") or 0)
    pts = runs + wkts * 25.0
    if balls_bat > 0:
        sr = 100.0 * runs / balls_bat
        if sr >= 170:
            pts += 6.0
        elif sr >= 150:
            pts += 4.0
        elif sr >= 130:
            pts += 2.0
    if balls_bowl >= 6:
        overs = balls_bowl / 6.0
        econ = rc / overs if overs > 0 else 0.0
        if econ <= 6:
            pts += 6.0
        elif econ <= 8:
            pts += 4.0
        elif econ <= 10:
            pts += 2.0
    return pts


def build_ipl2026_aggregates(deliveries_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Set[str]]]]:
    """
    Returns:
      - player_match: columns match_no, player, team_short, runs_bat, balls_bat, wickets, balls_bowl,
                      runs_conceded, fantasy_proxy
      - xi_by_team_match: team_short -> { match_no: set(player_name) }
    """
    if not os.path.exists(deliveries_path):
        return pd.DataFrame(), {}

    raw = pd.read_csv(deliveries_path, low_memory=False)
    if raw.empty:
        return pd.DataFrame(), {}

    raw = _dedupe_deliveries(raw)
    raw["match_no"] = pd.to_numeric(raw["match_no"], errors="coerce").fillna(0).astype(int)
    raw["runs_of_bat"] = pd.to_numeric(raw.get("runs_of_bat", 0), errors="coerce").fillna(0)
    for c in ("wide", "noballs"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)
        else:
            raw[c] = 0
    raw["bowler_runs_against"] = raw["runs_of_bat"] + raw["wide"] + raw["noballs"]
    raw["is_wicket_ball"] = raw.get("wicket_type", "").fillna("").astype(str).str.len() > 0

    rows: List[dict] = []

    for mno in sorted(raw["match_no"].unique()):
        sub = raw[raw["match_no"] == mno]
        for team_short in SHORT_TO_SCHEDULE_NAME:
            team_bat = sub[sub["batting_team"] == team_short]
            team_bowl = sub[sub["bowling_team"] == team_short]

            if not team_bat.empty:
                g = team_bat.groupby("striker", as_index=False).agg(
                    runs_bat=("runs_of_bat", "sum"),
                    balls_bat=("runs_of_bat", "count"),
                )
                for _, r in g.iterrows():
                    rows.append(
                        {
                            "match_no": int(mno),
                            "player": str(r["striker"]).strip(),
                            "team_short": team_short,
                            "runs_bat": float(r["runs_bat"]),
                            "balls_bat": int(r["balls_bat"]),
                            "wickets": 0.0,
                            "balls_bowl": 0,
                            "runs_conceded": 0.0,
                        }
                    )

            if not team_bowl.empty:
                g2 = team_bowl.groupby("bowler", as_index=False).agg(
                    runs_conceded=("bowler_runs_against", "sum"),
                    balls_bowl=("bowler", "count"),
                    wickets=("is_wicket_ball", lambda x: int(x.sum())),
                )
                for _, r in g2.iterrows():
                    rows.append(
                        {
                            "match_no": int(mno),
                            "player": str(r["bowler"]).strip(),
                            "team_short": team_short,
                            "runs_bat": 0.0,
                            "balls_bat": 0,
                            "wickets": float(r["wickets"]),
                            "balls_bowl": int(r["balls_bowl"]),
                            "runs_conceded": float(r["runs_conceded"]),
                        }
                    )

                if "fielder" in team_bowl.columns:
                    for fname in team_bowl["fielder"].dropna().unique():
                        fn = str(fname).strip()
                        if not fn:
                            continue
                        rows.append(
                            {
                                "match_no": int(mno),
                                "player": fn,
                                "team_short": team_short,
                                "runs_bat": 0.0,
                                "balls_bat": 0,
                                "wickets": 0.0,
                                "balls_bowl": 0,
                                "runs_conceded": 0.0,
                            }
                        )

    if not rows:
        return pd.DataFrame(), {}

    opp_pair: Dict[Tuple[int, str], str] = {}
    for mno in sorted(raw["match_no"].unique()):
        sub = raw[raw["match_no"] == mno]
        ts = set(sub["batting_team"].dropna().astype(str).unique()) | set(
            sub["bowling_team"].dropna().astype(str).unique()
        )
        ts = {t for t in ts if t}
        if len(ts) == 2:
            a, b = tuple(ts)
            opp_pair[(int(mno), a)] = b
            opp_pair[(int(mno), b)] = a

    pm = pd.DataFrame(rows)
    pm = (
        pm.groupby(["match_no", "player", "team_short"], as_index=False)
        .agg(
            {
                "runs_bat": "sum",
                "balls_bat": "sum",
                "wickets": "sum",
                "balls_bowl": "sum",
                "runs_conceded": "sum",
            }
        )
    )
    pm["fantasy_proxy"] = pm.apply(_simplified_match_fantasy_proxy, axis=1)
    pm["opponent_short"] = pm.apply(
        lambda r: opp_pair.get((int(r["match_no"]), str(r["team_short"])), ""), axis=1
    )

    # Keep only players who actually batted, bowled, or took a wicket (excludes padded rows / non-participants)
    inv = (
        pm["runs_bat"].fillna(0)
        + pm["balls_bat"].fillna(0)
        + pm["balls_bowl"].fillna(0)
        + pm["wickets"].fillna(0)
    )
    pm = pm[inv > 0].copy()

    xi_by_team_match: Dict[str, Dict[int, Set[str]]] = {}
    for team_short in SHORT_TO_SCHEDULE_NAME:
        xi_by_team_match[team_short] = {}
        for mno in sorted(pm["match_no"].unique()):
            tsub = pm[(pm["match_no"] == mno) & (pm["team_short"] == team_short)]
            if tsub.empty:
                continue
            tsub = tsub.copy()
            tsub["involvement"] = (
                tsub["balls_bat"].fillna(0)
                + tsub["balls_bowl"].fillna(0)
                + tsub["wickets"].fillna(0) * 5
            )
            tsub = tsub.sort_values("involvement", ascending=False)
            picked = tsub.head(11)["player"].astype(str).tolist()
            xi_by_team_match[team_short][int(mno)] = set(picked)

    return pm, xi_by_team_match


def expected_pool_for_fixture(
    xi_by_team_match: Dict[str, Dict[int, Set[str]]],
    team1_schedule_name: str,
    team2_schedule_name: str,
    fixture_match_num: int,
) -> Set[str]:
    """
    For an upcoming fixture (no ball data yet), use each side's last completed match before fixture_match_num.
    For a completed fixture, use that match number's XI if present.
    """
    s1 = schedule_team_to_short(team1_schedule_name)
    s2 = schedule_team_to_short(team2_schedule_name)
    if not s1 or not s2:
        return set()

    pool: Set[str] = set()

    def last_xi(team: str, before: int) -> Set[str]:
        d = xi_by_team_match.get(team) or {}
        nums = [n for n in d.keys() if n < before]
        if not nums:
            return set()
        m = max(nums)
        return set(d.get(m, set()))

    for t in (s1, s2):
        d = xi_by_team_match.get(t) or {}
        if fixture_match_num in d and d[fixture_match_num]:
            pool |= set(d[fixture_match_num])

    if pool:
        return pool

    pool |= last_xi(s1, fixture_match_num)
    pool |= last_xi(s2, fixture_match_num)
    return pool


def expected_xi_for_franchise(
    xi_by_team_match: Dict[str, Dict[int, Set[str]]],
    franchise_schedule_name: str,
    fixture_match_num: int,
) -> Set[str]:
    """
    Expected XI for one franchise only (not merged with opponent).
    Uses that match's XI if present in data; else last completed XI before fixture_match_num.
    """
    ts = schedule_team_to_short(franchise_schedule_name)
    if not ts:
        return set()
    d = xi_by_team_match.get(ts) or {}
    fm = int(fixture_match_num)
    if fm in d and d[fm]:
        return set(d[fm])
    nums = [n for n in d.keys() if n < fm]
    if not nums:
        return set()
    m = max(nums)
    return set(d.get(m, set()))


def ipl2026_last5_mean_proxy(
    player_match: pd.DataFrame, player_name: str, names_match, before_match_num: Optional[int] = None
) -> Optional[float]:
    """Mean fantasy_proxy over last 5 IPL 2026 games for this player (by match_no)."""
    if player_match is None or player_match.empty:
        return None
    sub = player_match[player_match["player"].apply(lambda x: names_match(player_name, x))]
    if sub.empty:
        return None
    sub = sub.sort_values("match_no")
    if before_match_num is not None:
        sub = sub[sub["match_no"] < int(before_match_num)]
    if sub.empty:
        return None
    tail = sub.tail(5)
    return float(tail["fantasy_proxy"].mean())


def ipl2026_season_l5_percentile_line(
    player_match: pd.DataFrame,
    before_match_num: Optional[int] = None,
    quantile: float = 0.74,
) -> Optional[float]:
    """
    League-wide cut on last-5 mean fantasy_proxy (players with >=2 games): used to tag 'standout' season form
    even when the ML projection is conservative (e.g. young players breaking out in IPL 2026).
    """
    if player_match is None or player_match.empty:
        return None
    sub = player_match
    if before_match_num is not None:
        sub = sub[sub["match_no"] < int(before_match_num)]
    if sub.empty:
        return None
    vals: List[float] = []
    for pname in sub["player"].unique():
        one = sub[sub["player"] == pname].sort_values("match_no")
        if len(one) < 2:
            continue
        tail = one.tail(min(5, len(one)))
        vals.append(float(tail["fantasy_proxy"].mean()))
    if len(vals) < 10:
        return None
    return float(pd.Series(vals).quantile(quantile))


def ipl2026_form_context(
    player_match: pd.DataFrame,
    player_name: str,
    names_match,
    before_match_num: Optional[int] = None,
    franchise_schedule_name: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    IPL 2026-only form: last-5 mean vs mean of all earlier 2026 games (trend).
    Used for hot-tag and calibrated projection bumps.
    If franchise_schedule_name is set, only games for that team (e.g. Delhi Capitals) count.
    """
    out: Dict[str, Optional[float]] = {"l5": None, "prior": None, "n": 0}
    if player_match is None or player_match.empty:
        return out
    sub = player_match[player_match["player"].apply(lambda x: names_match(player_name, x))]
    if sub.empty:
        return out
    if franchise_schedule_name:
        fts = schedule_team_to_short(franchise_schedule_name)
        if fts:
            sub = sub[sub["team_short"] == fts]
    if sub.empty:
        return out
    sub = sub.sort_values("match_no")
    if before_match_num is not None:
        sub = sub[sub["match_no"] < int(before_match_num)]
    if sub.empty:
        return out
    n = int(len(sub))
    out["n"] = n
    tail = sub.tail(min(5, n))
    out["l5"] = float(tail["fantasy_proxy"].mean())
    head = sub.iloc[: max(0, len(sub) - len(tail))]
    if len(head) >= 1:
        out["prior"] = float(head["fantasy_proxy"].mean())
    return out


def ipl2026_recent_matches_for_player(
    player_match: pd.DataFrame, player_name: str, names_match, limit: int = 5
) -> pd.DataFrame:
    """Rows from 2026 aggregates for modal /player_stats recent form."""
    if player_match is None or player_match.empty:
        return pd.DataFrame()
    sub = player_match[player_match["player"].apply(lambda x: names_match(player_name, x))]
    if sub.empty:
        return pd.DataFrame()
    return sub.sort_values("match_no").tail(limit)


def ipl2026_onfield_appearances_before_fixture(
    player_match: pd.DataFrame,
    player_name: str,
    names_match,
    fixture_match_num: int,
    fixture_is_upcoming: bool,
    franchise_schedule_name: Optional[str] = None,
) -> int:
    """
    Count IPL 2026 on-field games for this player before the fixture.
    If franchise_schedule_name is set, only games played for that franchise count
    (excludes minutes earned for another team before a transfer).
    """
    if player_match is None or player_match.empty:
        return 0
    sub = player_match[player_match["player"].apply(lambda x: names_match(player_name, x))]
    if sub.empty:
        return 0
    if franchise_schedule_name:
        fts = schedule_team_to_short(franchise_schedule_name)
        if fts:
            sub = sub[sub["team_short"] == fts]
    if sub.empty:
        return 0
    mno = int(fixture_match_num)
    if fixture_is_upcoming:
        sub = sub[sub["match_no"] < mno]
    else:
        sub = sub[sub["match_no"] <= mno]
    return int(len(sub))
