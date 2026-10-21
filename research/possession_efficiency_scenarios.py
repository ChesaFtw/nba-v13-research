"""Pregame possession/efficiency scenario engine for NBA V13.

Research-only. Inputs MUST be pregame, timestamped and independently sourced.
No team statistics, lineup data, or predictive accuracy are invented here.
"""
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class Team:
    pace: float                   # possessions per 48, pregame rolling estimate
    offensive_rating: float       # points per 100 possessions
    defensive_rating: float       # points allowed per 100 possessions
    transition_adjustment: float = 0.0  # points per 100, pregame matchup estimate
    lineup_adjustment: float = 0.0      # points per 100, pregame expected lineup
    rest_adjustment: float = 0.0        # points per 100, pregame travel/rest

@dataclass(frozen=True)
class Scenario:
    possessions: float
    away_points: float
    home_points: float
    total: float
    home_margin: float

def project(away: Team, home: Team, league_rating: float,
            pace_adjustment: float = 0.0, home_court_points: float = 0.0) -> Scenario:
    """Neutral multiplicative offense-vs-defense baseline; calibrated coefficients TBD.

    Pace uses harmonic mean to avoid fast team dominating slow team.
    Home court applied as score difference, holding game total fixed.
    """
    vals=(away.pace,home.pace,away.offensive_rating,home.offensive_rating,
          away.defensive_rating,home.defensive_rating,league_rating)
    if not all(isfinite(x) and x>0 for x in vals):
        raise ValueError("Positive finite pace/ratings required")
    if not all(isfinite(x) for x in (pace_adjustment,home_court_points,
             away.transition_adjustment,home.transition_adjustment,
             away.lineup_adjustment,home.lineup_adjustment,
             away.rest_adjustment,home.rest_adjustment)):
        raise ValueError("Finite adjustments required")
    possessions=2*away.pace*home.pace/(away.pace+home.pace)+pace_adjustment
    if possessions<=0: raise ValueError("Invalid adjusted possessions")
    away_eff=away.offensive_rating*home.defensive_rating/league_rating
    home_eff=home.offensive_rating*away.defensive_rating/league_rating
    away_eff+=away.transition_adjustment+away.lineup_adjustment+away.rest_adjustment
    home_eff+=home.transition_adjustment+home.lineup_adjustment+home.rest_adjustment
    if min(away_eff,home_eff)<=0:raise ValueError("Invalid adjusted efficiency")
    away_pts=possessions*away_eff/100-home_court_points/2
    home_pts=possessions*home_eff/100+home_court_points/2
    return Scenario(possessions,away_pts,home_pts,away_pts+home_pts,home_pts-away_pts)

def sensitivity(away: Team, home: Team, league_rating: float,
                pace_shocks=(-3.0,0.0,3.0), efficiency_shocks=(-3.0,0.0,3.0)):
    """Scenario grid, not probabilities: paired efficiency shock in points/100."""
    rows=[]
    for pace in pace_shocks:
        for eff in efficiency_shocks:
            a=Team(away.pace,away.offensive_rating,away.defensive_rating,
                   away.transition_adjustment,away.lineup_adjustment+eff,away.rest_adjustment)
            h=Team(home.pace,home.offensive_rating,home.defensive_rating,
                   home.transition_adjustment,home.lineup_adjustment+eff,home.rest_adjustment)
            p=project(a,h,league_rating,pace)
            rows.append({"pace_shock":pace,"efficiency_shock":eff,
                         "possessions":round(p.possessions,2),
                         "total":round(p.total,2),"home_margin":round(p.home_margin,2)})
    return rows
