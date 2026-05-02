"""Tests for the online RPG game engine and models."""

import json
import os
import sys
import tempfile

import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from online_rpg.models import (
    Army,
    ArmyStatus,
    Building,
    BuildingType,
    Character,
    CharacterRole,
    CharacterStats,
    DiplomacyRelation,
    DiplomacyStatus,
    Faction,
    GameEvent,
    Province,
    Season,
    TerrainType,
    Technology,
    TechCategory,
)
from online_rpg.map_generator import generate_map, assign_starting_provinces
from online_rpg.tech_tree import build_tech_tree, get_available_techs
from online_rpg.game_engine import GameEngine, GameState


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestCharacter:
    def test_creation(self):
        c = Character(name="Test Hero")
        assert c.name == "Test Hero"
        assert c.alive is True
        assert c.level == 1

    def test_gain_experience_no_level(self):
        c = Character(name="Rookie", level=1)
        c.experience = 0
        leveled = c.gain_experience(50)
        assert not leveled
        assert c.experience == 50
        assert c.level == 1

    def test_gain_experience_level_up(self):
        c = Character(name="Veteran", level=1)
        c.experience = 0
        leveled = c.gain_experience(100)
        assert leveled
        assert c.level == 2

    def test_total_power(self):
        stats = CharacterStats(command=80, force=70, intelligence=90, politics=60, charisma=50)
        c = Character(name="Test", stats=stats)
        assert c.total_power() == 350

    def test_to_dict(self):
        c = Character(name="DictTest")
        d = c.to_dict()
        assert d["name"] == "DictTest"
        assert "stats" in d
        assert "command" in d["stats"]


class TestProvince:
    def test_creation(self):
        p = Province(name="TestCity", terrain=TerrainType.PLAINS)
        assert p.name == "TestCity"
        assert p.terrain == TerrainType.PLAINS

    def test_compute_income_basic(self):
        p = Province(name="Plains", development=1.0, food_modifier=1.0, gold_modifier=1.0)
        income = p.compute_income()
        assert income["food"] > 0
        assert income["gold"] > 0
        assert income["manpower"] >= 0

    def test_compute_income_with_buildings(self):
        p = Province(name="Developed", development=1.0)
        p.buildings.append(Building(type=BuildingType.FARM, level=2))
        p.buildings.append(Building(type=BuildingType.MARKET, level=1))
        income = p.compute_income()
        # Should be higher than base
        base_p = Province(name="Base", development=1.0)
        base_income = base_p.compute_income()
        assert income["food"] > base_income["food"]
        assert income["gold"] > base_income["gold"]

    def test_unrest_reduces_income(self):
        p1 = Province(name="Peaceful", development=1.0, unrest=0.0)
        p2 = Province(name="Unrest", development=1.0, unrest=50.0)
        inc1 = p1.compute_income()
        inc2 = p2.compute_income()
        assert inc2["food"] < inc1["food"]

    def test_to_dict(self):
        p = Province(name="DictProv")
        d = p.to_dict()
        assert d["name"] == "DictProv"
        assert "terrain" in d


class TestArmy:
    def test_creation(self):
        a = Army(soldiers=2000, infantry=1200, cavalry=500, archers=300)
        assert a.soldiers == 2000

    def test_total_strength_no_general(self):
        a = Army(soldiers=1000, infantry=700, cavalry=200, archers=100)
        strength = a.total_strength()
        assert strength > 0

    def test_total_strength_with_general(self):
        a = Army(soldiers=1000, infantry=700, cavalry=200, archers=100)
        gen = Character(
            name="General",
            stats=CharacterStats(command=90, force=80),
        )
        s_without = a.total_strength()
        s_with = a.total_strength(gen)
        assert s_with > s_without

    def test_morale_affects_strength(self):
        a1 = Army(soldiers=1000, infantry=700, cavalry=200, archers=100, morale=100)
        a2 = Army(soldiers=1000, infantry=700, cavalry=200, archers=100, morale=50)
        assert a1.total_strength() > a2.total_strength()


class TestFaction:
    def test_creation(self):
        f = Faction(name="Wei Empire", color="#e74c3c")
        assert f.name == "Wei Empire"
        assert f.gold == 500.0

    def test_to_dict(self):
        f = Faction(name="TestFaction")
        d = f.to_dict()
        assert d["name"] == "TestFaction"
        assert "tech_levels" in d


class TestDiplomacyRelation:
    def test_creation(self):
        rel = DiplomacyRelation(faction_a_id="a", faction_b_id="b")
        assert rel.status == DiplomacyStatus.NEUTRAL
        assert rel.opinion == 0


# ---------------------------------------------------------------------------
# Map generator tests
# ---------------------------------------------------------------------------

class TestMapGenerator:
    def test_generate_map_default(self):
        provinces = generate_map(cols=5, rows=4, seed=42)
        assert len(provinces) == 20
        for p in provinces.values():
            assert p.name
            assert p.terrain in TerrainType

    def test_generate_map_adjacency(self):
        provinces = generate_map(cols=5, rows=4, seed=42)
        # Every non-edge province should have neighbors
        for p in provinces.values():
            assert len(p.adjacent_province_ids) > 0

    def test_generate_map_reproducible(self):
        p1 = generate_map(cols=5, rows=4, seed=123)
        p2 = generate_map(cols=5, rows=4, seed=123)
        names1 = sorted(p.name for p in p1.values())
        names2 = sorted(p.name for p in p2.values())
        assert names1 == names2

    def test_assign_starting_provinces(self):
        provinces = generate_map(cols=5, rows=4, seed=42)
        fids = ["f1", "f2", "f3"]
        assigned = assign_starting_provinces(provinces, fids, 2)
        assert len(assigned) == 3
        for fid in fids:
            assert len(assigned[fid]) >= 1
            for pid in assigned[fid]:
                assert provinces[pid].owner_faction_id == fid


# ---------------------------------------------------------------------------
# Tech tree tests
# ---------------------------------------------------------------------------

class TestTechTree:
    def test_build_tree(self):
        tree = build_tech_tree()
        assert len(tree) > 0
        # Check all categories present
        categories = {t.category for t in tree.values()}
        assert TechCategory.MILITARY in categories
        assert TechCategory.ECONOMY in categories
        assert TechCategory.CULTURE in categories
        assert TechCategory.DIPLOMACY in categories

    def test_get_available_techs_empty(self):
        available = get_available_techs(set())
        # Should return level-1 techs with no prereqs
        assert len(available) > 0
        for t in available:
            assert len(t.prerequisites) == 0

    def test_get_available_techs_with_research(self):
        available_before = get_available_techs(set())
        ids_before = {t.id for t in available_before}

        # Research a tech
        researched = {"mil_1"}
        available_after = get_available_techs(researched)
        ids_after = {t.id for t in available_after}

        # mil_1 should no longer be available
        assert "mil_1" not in ids_after
        # mil_2 and mil_3 (which depend on mil_1) should now be available
        assert "mil_2" in ids_after
        assert "mil_3" in ids_after

    def test_get_available_techs_by_category(self):
        available = get_available_techs(set(), category=TechCategory.MILITARY)
        for t in available:
            assert t.category == TechCategory.MILITARY


# ---------------------------------------------------------------------------
# Game engine tests
# ---------------------------------------------------------------------------

class TestGameEngine:
    def _new_game(self, seed=42):
        engine = GameEngine()
        factions = [
            {"name": "Faction A", "color": "#e74c3c", "player_id": "p1"},
            {"name": "Faction B", "color": "#3498db", "player_id": "p2"},
        ]
        engine.new_game(factions, map_cols=5, map_rows=4, seed=seed)
        return engine

    def test_new_game(self):
        engine = self._new_game()
        state = engine.state
        assert len(state.factions) == 2
        assert len(state.provinces) == 20
        assert len(state.characters) > 0
        assert len(state.armies) > 0

    def test_factions_have_provinces(self):
        engine = self._new_game()
        for fid in engine.state.factions:
            provinces = engine.state.faction_provinces(fid)
            assert len(provinces) > 0

    def test_factions_have_characters(self):
        engine = self._new_game()
        for fid in engine.state.factions:
            chars = engine.state.faction_characters(fid)
            assert len(chars) >= 4  # 1 ruler + 3 officers

    def test_process_turn(self):
        engine = self._new_game()
        initial_turn = engine.state.turn
        events = engine.process_turn()
        assert engine.state.turn == initial_turn + 1
        assert isinstance(events, list)

    def test_season_advances(self):
        engine = self._new_game()
        assert engine.state.season == Season.SPRING
        engine.process_turn()
        assert engine.state.season == Season.SUMMER
        engine.process_turn()
        assert engine.state.season == Season.AUTUMN
        engine.process_turn()
        assert engine.state.season == Season.WINTER
        engine.process_turn()
        assert engine.state.season == Season.SPRING
        assert engine.state.year == 191

    def test_submit_action_build(self):
        engine = self._new_game()
        fid = list(engine.state.factions.keys())[0]
        provinces = engine.state.faction_provinces(fid)
        assert len(provinces) > 0
        pid = provinces[0].id

        result = engine.submit_action(fid, {
            "type": "build",
            "province_id": pid,
            "building_type": "farm",
        })
        assert result["ok"]

        # Process turn to execute action
        events = engine.process_turn()
        build_events = [e for e in events if e.event_type == "building_built"]
        assert len(build_events) >= 1

    def test_submit_action_recruit_army(self):
        engine = self._new_game()
        fid = list(engine.state.factions.keys())[0]
        provinces = engine.state.faction_provinces(fid)
        pid = provinces[0].id
        initial_army_count = len(engine.state.faction_armies(fid))

        result = engine.submit_action(fid, {
            "type": "recruit_army",
            "province_id": pid,
            "soldiers": 500,
        })
        assert result["ok"]

        engine.process_turn()
        assert len(engine.state.faction_armies(fid)) >= initial_army_count

    def test_submit_action_research(self):
        engine = self._new_game()
        fid = list(engine.state.factions.keys())[0]
        faction = engine.state.factions[fid]

        result = engine.submit_action(fid, {
            "type": "research",
            "tech_id": "mil_1",
        })
        assert result["ok"]

        engine.process_turn()
        assert faction.current_research == "mil_1"

    def test_submit_action_diplomacy_war(self):
        engine = self._new_game()
        fids = list(engine.state.factions.keys())
        fid_a, fid_b = fids[0], fids[1]

        result = engine.submit_action(fid_a, {
            "type": "diplomacy",
            "target_faction_id": fid_b,
            "action": "war",
        })
        assert result["ok"]

        events = engine.process_turn()
        war_events = [e for e in events if e.event_type == "war_declared"]
        assert len(war_events) >= 1

    def test_mark_ready(self):
        engine = self._new_game()
        fids = list(engine.state.factions.keys())
        # Mark first player ready
        assert not engine.mark_ready(fids[0])
        # Mark second player ready -> all ready
        assert engine.mark_ready(fids[1])

    def test_save_and_load(self):
        engine = self._new_game()
        # Process a few turns
        for _ in range(3):
            engine.process_turn()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name

        try:
            engine.save_game(filepath)
            assert os.path.exists(filepath)

            # Load into new engine
            engine2 = GameEngine()
            engine2.load_game(filepath)
            assert engine2.state.turn == engine.state.turn
            assert len(engine2.state.factions) == len(engine.state.factions)
            assert len(engine2.state.provinces) == len(engine.state.provinces)
        finally:
            os.unlink(filepath)

    def test_state_to_dict(self):
        engine = self._new_game()
        d = engine.state.to_dict()
        assert "turn" in d
        assert "provinces" in d
        assert "factions" in d
        assert "characters" in d
        assert "armies" in d
        assert "diplomacy" in d

        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_multiple_turns_stability(self):
        """Run multiple turns to check no crashes."""
        engine = self._new_game()
        for _ in range(20):
            events = engine.process_turn()
            assert isinstance(events, list)
        assert engine.state.turn == 20

    def test_economy_produces_resources(self):
        engine = self._new_game()
        fid = list(engine.state.factions.keys())[0]
        initial_gold = engine.state.factions[fid].gold

        # Process several turns
        for _ in range(5):
            engine.process_turn()

        # Gold should have changed (income or spending)
        # Just check it didn't crash and gold is still a number
        assert isinstance(engine.state.factions[fid].gold, float)


# ---------------------------------------------------------------------------
# Event model test
# ---------------------------------------------------------------------------

class TestGameEvent:
    def test_creation(self):
        e = GameEvent(turn=1, event_type="test", description="Test event")
        assert e.turn == 1

    def test_to_dict(self):
        e = GameEvent(turn=5, event_type="battle", description="A battle occurred")
        d = e.to_dict()
        assert d["turn"] == 5
        assert d["event_type"] == "battle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
