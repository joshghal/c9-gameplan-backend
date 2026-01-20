"""Tests for weapon_system.py"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.weapon_system import (
    WeaponDatabase, WeaponStats, ArmorStats, WeaponCategory,
    meters_from_normalized
)


class TestWeaponDatabase:
    """Tests for WeaponDatabase."""

    def test_all_weapons_exist(self):
        """Test that all expected weapons are in the database."""
        expected_weapons = [
            'classic', 'shorty', 'frenzy', 'ghost', 'sheriff',
            'stinger', 'spectre', 'bucky', 'judge',
            'bulldog', 'guardian', 'phantom', 'vandal',
            'marshal', 'outlaw', 'operator',
            'ares', 'odin'
        ]
        for weapon_id in expected_weapons:
            assert weapon_id in WeaponDatabase.WEAPONS, f"Missing weapon: {weapon_id}"

    def test_weapon_stats_valid(self):
        """Test that weapon stats are valid."""
        for weapon_id, weapon in WeaponDatabase.WEAPONS.items():
            assert weapon.cost >= 0, f"{weapon_id} has negative cost"
            assert weapon.head_damage > 0, f"{weapon_id} has no head damage"
            assert weapon.body_damage > 0, f"{weapon_id} has no body damage"
            assert weapon.fire_rate > 0, f"{weapon_id} has no fire rate"
            assert 0 <= weapon.first_shot_accuracy <= 1, f"{weapon_id} invalid accuracy"

    def test_armor_types(self):
        """Test that all armor types exist."""
        assert 'none' in WeaponDatabase.ARMOR
        assert 'light' in WeaponDatabase.ARMOR
        assert 'heavy' in WeaponDatabase.ARMOR

    def test_armor_values(self):
        """Test armor shield values."""
        assert WeaponDatabase.ARMOR['none'].shield_value == 0
        assert WeaponDatabase.ARMOR['light'].shield_value == 25
        assert WeaponDatabase.ARMOR['heavy'].shield_value == 50


class TestDamageCalculation:
    """Tests for damage calculations."""

    def test_headshot_damage(self):
        """Test headshot damage calculation."""
        vandal = WeaponDatabase.WEAPONS['vandal']
        armor = WeaponDatabase.ARMOR['none']

        health_dmg, shield_dmg, new_shield = WeaponDatabase.calculate_damage(
            weapon=vandal,
            distance_meters=20.0,
            hit_region='head',
            armor=armor,
            current_health=100,
            current_shield=0
        )

        # Vandal headshot should be 160 damage (one-shot kill)
        assert health_dmg == 160

    def test_body_damage(self):
        """Test body damage calculation."""
        vandal = WeaponDatabase.WEAPONS['vandal']
        armor = WeaponDatabase.ARMOR['none']

        health_dmg, shield_dmg, new_shield = WeaponDatabase.calculate_damage(
            weapon=vandal,
            distance_meters=20.0,
            hit_region='body',
            armor=armor,
            current_health=100,
            current_shield=0
        )

        # Vandal body should be 40 damage
        assert health_dmg == 40

    def test_armor_reduction(self):
        """Test armor damage reduction."""
        vandal = WeaponDatabase.WEAPONS['vandal']
        armor = WeaponDatabase.ARMOR['heavy']

        health_dmg, shield_dmg, new_shield = WeaponDatabase.calculate_damage(
            weapon=vandal,
            distance_meters=20.0,
            hit_region='body',
            armor=armor,
            current_health=100,
            current_shield=50
        )

        # With armor: 40 * 0.66 = 26 damage
        # All goes to shield first
        assert shield_dmg == 26
        assert health_dmg == 0
        assert new_shield == 24

    def test_phantom_falloff(self):
        """Test Phantom damage falloff at range."""
        phantom = WeaponDatabase.WEAPONS['phantom']
        armor = WeaponDatabase.ARMOR['none']

        # Close range (within 15m)
        close_dmg, _, _ = WeaponDatabase.calculate_damage(
            weapon=phantom,
            distance_meters=10.0,
            hit_region='head',
            armor=armor,
            current_health=100,
            current_shield=0
        )

        # Long range (beyond 30m)
        far_dmg, _, _ = WeaponDatabase.calculate_damage(
            weapon=phantom,
            distance_meters=40.0,
            hit_region='head',
            armor=armor,
            current_health=100,
            current_shield=0
        )

        # Close range should do more damage
        assert close_dmg > far_dmg


class TestKillProbability:
    """Tests for kill probability calculations."""

    def test_equal_weapons_equal_odds(self):
        """Test that equal weapons give roughly equal odds."""
        vandal = WeaponDatabase.WEAPONS['vandal']
        armor = WeaponDatabase.ARMOR['heavy']

        atk_win, def_win = WeaponDatabase.calculate_kill_probability(
            attacker_weapon=vandal,
            defender_weapon=vandal,
            distance_meters=20.0,
            attacker_headshot_rate=0.25,
            defender_headshot_rate=0.25,
            attacker_armor=armor,
            defender_armor=armor
        )

        # Should be roughly 50/50
        assert 0.4 < atk_win < 0.6
        assert 0.4 < def_win < 0.6

    def test_rifle_vs_pistol(self):
        """Test that rifle has advantage over pistol."""
        vandal = WeaponDatabase.WEAPONS['vandal']
        classic = WeaponDatabase.WEAPONS['classic']
        armor = WeaponDatabase.ARMOR['heavy']

        atk_win, def_win = WeaponDatabase.calculate_kill_probability(
            attacker_weapon=vandal,
            defender_weapon=classic,
            distance_meters=25.0,
            attacker_headshot_rate=0.25,
            defender_headshot_rate=0.25,
            attacker_armor=armor,
            defender_armor=WeaponDatabase.ARMOR['none']
        )

        # Vandal should have significant advantage
        assert atk_win > 0.6

    def test_operator_long_range(self):
        """Test that Operator dominates at long range."""
        operator = WeaponDatabase.WEAPONS['operator']
        vandal = WeaponDatabase.WEAPONS['vandal']
        armor = WeaponDatabase.ARMOR['heavy']

        atk_win, def_win = WeaponDatabase.calculate_kill_probability(
            attacker_weapon=operator,
            defender_weapon=vandal,
            distance_meters=40.0,
            attacker_headshot_rate=0.8,  # OP players hit body
            defender_headshot_rate=0.25,
            attacker_armor=armor,
            defender_armor=armor
        )

        # Operator should have strong advantage at range
        assert atk_win > 0.55


class TestHelpers:
    """Tests for helper functions."""

    def test_get_weapon(self):
        """Test weapon lookup by name."""
        weapon = WeaponDatabase.get_weapon('Vandal')
        assert weapon is not None
        assert weapon.name == 'Vandal'

        weapon = WeaponDatabase.get_weapon('vandal')
        assert weapon is not None

        weapon = WeaponDatabase.get_weapon('nonexistent')
        assert weapon is None

    def test_get_armor(self):
        """Test armor lookup by name."""
        armor = WeaponDatabase.get_armor('Heavy Armor')
        assert armor is not None
        assert armor.shield_value == 50

        armor = WeaponDatabase.get_armor('light armor')
        assert armor is not None

    def test_meters_from_normalized(self):
        """Test distance conversion."""
        # 10% of 100m map = 10m
        assert meters_from_normalized(0.1) == 10.0

        # 50% of 100m map = 50m
        assert meters_from_normalized(0.5) == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
