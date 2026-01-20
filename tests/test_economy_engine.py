"""Tests for economy_engine.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.economy_engine import (
    EconomyEngine, BuyType, Loadout, TeamEconomy
)
from app.services.weapon_system import WeaponDatabase, WeaponCategory


class TestBuyTypeClassification:
    """Tests for buy type classification."""

    def test_pistol_round(self):
        """Test pistol round classification."""
        buy_type = EconomyEngine.classify_buy_type(
            team_loadout_value=4000,
            round_num=0,
            is_pistol_round=True
        )
        assert buy_type == BuyType.PISTOL

    def test_eco_classification(self):
        """Test eco round classification."""
        # Low loadout value = eco
        buy_type = EconomyEngine.classify_buy_type(
            team_loadout_value=5000,  # 1000 per player
            round_num=5,
            is_pistol_round=False
        )
        assert buy_type == BuyType.ECO

    def test_full_buy_classification(self):
        """Test full buy classification."""
        buy_type = EconomyEngine.classify_buy_type(
            team_loadout_value=25000,  # 5000 per player
            round_num=10,
            is_pistol_round=False
        )
        assert buy_type == BuyType.FULL


class TestLoadoutGeneration:
    """Tests for loadout generation."""

    def test_pistol_round_loadout(self):
        """Test pistol round generates appropriate weapons."""
        economy = TeamEconomy(credits=[800] * 5)
        loadouts = EconomyEngine.generate_team_loadout(
            team_economy=economy,
            round_num=0,
            side='attack',
            forced_buy_type=BuyType.PISTOL
        )

        assert len(loadouts) == 5
        for loadout in loadouts:
            # Should only have sidearms on pistol
            assert loadout.weapon.category == WeaponCategory.SIDEARM
            # Total value should be <= 800
            assert loadout.total_value <= 800

    def test_full_buy_loadout(self):
        """Test full buy generates rifles."""
        economy = TeamEconomy(credits=[5000] * 5)
        loadouts = EconomyEngine.generate_team_loadout(
            team_economy=economy,
            round_num=10,
            side='attack',
            forced_buy_type=BuyType.FULL
        )

        assert len(loadouts) == 5
        # Most players should have rifles on full buy
        rifle_count = sum(
            1 for l in loadouts
            if l.weapon.category in [WeaponCategory.RIFLE, WeaponCategory.SNIPER]
        )
        assert rifle_count >= 3

        # All should have heavy armor
        for loadout in loadouts:
            assert loadout.armor.shield_value == 50

    def test_eco_round_saves_money(self):
        """Test eco round uses cheap weapons."""
        economy = TeamEconomy(credits=[2000] * 5)
        loadouts = EconomyEngine.generate_team_loadout(
            team_economy=economy,
            round_num=5,
            side='attack',
            forced_buy_type=BuyType.ECO
        )

        for loadout in loadouts:
            # Should be cheap weapons
            assert loadout.weapon.cost < 1500


class TestEconomyCalculations:
    """Tests for economy income calculations."""

    def test_win_bonus(self):
        """Test win round income."""
        income = EconomyEngine.calculate_round_income(
            won=True,
            loss_streak=0,
            kills=0
        )
        assert income == 3000

    def test_loss_bonus_progression(self):
        """Test loss streak bonus increases."""
        # First loss
        income1 = EconomyEngine.calculate_round_income(
            won=False, loss_streak=0, kills=0
        )
        assert income1 == 1900

        # Second loss
        income2 = EconomyEngine.calculate_round_income(
            won=False, loss_streak=1, kills=0
        )
        assert income2 == 2400

        # Third loss
        income3 = EconomyEngine.calculate_round_income(
            won=False, loss_streak=2, kills=0
        )
        assert income3 == 2900

    def test_kill_bonus(self):
        """Test kill bonus adds correctly."""
        income = EconomyEngine.calculate_round_income(
            won=True,
            loss_streak=0,
            kills=3
        )
        assert income == 3000 + (3 * 200)

    def test_spike_plant_bonus(self):
        """Test spike plant bonus for attackers."""
        income = EconomyEngine.calculate_round_income(
            won=False,
            loss_streak=0,
            kills=0,
            spike_planted=True,
            is_attacker=True
        )
        assert income == 1900 + 300


class TestTeamEconomy:
    """Tests for TeamEconomy class."""

    def test_initial_state(self):
        """Test initial economy state."""
        economy = TeamEconomy()
        assert economy.total_credits == 4000  # 800 * 5
        assert economy.average_credits == 800
        assert economy.loss_streak == 0

    def test_update_after_win(self):
        """Test economy update after winning."""
        economy = TeamEconomy(credits=[800] * 5, loss_streak=2)
        new_economy = EconomyEngine.update_team_economy(
            team_economy=economy,
            won=True,
            player_kills=[1, 0, 2, 0, 1],
            spike_planted=False,
            is_attacker=True
        )

        # Loss streak should reset
        assert new_economy.loss_streak == 0
        # All players should have more money
        for old, new in zip(economy.credits, new_economy.credits):
            assert new > old

    def test_credit_cap(self):
        """Test that credits are capped at 9000."""
        economy = TeamEconomy(credits=[8500] * 5)
        new_economy = EconomyEngine.update_team_economy(
            team_economy=economy,
            won=True,
            player_kills=[5, 5, 5, 5, 5],  # Lots of kills
            spike_planted=True,
            is_attacker=True
        )

        for credits in new_economy.credits:
            assert credits <= 9000


class TestBuyDecisions:
    """Tests for buy decision logic."""

    def test_should_eco_low_money(self):
        """Test eco recommendation with low money."""
        economy = TeamEconomy(credits=[1500] * 5)
        should_eco = EconomyEngine.should_eco(economy, round_num=5)
        assert should_eco is True

    def test_should_not_eco_full_money(self):
        """Test no eco with full money."""
        economy = TeamEconomy(credits=[5000] * 5)
        should_eco = EconomyEngine.should_eco(economy, round_num=5)
        assert should_eco is False

    def test_recommended_buy_type(self):
        """Test recommended buy type calculation."""
        # Rich economy = full buy
        rich = TeamEconomy(credits=[5000] * 5)
        assert EconomyEngine.get_recommended_buy_type(rich, 5) == BuyType.FULL

        # Poor economy = eco
        poor = TeamEconomy(credits=[1500] * 5)
        assert EconomyEngine.get_recommended_buy_type(poor, 5) == BuyType.ECO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
