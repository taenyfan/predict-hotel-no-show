# Standard library imports
# (none required)

# Related third-party imports
import numpy as np
import pandas as pd
import pytest

# Local application imports
from data_preparation import (
    DataPreparation,
    month_to_num,
    cyclical_encode,
    cyclical_names,
)


# --- month_to_num --------------------------------------------------------------

class TestMonthToNum:
    """Tests for mapping month names to numbers 1-12."""

    def test_january_is_one(self):
        assert month_to_num(pd.Series(["January"]))[0] == 1

    def test_december_is_twelve(self):
        assert month_to_num(pd.Series(["December"]))[0] == 12

    def test_all_months_ordered(self):
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        result = month_to_num(pd.Series(months))
        assert list(result) == list(range(1, 13))

    def test_invalid_month_is_nan(self):
        assert pd.isna(month_to_num(pd.Series(["Smarch"]))[0])


# --- _convert_price_to_sgd -----------------------------------------------------

class TestConvertPriceToSgd:
    """Tests for converting 'SGD$/USD$' price strings to numeric SGD."""

    def test_sgd_unchanged(self):
        prices = pd.Series(["SGD$ 100", "SGD$ 250.5"])
        result = DataPreparation._convert_price_to_sgd(prices)
        assert result[0] == 100.0
        assert result[1] == 250.5

    def test_usd_scaled(self):
        prices = pd.Series(["USD$ 100"])
        result = DataPreparation._convert_price_to_sgd(prices)
        assert result[0] == pytest.approx(137.0)  # 100 * 1.37

    def test_custom_rate(self):
        prices = pd.Series(["USD$ 100"])
        result = DataPreparation._convert_price_to_sgd(prices, usd_to_sgd=1.5)
        assert result[0] == pytest.approx(150.0)

    def test_mixed_currencies(self):
        prices = pd.Series(["SGD$ 200", "USD$ 200"])
        result = DataPreparation._convert_price_to_sgd(prices)
        assert result[0] == 200.0                 # SGD unchanged
        assert result[1] == pytest.approx(274.0)  # USD scaled

    def test_unparseable_becomes_nan(self):
        result = DataPreparation._convert_price_to_sgd(pd.Series(["SGD$ abc"]))
        assert pd.isna(result[0])


# --- _compute_lead_time --------------------------------------------------------

class TestComputeLeadTime:
    """Tests for the months-between-booking-and-arrival feature."""

    def test_same_month(self):
        df = pd.DataFrame({"booking_month": ["June"], "arrival_month": ["June"]})
        assert DataPreparation._compute_lead_time(df)[0] == 0

    def test_normal_forward(self):
        df = pd.DataFrame({"booking_month": ["March"], "arrival_month": ["May"]})
        assert DataPreparation._compute_lead_time(df)[0] == 2

    def test_year_wrap(self):
        # booked December, arrive February -> 2 (not -10)
        df = pd.DataFrame({"booking_month": ["December"], "arrival_month": ["February"]})
        assert DataPreparation._compute_lead_time(df)[0] == 2

    def test_max_lead(self):
        # booked January, arrive December -> 11
        df = pd.DataFrame({"booking_month": ["January"], "arrival_month": ["December"]})
        assert DataPreparation._compute_lead_time(df)[0] == 11

    def test_wrap_boundary(self):
        # booked December, arrive January -> 1
        df = pd.DataFrame({"booking_month": ["December"], "arrival_month": ["January"]})
        assert DataPreparation._compute_lead_time(df)[0] == 1

    def test_output_range(self):
        # every combination of months must yield a value in 0-11
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        df = pd.DataFrame({
            "booking_month": months * 12,
            "arrival_month": [m for m in months for _ in range(12)],
        })
        result = DataPreparation._compute_lead_time(df)
        assert result.min() >= 0
        assert result.max() <= 11


# --- _impute_price -------------------------------------------------------------

class TestImputePrice:
    """Tests for filling missing price with the (branch, room) median."""

    def test_fills_with_group_median(self):
        df = pd.DataFrame({
            "branch": ["Orchard", "Orchard", "Orchard"],
            "room":   ["King", "King", "King"],
            "price":  [100.0, 200.0, np.nan],
        })
        result = DataPreparation._impute_price(df)
        assert result["price"][2] == 150.0  # median of [100, 200]

    def test_no_nulls_remain(self):
        df = pd.DataFrame({
            "branch": ["Orchard", "Orchard", "Changi", "Changi"],
            "room":   ["King", "King", "Queen", "Queen"],
            "price":  [100.0, np.nan, 80.0, np.nan],
        })
        result = DataPreparation._impute_price(df)
        assert result["price"].isna().sum() == 0

    def test_existing_values_unchanged(self):
        df = pd.DataFrame({
            "branch": ["Orchard", "Orchard"],
            "room":   ["King", "King"],
            "price":  [100.0, 200.0],
        })
        result = DataPreparation._impute_price(df)
        assert result["price"][0] == 100.0
        assert result["price"][1] == 200.0

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "branch": ["Orchard", "Orchard"],
            "room":   ["King", "King"],
            "price":  [100.0, np.nan],
        })
        _ = DataPreparation._impute_price(df)
        assert pd.isna(df["price"][1])  # original still has the NaN


# --- _impute_room --------------------------------------------------------------

class TestImputeRoom:
    """Tests for imputing room from branch + price band."""

    def _make_df(self):
        # Two clearly separated price bands per room within one branch,
        # plus two missing-room rows: one clearly in a band, one out of range.
        return pd.DataFrame({
            "branch": ["Orchard"] * 6 + ["Orchard", "Orchard"],
            "room":   ["Single", "Single", "Single",
                       "King", "King", "King",
                       np.nan, np.nan],
            "price":  [100.0, 110.0, 120.0,
                       500.0, 510.0, 520.0,
                       105.0,        # clearly in the Single band
                       9999.0],      # outside all bands -> Unknown
        })

    def test_confident_match_assigned(self):
        result = DataPreparation._impute_room(self._make_df())
        assert result["room"][6] == "Single"  # price 105 -> Single band

    def test_out_of_band_is_unknown(self):
        result = DataPreparation._impute_room(self._make_df())
        assert result["room"][7] == "Unknown"  # price 9999 -> no band

    def test_no_nulls_remain(self):
        result = DataPreparation._impute_room(self._make_df())
        assert result["room"].isna().sum() == 0

    def test_known_rooms_unchanged(self):
        result = DataPreparation._impute_room(self._make_df())
        assert result["room"][0] == "Single"
        assert result["room"][3] == "King"


# --- cyclical_encode -----------------------------------------------------------

class TestCyclicalEncode:
    """Tests for sin/cos encoding of month columns."""

    def test_january_values(self):
        result = cyclical_encode(pd.DataFrame({"arrival_month": ["January"]}))
        # month 1 -> angle 2pi/12 = pi/6 ; sin=0.5, cos=0.866
        assert result["arrival_month_sin"][0] == pytest.approx(0.5, abs=1e-6)
        assert result["arrival_month_cos"][0] == pytest.approx(0.8660254, abs=1e-6)

    def test_output_columns(self):
        X = pd.DataFrame({"arrival_month": ["June"], "checkout_month": ["July"]})
        result = cyclical_encode(X)
        assert list(result.columns) == [
            "arrival_month_sin", "arrival_month_cos",
            "checkout_month_sin", "checkout_month_cos",
        ]

    def test_values_in_unit_range(self):
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        result = cyclical_encode(pd.DataFrame({"arrival_month": months}))
        assert result["arrival_month_sin"].between(-1, 1).all()
        assert result["arrival_month_cos"].between(-1, 1).all()

    def test_december_adjacent_to_january(self):
        # cyclical: Dec and Jan should be closer than Jan and Jun
        r = cyclical_encode(pd.DataFrame({"arrival_month": ["December", "January"]}))
        dec = np.array([r["arrival_month_sin"][0], r["arrival_month_cos"][0]])
        jan = np.array([r["arrival_month_sin"][1], r["arrival_month_cos"][1]])
        dist_dec_jan = np.linalg.norm(dec - jan)

        r2 = cyclical_encode(pd.DataFrame({"arrival_month": ["January", "June"]}))
        jan2 = np.array([r2["arrival_month_sin"][0], r2["arrival_month_cos"][0]])
        jun = np.array([r2["arrival_month_sin"][1], r2["arrival_month_cos"][1]])
        dist_jan_jun = np.linalg.norm(jan2 - jun)

        assert dist_dec_jan < dist_jan_jun

    def test_preserves_index(self):
        result = cyclical_encode(pd.DataFrame({"arrival_month": ["June"]}, index=[42]))
        assert list(result.index) == [42]


# --- cyclical_names ------------------------------------------------------------

class TestCyclicalNames:
    """Tests that output names match the encoded column order."""

    def test_names_match_encode_order(self):
        input_features = ["arrival_month", "checkout_month"]
        names = cyclical_names(None, input_features)
        X = pd.DataFrame({"arrival_month": ["June"], "checkout_month": ["July"]})
        encoded_cols = list(cyclical_encode(X).columns)
        assert list(names) == encoded_cols

    def test_two_names_per_column(self):
        names = cyclical_names(None, ["arrival_month"])
        assert list(names) == ["arrival_month_sin", "arrival_month_cos"]
