import math
import unittest

from neuralzip.mode_selection import (
    select_fastest_within_ratio,
    select_packed_time_budget,
)


class TimeAwareModeSelectionTests(unittest.TestCase):
    @staticmethod
    def _bytes(mode):
        return mode["bytes"]

    @staticmethod
    def _seconds(mode):
        return mode["seconds"]

    def test_faster_packed_mode_can_replace_ratio_best_local(self):
        local = {"kind": "local_huff0", "bytes": 100, "seconds": 0.10}
        shared = {"kind": "shared_huff0", "bytes": 102, "seconds": 0.04}
        packed = {
            "kind": "packed_huff0",
            "bytes": 103,
            "seconds": 0.02,
        }

        result = select_fastest_within_ratio(
            [local, shared, packed],
            estimated_bytes=self._bytes,
            estimated_seconds=self._seconds,
            ratio_tolerance=0.03,
        )

        self.assertIs(result.selected, packed)
        self.assertEqual(result.metrics["ratio_best_kind"], "local_huff0")
        self.assertEqual(result.metrics["selected_kind"], "packed_huff0")
        self.assertAlmostEqual(result.metrics["selected_relative_byte_overhead"], 0.03)
        self.assertTrue(result.metrics["avoided_local_huff0"])

    def test_zero_ratio_tolerance_keeps_smallest_representation(self):
        local = {"kind": "local_huff0", "bytes": 100, "seconds": 0.10}
        shared = {"kind": "shared_huff0", "bytes": 101, "seconds": 0.01}

        result = select_fastest_within_ratio(
            [shared, local],
            estimated_bytes=self._bytes,
            estimated_seconds=self._seconds,
            ratio_tolerance=0.0,
        )

        self.assertIs(result.selected, local)
        self.assertEqual(result.metrics["eligible_candidate_count"], 1)
        self.assertFalse(result.metrics["avoided_local_huff0"])

    def test_faster_mode_outside_ratio_limit_is_ineligible(self):
        local = {"kind": "local_huff0", "bytes": 100, "seconds": 0.10}
        raw = {"kind": "raw", "bytes": 111, "seconds": 0.0}

        result = select_fastest_within_ratio(
            [raw, local],
            estimated_bytes=self._bytes,
            estimated_seconds=self._seconds,
            ratio_tolerance=0.10,
        )

        self.assertIs(result.selected, local)
        self.assertEqual(result.metrics["eligible_candidate_count"], 1)

    def test_selection_is_stable_across_candidate_order(self):
        local = {"kind": "local_huff0", "bytes": 100, "seconds": 0.10}
        shared = {"kind": "shared_huff0", "bytes": 100, "seconds": 0.10}

        forward = select_fastest_within_ratio(
            [local, shared],
            estimated_bytes=self._bytes,
            estimated_seconds=self._seconds,
        )
        reverse = select_fastest_within_ratio(
            [shared, local],
            estimated_bytes=self._bytes,
            estimated_seconds=self._seconds,
        )

        self.assertEqual(forward.selected["kind"], "shared_huff0")
        self.assertEqual(reverse.selected["kind"], "shared_huff0")

    def test_invalid_ratio_inputs_are_rejected(self):
        mode = {"kind": "local_huff0", "bytes": 1, "seconds": 1}
        with self.assertRaises(ValueError):
            select_fastest_within_ratio(
                [],
                estimated_bytes=self._bytes,
                estimated_seconds=self._seconds,
            )
        with self.assertRaises(ValueError):
            select_fastest_within_ratio(
                [mode],
                estimated_bytes=self._bytes,
                estimated_seconds=self._seconds,
                ratio_tolerance=-0.1,
            )
        with self.assertRaises(ValueError):
            select_fastest_within_ratio(
                [{"kind": "bad", "bytes": math.inf, "seconds": 0}],
                estimated_bytes=self._bytes,
                estimated_seconds=self._seconds,
            )


class PackedTimeBudgetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.evaluations = [
            {
                "key": "zero",
                "base_seconds": 2.0,
                "opportunity": {"saved_bytes": 50, "extra_seconds": 0.0},
            },
            {
                "key": "efficient",
                "base_seconds": 3.0,
                "opportunity": {"saved_bytes": 100, "extra_seconds": 0.5},
            },
            {
                "key": "large",
                "base_seconds": 4.0,
                "opportunity": {"saved_bytes": 300, "extra_seconds": 2.0},
            },
            {
                "key": "fits-last",
                "base_seconds": 1.0,
                "opportunity": {"saved_bytes": 40, "extra_seconds": 0.4},
            },
        ]

    def test_global_budget_uses_historical_savings_per_second_order(self):
        result = select_packed_time_budget(
            self.evaluations,
            packed_time_budget_ratio=0.10,
        )

        self.assertEqual(
            result.selected_keys,
            frozenset({"zero", "efficient", "fits-last"}),
        )
        self.assertEqual(result.metrics["selection_strategy"], "time-budget")
        self.assertEqual(result.metrics["base_estimated_seconds"], 10.0)
        self.assertEqual(result.metrics["packed_time_budget_seconds"], 1.0)
        self.assertAlmostEqual(result.metrics["packed_extra_seconds"], 0.9)
        self.assertEqual(result.metrics["packed_estimated_bytes_saved"], 190.0)
        self.assertEqual(result.metrics["packed_candidates"], 4)
        self.assertEqual(result.metrics["packed_opportunities"], 4)
        self.assertEqual(result.metrics["packed_selected"], 3)
        self.assertEqual(result.metrics["packed_rejected_by_time_budget"], 1)
        self.assertEqual(result.metrics["packed_zero_extra_selected"], 1)

    def test_zero_budget_disables_even_zero_extra_opportunities(self):
        result = select_packed_time_budget(
            self.evaluations,
            packed_time_budget_ratio=0.0,
        )

        self.assertEqual(result.selected_keys, frozenset())
        self.assertEqual(result.metrics["packed_selected"], 0)
        self.assertEqual(result.metrics["packed_extra_seconds"], 0.0)
        self.assertEqual(result.metrics["packed_estimated_bytes_saved"], 0.0)

    def test_non_saving_opportunities_are_not_candidates(self):
        result = select_packed_time_budget(
            [
                {"key": "none", "base_seconds": 1.0, "opportunity": None},
                {
                    "key": "worse",
                    "base_seconds": 1.0,
                    "opportunity": {"saved_bytes": 0, "extra_seconds": 0},
                },
            ],
            packed_time_budget_ratio=1.0,
        )

        self.assertEqual(result.metrics["packed_candidates"], 0)
        self.assertEqual(result.metrics["packed_opportunities"], 0)
        self.assertEqual(result.selected_keys, frozenset())

    def test_invalid_time_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            select_packed_time_budget(
                [],
                packed_time_budget_ratio=-1,
            )
        with self.assertRaises(ValueError):
            select_packed_time_budget(
                [{"key": "bad", "base_seconds": math.nan}],
                packed_time_budget_ratio=1,
            )
        with self.assertRaises(ValueError):
            select_packed_time_budget(
                [
                    {
                        "key": "bad-extra",
                        "base_seconds": 1,
                        "opportunity": {
                            "saved_bytes": 1,
                            "extra_seconds": math.nan,
                        },
                    }
                ],
                packed_time_budget_ratio=1,
            )


if __name__ == "__main__":
    unittest.main()
