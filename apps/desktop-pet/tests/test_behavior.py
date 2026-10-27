import unittest

from desktop_pet.behavior import PetMetrics, compose_behavior


def metrics(cpu=20, ram=50, gpu=40, rx="LOW", tx="LOW", disk=50, busy=0):
    return PetMetrics("test", cpu, ram, gpu, rx, tx, 0.0, 0.0, disk, busy)


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.pet = {"personality": "AFFECTIONATE", "mood": "CALM", "bond": 20,
                    "satisfaction": 50}
        self.settings = {"window": {"max_area_percent": 3.0}}

    def test_cpu_high_and_ram_low_keeps_sleepy_expression_but_shuffles(self):
        result = compose_behavior(metrics(cpu=95, ram=15), self.pet, self.settings,
                                  {"gait_style": "RUN", "expression": "ALERT"})
        self.assertEqual(result.gait_style, "DROWSY_SHUFFLE")
        self.assertEqual(result.expression, "SLEEPY")
        self.assertGreater(result.movement_rate, 1.5)
        self.assertIn("SAFETY_CONSTRAINT_APPLIED", result.warnings)

    def test_cpu_low_and_ram_high_is_alert_but_moves_little(self):
        result = compose_behavior(metrics(cpu=5, ram=85), self.pet, self.settings,
                                  {"expression": "ALERT", "gait_style": "BRISK_WALK"})
        self.assertEqual(result.expression, "ALERT")
        self.assertLess(result.movement_rate, 0.2)

    def test_gpu_is_only_display_area_and_unavailable_is_not_zero(self):
        high = compose_behavior(metrics(gpu=100), self.pet, self.settings, {})
        unavailable = compose_behavior(metrics(gpu=None), self.pet, self.settings, {})
        self.assertEqual(high.display_area_percent, 3.0)
        self.assertEqual(unavailable.display_area_percent, 1.875)
        self.assertIsNone(metrics(gpu=None).gpu_percent)

    def test_disk_capacity_changes_body_fullness_but_not_pet_state(self):
        low = compose_behavior(metrics(disk=10), self.pet, self.settings, {})
        high = compose_behavior(metrics(disk=90), self.pet, self.settings, {})
        self.assertEqual(low.body_fullness, 0.1)
        self.assertEqual(high.body_fullness, 0.9)
        self.assertEqual(self.pet["satisfaction"], 50)

    def test_unavailable_metrics_are_explicit_in_model_state(self):
        state = metrics(cpu=None, ram=None, gpu=None, rx=None, tx=None,
                        disk=None, busy=None).as_model_state(self.pet)
        self.assertIn("cpu_usage_percent: UNAVAILABLE", state)
        self.assertIn("gpu_usage_percent: UNAVAILABLE", state)
        self.assertIn("network_rx_level: UNAVAILABLE", state)
        self.assertIn("wakefulness: UNAVAILABLE", state)

    def test_high_transmit_keeps_the_meow_style_aligned_with_wakefulness(self):
        sleepy = compose_behavior(metrics(cpu=10, ram=15, tx="HIGH"), self.pet,
                                  self.settings, {"meow_style": "NONE"})
        alert = compose_behavior(metrics(cpu=10, ram=85, tx="HIGH"), self.pet,
                                 self.settings, {"meow_style": "SLEEPY_MEOW"})
        self.assertEqual(sleepy.meow_style, "SLEEPY_MEOW")
        self.assertEqual(alert.meow_style, "SHORT_MEOW")

    def test_high_receive_moves_ears_and_high_disk_activity_selects_dig(self):
        low_rx = compose_behavior(metrics(rx="LOW"), self.pet, self.settings, {})
        high_rx = compose_behavior(metrics(rx="HIGH"), self.pet, self.settings, {})
        digging = compose_behavior(metrics(busy=80), self.pet, self.settings, {})
        self.assertGreater(high_rx.ear_rate, low_rx.ear_rate)
        self.assertEqual(digging.animation, "dig")

    def test_personality_and_bond_are_present_in_the_structured_npu_state(self):
        first = metrics().as_model_state({"personality": "AFFECTIONATE", "bond": 10})
        second = metrics().as_model_state({"personality": "CAUTIOUS", "bond": 80})
        self.assertNotEqual(first, second)
        self.assertIn("personality: CAUTIOUS", second)
        self.assertIn("bond_level: 80", second)


if __name__ == "__main__":
    unittest.main()
