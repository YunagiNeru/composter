import unittest
from types import SimpleNamespace

from desktop_pet.sensors import MetricSampler, _safe, _traffic_level


class SensorTests(unittest.TestCase):
    def test_network_rates_only_use_the_explicitly_selected_physical_adapter(self):
        sampler = object.__new__(MetricSampler)
        sampler._last_net = {
            "Ethernet": SimpleNamespace(bytes_recv=1000, bytes_sent=500),
            "VPN": SimpleNamespace(bytes_recv=8000, bytes_sent=4000),
        }
        current = {
            "Ethernet": SimpleNamespace(bytes_recv=3000, bytes_sent=1500),
            "VPN": SimpleNamespace(bytes_recv=18000, bytes_sent=9000),
        }

        self.assertEqual(sampler._network_rates(current, ["Ethernet"], 2),
                         (1000.0, 500.0))

    def test_missing_adapter_keeps_rates_unavailable(self):
        sampler = object.__new__(MetricSampler)
        sampler._last_net = {}
        self.assertEqual(sampler._network_rates({}, ["Ethernet"], 1), (None, None))

    def test_traffic_sensitivity_uses_megabits_per_second_without_fake_values(self):
        self.assertEqual(_traffic_level(250_000, 1), "LOW")
        self.assertEqual(_traffic_level(2_500_000, 1), "MEDIUM")
        self.assertEqual(_traffic_level(30_000_000, 1), "HIGH")
        self.assertIsNone(_safe(None))
        self.assertIsNone(_safe(float("nan")))


if __name__ == "__main__":
    unittest.main()
