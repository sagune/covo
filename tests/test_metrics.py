import unittest

from covo.metrics import cer, edit_distance


class MetricsTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance(list("abc"), list("adc")), 1)
        self.assertEqual(edit_distance(list("abc"), list("ab")), 1)

    def test_cer(self):
        self.assertAlmostEqual(cer("中界", "中介"), 0.5)
        self.assertEqual(cer("中介", "中介"), 0.0)


if __name__ == "__main__":
    unittest.main()
