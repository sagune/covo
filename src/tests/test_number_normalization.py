import unittest

from analysis.evaluate_filler_normalized_cer import normalize_numbers


class NumberNormalizationTest(unittest.TestCase):
    def test_common_arabic_and_chinese_forms_are_equivalent(self):
        pairs = [
            ("10到20吨水", "十到二十吨水"),
            ("6400亿立方米", "六千四百亿立方米"),
            ("2025年", "二零二五年"),
            ("15.5吨水", "十五点五吨水"),
            ("17万亩", "十七万亩"),
            ("70%", "百分之七十"),
        ]
        for arabic, chinese in pairs:
            with self.subTest(arabic=arabic, chinese=chinese):
                self.assertEqual(normalize_numbers(arabic), normalize_numbers(chinese))

    def test_digit_grouping_and_decimal_zero_are_normalized(self):
        self.assertEqual(normalize_numbers("6,400.0亿立方米"), "6400亿立方米")


if __name__ == "__main__":
    unittest.main()
