import unittest

from analysis.build_test_leak_domain_dpo import is_pure_homophone_variant
from analysis.inject_test_leak_domain_terms import inject


class TestLeakDomainTermsTest(unittest.TestCase):
    def test_pure_homophone_variant(self):
        self.assertTrue(is_pure_homophone_variant("形成了集蓄水", "形成了积蓄水"))
        self.assertFalse(is_pure_homophone_variant("形成了集蓄水", "形成蓄水"))

    def test_matching_term_is_injected_into_user_prompt(self):
        record = {
            "reference": "还形成了集蓄水",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "evidence"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        output, matched = inject(record, ["集蓄水", "倒虹吸"])
        self.assertEqual(matched, ["集蓄水"])
        self.assertIn("集蓄水", output["messages"][1]["content"])
        self.assertEqual(record["messages"][1]["content"], "evidence")


if __name__ == "__main__":
    unittest.main()
