import os
import unittest

from app import icd9


class TestIcd9Dictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "icd9_ab.csv")
        os.environ["CENTAUR_ICD9_PATH"] = fixture
        icd9.reset_icd9_cache()
        icd9.load_icd9_dictionary()

    def test_search_by_code_prefix(self):
        results = icd9.search_icd9("401", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0]["code"], "401")

    def test_search_by_label(self):
        results = icd9.search_icd9("hypertension", limit=5)
        codes = [r["code"] for r in results]
        self.assertIn("401", codes)

    def test_search_by_synonym(self):
        results = icd9.search_icd9("high blood pressure", limit=5)
        codes = [r["code"] for r in results]
        self.assertIn("401", codes)

    def test_lookup(self):
        rec = icd9.get_icd9_by_code("428.0")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["code"], "428.0")

    def test_empty_query(self):
        results = icd9.search_icd9("", limit=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
