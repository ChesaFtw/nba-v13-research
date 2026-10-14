import unittest
from scripts.build_advanced_features import join_advanced

class AdvancedFeatureTests(unittest.TestCase):
    def test_prior_only_and_same_day(self):
        p=lambda gid,date:{"player":{"id":7},"game":{"id":gid,"date":date}}
        a=lambda gid,pace:{"player":{"id":7},"game":{"id":gid},"period":0,"pace":pace}
        rows=join_advanced([p(1,"2025-10-01"),p(2,"2025-10-02"),p(3,"2025-10-02"),p(4,"2025-10-03")],[a(1,90),a(2,110),a(3,120),a(4,130)])
        lookup={r["game_id"]:r for r in rows}
        self.assertIsNone(lookup[1]["advanced_prior_10"]["pace"])
        self.assertEqual(lookup[2]["advanced_prior_10"]["pace"],90)
        self.assertEqual(lookup[3]["advanced_prior_10"]["pace"],90)
        self.assertEqual(lookup[4]["advanced_prior_10"]["pace"],(90+110+120)/3)
