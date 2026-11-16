import unittest
from project_demo.training import assert_prompt_prefix
class TrainingContractTests(unittest.TestCase):
    def test_prompt_mask_never_swallows_changed_boundary(self):
        self.assertEqual(assert_prompt_prefix([1,2],[1,2,3]),2)
        with self.assertRaises(ValueError): assert_prompt_prefix([1,2],[1,9,3])
        with self.assertRaises(ValueError): assert_prompt_prefix([1,2],[1,2])
