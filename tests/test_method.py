import tempfile
from pathlib import Path
import unittest
from project_demo.pipeline import bounded_edit, metrics, optimize, fixture_corrector, load_splits

class VoiceMemoryTests(unittest.TestCase):
    def test_gate_accepts_improvement_rejects_harmful_edit(self):
        splits = load_splits({}, 'fixture')
        edits = [{'operation':'add','text':'REPLACE new mo => NeMo'},
                 {'operation':'add','text':'REPLACE right => write'}]
        with tempfile.TemporaryDirectory() as folder:
            memory, history = optimize(splits['train'], splits['selection'], fixture_corrector,
                                       lambda r,m,i: edits[i], 2, Path(folder))
        self.assertEqual([h['accepted'] for h in history], [True,False])
        self.assertEqual(len(memory),1)

    def test_zero_headroom_is_not_division_by_zero(self):
        r=[{'reference':'already correct','hypotheses':['already correct']}]
        m=metrics(r,['already correct'])
        self.assertIsNone(m['recoverable_headroom_fraction'])
        self.assertIsNone(m['harmful_edit_rate_among_changed'])

    def test_memory_edits_are_bounded(self):
        with self.assertRaises(ValueError): bounded_edit([], {'operation':'delete','index':0})
        with self.assertRaises(ValueError): bounded_edit([], {'operation':'add','text':'字'*600})
        self.assertEqual(bounded_edit(['a'], {'operation':'replace','index':0,'text':'b'}), ['b'])

    def test_corpus_weighted_wer(self):
        rows=[{'reference':'a b c','hypotheses':['a b c']},{'reference':'d','hypotheses':['x']}]
        self.assertEqual(metrics(rows,['a b c','x'])['wer'],0.25)

if __name__=='__main__': unittest.main()
