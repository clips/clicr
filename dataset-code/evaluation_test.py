import unittest

from describe_data import *
from evaluate import evaluate
from util import load_json


class EvaluationTest(unittest.TestCase):
    def test_identity(self):
        print("Preparing data...")
        dataset_file = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_json_concept_annotated/dev1.0.json"
        dataset = load_json(dataset_file)
        data = dataset[DATA_KEY]
        fake_predictions = {}
        for datum in data:
            for qa in datum[DOC_KEY][QAS_KEY]:
                for ans in qa[ANS_KEY]:
                    if ans[ORIG_KEY] == "dataset":
                        fake_predictions[qa[ID_KEY]] = ans[TXT_KEY]
        print("Evaluating...")
        score = evaluate(dataset, fake_predictions, extended=True, embeddings_file="/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/embeddings/c47cfee6-3fc4-11e7-b5a2-4ccc6a436494/embeddings", downcase=True)

        print("Testing {}".format(score.keys()))
        self.assertEqual(score["f1"], 100)
        self.assertEqual(score["exact_match"], 100)
        self.assertAlmostEqual(score["Bleu_1"], 1)
        self.assertAlmostEqual(score["Bleu_2"], 1)
        self.assertAlmostEqual(score["Bleu_3"], 1)
        self.assertAlmostEqual(score["Bleu_4"], 1)
        self.assertAlmostEqual(score["ROUGE_L"], 1)
        self.assertAlmostEqual(score["emb-average"], 1)
        self.assertAlmostEqual(score["emb-greedy"], 1)
        self.assertAlmostEqual(score["emb-extrema"], 1)


if __name__ == '__main__':
    unittest.main()
