from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

__author__ = ['tylin', 'simonsuster']


class COCOEvalCap:
    def __init__(self, ground, predictions):
        self.eval = {}
        self.ground = ground
        self.predictions = predictions

    def evaluate(self):
        assert len(self.ground) == len(self.predictions)

        # =================================================
        # Set up scorers
        # =================================================
        #print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            #print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.ground, self.predictions)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    #print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                #print("%s: %0.3f" % (method, score))

    def setEval(self, score, method):
        self.eval[method] = score
