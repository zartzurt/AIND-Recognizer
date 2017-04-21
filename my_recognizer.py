import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word, (X, lengths) in test_set.get_all_Xlengths().items():
      max_scoring_word_score = (None,float('-inf'))
      for candidate_word, candidate_model in models.items():
        score = float('-inf')
        try:
          score = candidate_model.score(X,lengths)
        except Exception:
           pass
        if score > max_scoring_word_score[1]:
          max_scoring_word_score = candidate_word, score
      probabilities.append(max_scoring_word_score[1])
      guesses.append(max_scoring_word_score[0])
    return probabilities, guesses