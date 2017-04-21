import math
import statistics
import warnings
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(name)s:  %(message)s')

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def train_and_score_model(self, n_components):
        model = self.base_model(n_components)
        if(model != None):
            try:
                return model.score(self.X, self.lengths)
            except ValueError as e:
                logging.error("failure to score model for {} with {} states. error: {}".format(
                        self.this_word, n_components, e))
        return None

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scores = pd.DataFrame([{'n_components': n_components,
                                'score': self.train_and_score_model(n_components),
                                'features': self.X.shape[1],
                                'data_points': len(self.X)}
                               for n_components in range(self.min_n_components, self.max_n_components+1)])
        # p = n*(n-1) + (n-1) + 2*d*n where d = features and n = hmm_states
        # p = n^2 + 2*d*n - 1
        # BIC = -2 * logL + p * logN
        scores['bic_score'] = (-2 * scores['score']) + (scores['n_components']
                                                        ** 2) + (2 * scores['data_points']*scores['n_components']) - 1
        if pd.isnull(scores['bic_score'].idxmin()):
            logging.info("All scores are NaN returning deafult model")
            return self.base_model(self.n_constant)

        min_scoring_n_components = scores.ix[
            scores['bic_score'].idxmin(), 'n_components']
        return self.base_model(min_scoring_n_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        scores_list = list()
        for n_components in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n_components)
            if(model == None):
                continue
            for word, (X, lengths) in self.hwords.items():
                score = None
                try:
                    score = model.score(X, lengths)
                except ValueError as e:
                    logging.error("failure to score model for {} with {} states. error: {}".format(
                        word, n_components, e))
                scores_list.append({
                    'n_components': n_components,
                    'score': score,
                    'is_trained_word': True if word == self.this_word else False})
        scores = pd.DataFrame(scores_list)
        sums = scores.groupby(['n_components']).apply(
            lambda x: pd.Series(dict(
                trained_word_score=x[
                    x['is_trained_word'] == True]['score'].sum(),
                other_words_score_sum=x[
                    x['is_trained_word'] == False]['score'].sum(),
                other_words_count=x[x['is_trained_word']
                                    == False]['score'].count()
            )))
        sums['dic_score'] = sums['trained_word_score'] - \
            (sums['other_words_score_sum']/(sums['other_words_score_sum']-1))
        if pd.isnull(sums['dic_score'].idxmax()):
            return None
        max_scoring_n_components = sums['dic_score'].idxmax()
        return self.base_model(max_scoring_n_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def train_and_score_model(self, n_components, cv_train_idx, cv_test_idx):
        train_x, train_lengths = combine_sequences(
            cv_train_idx, self.sequences)
        try:
            hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state).fit(train_x, train_lengths)
            test_x, test_lengths = combine_sequences(
                cv_test_idx, self.sequences)
            return hmm_model.score(test_x, test_lengths)
        except ValueError as e:
            logging.error("failure to score model for {} with {} states. error: {}".format(
                        self.this_word, n_components, e))
        return None

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n_components_range = range(self.min_n_components, self.max_n_components+1)
        folds = [([0], [0])]
        if len(self.sequences) < 2:
            logging.info("Not enought sequences[{}] to train CV".format(len(self.sequences)))
        else:
            folds = KFold(n_splits=min(3, len(self.sequences))).split(self.sequences)
        scores = pd.DataFrame([{'n_components': n_components, 'score': self.train_and_score_model(n_components, cv_train_idx, cv_test_idx)}
                               for cv_train_idx, cv_test_idx in folds for n_components in n_components_range]).fillna(value=float('-inf'))
        max_scoring_n_component = scores.groupby(
            ['n_components'])['score'].mean().argmax()
        return self.base_model(max_scoring_n_component)
