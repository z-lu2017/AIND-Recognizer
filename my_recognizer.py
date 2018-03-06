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
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        test_X, test_lengths = test_set.get_all_Xlengths()[word_id]
        prob = {}
        best_score = float('-inf')
        best_guess = ""
        for word, model in models.items():
            try:
                score = model.score(test_X,test_lengths)
                prob[word] = score
                if logL > best_score:
                    best_score = score
                    best_guess = word
            except:
                prob[word] = float('-inf')
        probabilities.append(prob)
        guesses.append(best_guess)
    return (probabilities, guesses)
