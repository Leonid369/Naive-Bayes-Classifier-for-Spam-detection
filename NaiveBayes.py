# Submitted by
# Saraswathi Shanmugamoorthy
# CS 6375.003
# Assignment 2 - Spam detection using Multinomial Naive Bayes 

import collections
import re
import math
import copy
import os
import sys

# To save or store emails as dictionaries.
train_set = dict()
testing_set = dict()


# stop words list
stop_words = []

# Filtered sets without stop words
train_set_filtered = dict()
testing_set_filtered = dict()

classes = ["ham", "spam"]


#priors
prior = dict()
prior_filtered = dict()

# Conditional probabilities of train set
cond_probability = dict()
cond_probability_filtered = dict()


def dataSetsMaking(store_dict, directory, classes_true):
    for directory_entry in os.listdir(directory):
        directory_entry_path = os.path.join(directory, directory_entry)
        if os.path.isfile(directory_entry_path):
            with open(directory_entry_path, 'r') as text_file:
                text = text_file.read()
                store_dict.update({directory_entry_path: Docs(text, wordsFreq(text), classes_true)})


def settingStopWords():
    stop = []
    with open('stop_words.txt', 'r') as txt:
        stop = (txt.read().splitlines())
    return stop

def vocabWords(data_sets):
    all_text = ""
    v = []
    for x in data_sets:
        all_text += data_sets[x].getTexts()
    for y in wordsFreq(all_text):
        v.append(y)
    return v

# For counting frequency of the words
def wordsFreq(text):
    wordsFrequency = collections.Counter(re.findall(r'\w+', text))
    return dict(wordsFrequency)


# To remove the stop words
def stopWordsRemoving(stop, data_sets):
    data_sets_filtered = copy.deepcopy(data_sets)
    for i in stop:
        for j in data_sets_filtered:
            if i in data_sets_filtered[j].getWordFrequency():
                del data_sets_filtered[j].getWordFrequency()[i]
    return data_sets_filtered



def multiNBtrain(train, priors, cond):
    v = vocabWords(train)
    n = len(train)
    for c in classes:
        n_c = 0.0
        text_c = ""
        for i in train:
            if train[i].getClassesTrue() == c:
                n_c += 1
                text_c += train[i].getTexts()
        priors[c] = float(n_c) / float(n)
        token_frequency = wordsFreq(text_c)
        for t in v:
            if t in token_frequency:
                cond.update({t + "_" + c: (float((token_frequency[t] + 1.0)) / float((len(text_c) + len(token_frequency))))})
            else:
                cond.update({t + "_" + c: (float(1.0) / float((len(text_c) + len(token_frequency))))})


# Applying the multinomial NB algorithm
def applyingMultiNB(data_instance, priors, cond):
    count_scores = {}
    for c in classes:
        count_scores[c] = math.log10(float(priors[c]))
        for t in data_instance.getWordFrequency():
            if (t + "_" + c) in cond:
                count_scores[c] += float(math.log10(cond[t + "_" + c]))
    if count_scores["spam"] > count_scores["ham"]:
        return "spam"
    else:
        return "ham"


class Docs:
    text = ""
    word_frequency = {}

    classes_true = ""
    learned_classes = ""

    def __init__(self, text, counter, classes_true):
        self.text = text
        self.word_frequency = counter
        self.classes_true = classes_true

    def getTexts(self):
        return self.text

    def getWordFrequency(self):
        return self.word_frequency

    def getClassesTrue(self):
        return self.classes_true

    def getLearnedClasses(self):
        return self.learned_classes

    def setLearnedClasses(self, guess):
        self.learned_classes = guess


def main(training_dir_spam, training_dir_ham, testing_dir_spam, testing_dir_ham):
    dataSetsMaking(train_set, training_dir_spam, classes[1])
    dataSetsMaking(train_set, training_dir_ham, classes[0])
    dataSetsMaking(testing_set, testing_dir_spam, classes[1])
    dataSetsMaking(testing_set, testing_dir_ham, classes[0])

    # list of Stop words
    stop_words = settingStopWords()

    train_set_filtered = stopWordsRemoving(stop_words, train_set)
    testing_set_filtered = stopWordsRemoving(stop_words, testing_set)

    multiNBtrain(train_set, prior, cond_probability)
    multiNBtrain(train_set_filtered, prior_filtered, cond_probability_filtered)

    correct_predicts = 0
    for i in testing_set:
        testing_set[i].setLearnedClasses(applyingMultiNB(testing_set[i], prior, cond_probability))
        if testing_set[i].getLearnedClasses() == testing_set[i].getClassesTrue():
            correct_predicts += 1

    correct_predicts_filtered = 0
    for i in testing_set_filtered:
        testing_set_filtered[i].setLearnedClasses(applyingMultiNB(testing_set_filtered[i], prior_filtered,
                                                                cond_probability_filtered))
        if testing_set_filtered[i].getLearnedClasses() == testing_set_filtered[i].getClassesTrue():
            correct_predicts_filtered += 1
    print "**********************************************************************"
    print " "
    print "Number of correct predictions without filtering the stop words:\t%d/%s" % (correct_predicts, len(testing_set))
    print "Accuracy without filtering the stop words:\t\t\t%.4f%%" % (100.0 * float(correct_predicts) / float(len(testing_set)))
    print "**********************************************************************"
    print " "
    print "Number of correct predictions after filtering the stop words:\t%d/%s" % (correct_predicts_filtered, len(testing_set_filtered))
    print "Accuracy after filtering the stop words:\t\t\t%.4f%%" % (100.0 * float(correct_predicts_filtered) / float(len(testing_set_filtered)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
