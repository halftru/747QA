import random
import pickle
from keras.preprocessing.sequence import pad_sequences


class Vocabulary(dict):
    def __init__(self, vocabulary_file_name):
        super().__init__()
        with open(vocabulary_file_name) as vocabulary_file:
            for line in vocabulary_file:
                key, value = line.split()
                self[int(key)] = value
        self[0] = '<PAD>'

    def __setitem__(self, key, value):
        if key in self:
            raise Exception('Repeat Key', key)
        if value in self:
            raise Exception('Repeat value', value)
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2


def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)


class QAData:
    def __init__(self):
        self.vocabulary = Vocabulary("./data/vocab_all.txt")
        self.sentence_length = 200
        self.answers = pickle.load(open("./data/answers.pkl", 'rb'))
        self.training_set = pickle.load(open("./data/train.pkl", 'rb'))

    def get_training_data(self):
        questions = []
        good_answers = []
        for j, qa in enumerate(self.training_set):
            questions.extend([qa['question']] * len(qa['answers']))
            good_answers.extend([self.answers[i] for i in qa['answers']])

        # pad the question and answers
        questions = pad(questions, self.sentence_length)
        good_answers = pad(good_answers, self.sentence_length)
        bad_answers = pad(random.sample(list(self.answers.values()), len(good_answers)), self.sentence_length)

        return questions, good_answers, bad_answers

    def process_data(self, d):
        indices = d['good'] + d['bad']
        answers = pad([self.answers[i] for i in indices], self.sentence_length)
        question = pad([d['question']] * len(indices), self.sentence_length)
        return indices, answers, question

    def process_test_data(self, question, answers):
        answer_unpadded = []
        for answer in answers:
            print(answer.split(' '))
            answer_unpadded.append([self.vocabulary[word] for word in answer.split(' ')])
        answers = pad(answer_unpadded, self.sentence_length)
        question = pad([[self.vocabulary[word] for word in question.split(' ')]] * len(answers), self.sentence_length)
        return answers, question
