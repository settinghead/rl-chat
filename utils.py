import random
import string


BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'
def load_conv_text():
    questions = []
    answers = []
    with open('conv.txt') as f:
        for line in f:
            question_answer_pair = line.split("||")
            question = question_answer_pair[0].strip()
            answer = question_answer_pair[1].strip()
            questions.append(question)
            answers.append(BEGIN_TAG + ' ' + answer + ' ' + END_TAG)
    return questions, answers
            

def random_utterance(min_len, max_len):
    utt_len = random.randint(min_len, max_len + 1)
    random_chars = [random.choice(
        string.ascii_uppercase +
        string.digits +
        string.ascii_lowercase + ' .,!?'
    ) for _ in range(utt_len)]
    result = ' '.join(random_chars)
    result = BEGIN_TAG + ' ' + result + ' ' + END_TAG
    print(result)
    return result
