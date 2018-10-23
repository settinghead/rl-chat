

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


def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)


MAX_LEN = 130


def load_opensubtitles_text():
    with open('dataset/movie_lines_selected_10k.txt', 'rb') as f:
        pairs = [
            (str(q).strip()[:MAX_LEN],
             f"{BEGIN_TAG} {str(a).strip()[:MAX_LEN]} {END_TAG}")
            for q, a in pairwise(f)]
        return tuple(zip(*pairs))