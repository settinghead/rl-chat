import random
import string


def random_utterance(min_len, max_len):
    utt_len = random.randint(min_len, max_len + 1)
    random_chars = [random.choice(
        string.ascii_uppercase +
        string.digits +
        string.ascii_lowercase + ' .,!?'
    ) for _ in range(utt_len)]
    result = ' '.join(random_chars)
    result = '<start> ' + result
    return result
