#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:12:52 2018

@author: alicerueda
"""
# this is an example code to use standford Sentiment Tree
# you need to install pycorenlp package first
# pip install pycorenlp

from pycorenlp import StanfordCoreNLP
import random
import math

# TODO: sample from a bigger source
sentences = [
    'hello. is anyone there?',
    'i am a tree and you are a bird.',
    "the world is spinning so fast i'm buying new nike shoes.",
    'i hate everyone.',
    "i'm glad it works",
    "This is fantastic! Excellent! It's awesome!"
]


CONVO_LEN = 15


class SentmentEnvironment:
    def __init__(self):
        self.reset()
        self.stanford = StanfordCoreNLP('http://localhost:9000')

    def step(self, action: str):
        result = self.stanford.annotate(action,
                                        properties={
                                            'annotators': 'sentiment',
                                            'outputFormat': 'json',
                                            'timeout': '5000'
                                        })

        # Result types from CoreNLP:
        # negative: 1; # neutral: 2; positive: 3
        s_scores = [int(s['sentimentValue']) - 2 for s in result['sentences']]
        reward = math.tanh(sum(s_scores))

        done = len(self.history) > CONVO_LEN
        self.history.append(action)

        state = random.sample(sentences, 1)[0]

        return state, reward, done

    def reset(self):
        self.history = []


def fake_bot(state):
    return random.sample(sentences, 1)[0]


# some test code
if __name__ == "__main__":
    env = SentmentEnvironment()
    done = False
    prev_state, reward, done = env.step("hello")
    while not done:
        action = fake_bot(prev_state)
        state, reward, done = env.step(action)
        print(f"env: {prev_state} | bot: {action} reward: {reward}")
        prev_state = state
