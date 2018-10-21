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
stanford = StanfordCoreNLP('http://localhost:9000')

sentences = [
    'hello. is anyone there?',
    'i am a tree and you are a bird.',
    "the world is spinning so fast i'm buying new nike shoes.",
    'i hate everyone.',
    "i'm glad it works",
    "This is fantastic! Excellent! It's awesome!"
]

for sentence in sentences:
    print(sentence)
    result = stanford.annotate(sentence,
                               properties={
                                   'annotators': 'sentiment',
                                   'outputFormat': 'json',
                                   'timeout': '5000'
                               })
    for s in result['sentences']:
        score = (s['sentimentValue'], s['sentiment'])
    print(f'\tScore: {score[0]}, Value: {score[1]}')
