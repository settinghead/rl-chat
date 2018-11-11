#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pycorenlp import StanfordCoreNLP
import random
import math
import numpy as np

USE_GLOVE = True

EMBEDDING_DIM = get_embedding_dim(USE_GLOVE)

sentimental_words = ["absolutely","abundant","accept","acclaimed","accomplishment","achievement","action","active","activist","acumen","adjust","admire","adopt","adorable","adored","adventure","affirmation","affirmative","affluent","agree","airy","alive","alliance","ally","alter","amaze","amity","animated","answer","appreciation","approve","aptitude","artistic","assertive","astonish","astounding","astute","attractive","authentic","basic","beaming","beautiful","believe","benefactor","benefit","bighearted","blessed","bliss","bloom","bountiful","bounty","brave","bright","brilliant","bubbly","bunch","burgeon","calm","care","celebrate","certain","change","character","charitable","charming","cheer","cherish","clarity","classy","clean","clever","closeness","commend","companionship","complete","comradeship","confident","connect","connected","constant","content","conviction","copious","core","coupled","courageous","creative","cuddle","cultivate","cure","curious","cute","dazzling","delight","direct","discover","distinguished","divine","donate","each","day","eager","earnest","easy","ecstasy","effervescent","efficient","effortless","electrifying","elegance","embrace","encompassing","encourage","endorse","energized","energy","enjoy","enormous","enthuse","enthusiastic","entirely","essence","established","esteem","everyday","everyone","excited","exciting","exhilarating","expand","explore","express","exquisite","exultant","faith","familiar","family","famous","feat","fit","flourish","fortunate","fortune","freedom","fresh","friendship","full","funny","gather","generous","genius","genuine","give","glad","glow","good","gorgeous","grace","graceful","gratitude","green","grin","group","grow","handsome","happy","harmony","healed","healing","healthful","healthy","heart","hearty","heavenly","helpful","here","highest","good","hold","holy","honest","honor","hug","i","affirm","i","allow","i","am","willing","i","am.","i","can","i","choose","i","create","i","follow","i","know","i","know,","without","a","doubt","i","make","i","realize","i","take","action","i","trust","idea","ideal","imaginative","increase","incredible","independent","ingenious","innate","innovate","inspire","instantaneous","instinct","intellectual","intelligence","intuitive","inventive","joined","jovial","joy","jubilation","keen","key","kind","kiss","knowledge","laugh","leader","learn","legendary","let","go","light","lively","love","loveliness","lucidity","lucrative","luminous","maintain","marvelous","master","meaningful","meditate","mend","metamorphosis","mind-blowing","miracle","mission","modify","motivate","moving","natural","nature","nourish","nourished","novel","now","nurture","nutritious","one","open","openhanded","optimistic","paradise","party","peace","perfect","phenomenon","pleasure","plenteous","plentiful","plenty","plethora","poise","polish","popular","positive","powerful","prepared","pretty","principle","productive","project","prominent","prosperous","protect","proud","purpose","quest","quick","quiet","ready","recognize","refinement","refresh","rejoice","rejuvenate","relax","reliance","rely","remarkable","renew","renowned","replenish","resolution","resound","resources","respect","restore","revere","revolutionize","rewarding","rich","robust","rousing","safe","secure","see","sensation","serenity","shift","shine","show","silence","simple","sincerity","smart","smile","smooth","solution","soul","sparkling","spirit","spirited","spiritual","splendid","spontaneous","still","stir","strong","style","success","sunny","support","sure","surprise","sustain","synchronized","team","thankful","therapeutic","thorough","thrilled","thrive","today","together","tranquil","transform","triumph","trust","truth","unity","unusual","unwavering","upbeat","value","vary","venerate","venture","very","vibrant","victory","vigorous","vision","visualize","vital","vivacious","voyage","wealthy","welcome","well","whole","wholesome","willing","wonder","wonderful","wondrous","xanadu","yes","yippee","young","youth","youthful","zeal","zest","zing","zip"]
sentimental_words_embd = get_GloVe_embeddings(
    sentimental_words, EMBEDDING_DIM)
sim_scores = np.dot(sentimental_words_embd, np.transpose(targ_lang_embd))
print(sim_scores.shape)

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
