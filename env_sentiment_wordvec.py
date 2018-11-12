#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import numpy as np
from corpus_utils import LanguageIndex
from data import load_conv_text
from embedding_utils import get_embedding_dim, get_GloVe_embeddings
from corpus_utils import tokenize_sentence
from sklearn.metrics.pairwise import cosine_similarity


USE_GLOVE = True
EMBEDDING_DIM = get_embedding_dim(USE_GLOVE)

pos_sentimental_words = ["absolutely","abundant","accept","acclaimed","accomplishment","achievement","action","active","activist","acumen","adjust","admire","adopt","adorable","adored","adventure","affirmation","affirmative","affluent","agree","airy","alive","alliance","ally","alter","amaze","amity","animated","answer","appreciation","approve","aptitude","artistic","assertive","astonish","astounding","astute","attractive","authentic","basic","beaming","beautiful","believe","benefactor","benefit","bighearted","blessed","bliss","bloom","bountiful","bounty","brave","bright","brilliant","bubbly","bunch","burgeon","calm","care","celebrate","certain","change","character","charitable","charming","cheer","cherish","clarity","classy","clean","clever","closeness","commend","companionship","complete","comradeship","confident","connect","connected","constant","content","conviction","copious","core","coupled","courageous","creative","cuddle","cultivate","cure","curious","cute","dazzling","delight","direct","discover","distinguished","divine","donate","each","day","eager","earnest","easy","ecstasy","effervescent","efficient","effortless","electrifying","elegance","embrace","encompassing","encourage","endorse","energized","energy","enjoy","enormous","enthuse","enthusiastic","entirely","essence","established","esteem","everyday","everyone","excited","exciting","exhilarating","expand","explore","express","exquisite","exultant","faith","familiar","family","famous","feat","fit","flourish","fortunate","fortune","freedom","fresh","friendship","full","funny","gather","generous","genius","genuine","give","glad","glow","good","gorgeous","grace","graceful","gratitude","green","grin","group","grow","handsome","happy","harmony","healed","healing","healthful","healthy","heart","hearty","heavenly","helpful","here","highest","good","hold","holy","honest","honor","hug","i","affirm","i","allow","i","am","willing","i","am.","i","can","i","choose","i","create","i","follow","i","know","i","know,","without","a","doubt","i","make","i","realize","i","take","action","i","trust","idea","ideal","imaginative","increase","incredible","independent","ingenious","innate","innovate","inspire","instantaneous","instinct","intellectual","intelligence","intuitive","inventive","joined","jovial","joy","jubilation","keen","key","kind","kiss","knowledge","laugh","leader","learn","legendary","let","go","light","lively","love","loveliness","lucidity","lucrative","luminous","maintain","marvelous","master","meaningful","meditate","mend","metamorphosis","mind-blowing","miracle","mission","modify","motivate","moving","natural","nature","nourish","nourished","novel","now","nurture","nutritious","one","open","openhanded","optimistic","paradise","party","peace","perfect","phenomenon","pleasure","plenteous","plentiful","plenty","plethora","poise","polish","popular","positive","powerful","prepared","pretty","principle","productive","project","prominent","prosperous","protect","proud","purpose","quest","quick","quiet","ready","recognize","refinement","refresh","rejoice","rejuvenate","relax","reliance","rely","remarkable","renew","renowned","replenish","resolution","resound","resources","respect","restore","revere","revolutionize","rewarding","rich","robust","rousing","safe","secure","see","sensation","serenity","shift","shine","show","silence","simple","sincerity","smart","smile","smooth","solution","soul","sparkling","spirit","spirited","spiritual","splendid","spontaneous","still","stir","strong","style","success","sunny","support","sure","surprise","sustain","synchronized","team","thankful","therapeutic","thorough","thrilled","thrive","today","together","tranquil","transform","triumph","trust","truth","unity","unusual","unwavering","upbeat","value","vary","venerate","venture","very","vibrant","victory","vigorous","vision","visualize","vital","vivacious","voyage","wealthy","welcome","well","whole","wholesome","willing","wonder","wonderful","wondrous","xanadu","yes","yippee","young","youth","youthful","zeal","zest","zing","zip"]
neg_sentimental_words = ["abysmal","adverse","alarming","angry","annoy","anxious","apathy","appalling","atrocious","awful","bad","banal","barbed","belligerent","bemoan","beneath","boring","broken","callous","can't","clumsy","coarse","cold","cold-hearted","collapse","confused","contradictory","contrary","corrosive","corrupt","crazy","creepy","criminal","cruel","cry","cutting","damage","damaging","dastardly","dead","decaying","deformed","deny","deplorable","depressed","deprived","despicable","detrimental","dirty","disease","disgusting","disheveled","dishonest","dishonorable","dismal","distress","don't","dreadful","dreary","enraged","eroding","evil","fail","fault","faulty","fear","feeble","fight","filthy","foul","frighten","frightful","gawky","ghastly","grave","greed","grim","grimace","gross","grotesque","gruesome","guilty","haggard","hard","hard-hearted","harmful","hate","hideous","homely","horrendous","horrible","hostile","hurt","hurtful","icky","ignorant","ignore","ill","immature","imperfect","impossible","inane","inelegant","infernal","injure","injurious","insane","insidious","insipid","jealous","junky","lose","lousy","lumpy","malicious","mean","menacing","messy","misshapen","missing","misunderstood","moan","moldy","monstrous","naive","nasty","naughty","negate","negative","never","no","nobody","nondescript","nonsense","not","noxious","objectionable","odious","offensive","old","oppressive","pain","perturb","pessimistic","petty","plain","poisonous","poor","prejudice","questionable","quirky","quit","reject","renege","repellant","reptilian","repugnant","repulsive","revenge","revolting","rocky","rotten","rude","ruthless","sad","savage","scare","scary","scream","severe","shocking","shoddy","sick","sickening","sinister","slimy","smelly","sobbing","sorry","spiteful","sticky","stinky","stormy","stressful","stuck","stupid","substandard","suspect","suspicious","tense","terrible","terrifying","threatening","ugly","undermine","unfair","unfavorable","unhappy","unhealthy","unjust","unlucky","unpleasant","unsatisfactory","unsightly","untoward","unwanted","unwelcome","unwholesome","unwieldy","unwise","upset","vice","vicious","vile","villainous","vindictive","wary","weary","wicked","woeful","worthless","wound","yell","yucky","zero", "shit"]
pos_sentimental_words_embd = get_GloVe_embeddings(pos_sentimental_words, EMBEDDING_DIM)
# neg_sentimental_words_embd = get_GloVe_embeddings(neg_sentimental_words, EMBEDDING_DIM)

CONVO_LEN = 15

class PoorMansSentimentEnvrionment:
    def __init__(self, targ_lang: LanguageIndex, by_word: bool = False):
        targ_lang_embd = get_GloVe_embeddings(targ_lang.vocab, EMBEDDING_DIM)
        pos_pairwise_sim_scores = np.dot(targ_lang_embd, pos_sentimental_words_embd.transpose())  / len(pos_sentimental_words) * 100
        # neg_pairwise_sim_scores = np.dot(neg_sentimental_words_embd, np.transpose(targ_lang_embd)) / len(neg_sentimental_words) * 100
        self.targ_lang = targ_lang
        self.by_word = by_word
        # calc sentiment score for all words in the vocab
        self.sentiment_scores = np.max(pos_pairwise_sim_scores, axis=1)
        self.reset()
        

    def step(self, action: str):
        action = action.lower()
        if(self.by_word):
            reward = self.calc_reward_w(action)
        else:
            # TODO: better sentence level sentiment analysis
            reward = np.mean([self.calc_reward_w(w) for w in tokenize_sentence(action)])
        done = len(self.history) > CONVO_LEN
        self.history.append(action)
        state = random.sample(questions, 1)[0]
        return state, reward, done

    def calc_reward_w(self, word: str):
        w_idx = self.targ_lang.word2idx[word]
        s = self.sentiment_scores[w_idx]
        # print(f"{word} ({w_idx}): {s}")
        return s

    def reset(self):
        self.history = []


def fake_bot(state):
    return random.sample(answers, 1)[0]

# some test code
if __name__ == "__main__":
    questions, answers = load_conv_text()
    inp_lang = LanguageIndex(questions)
    targ_lang = LanguageIndex(answers)

    env = PoorMansSentimentEnvrionment(targ_lang, by_word=False)
    words = ["good", "excellent", "sad", "dog", "fantastic", "meh", "party", "death", "disease"]
    print([(w, (env.sentiment_scores[env.targ_lang.word2idx[w]])) for w in words])
    done = False
    prev_state, reward, done = env.step("hello")
    while not done:
        action = fake_bot(prev_state)
        state, reward, done = env.step(action)
        print(f"bot: {action} reward: {reward}")
        prev_state = state
