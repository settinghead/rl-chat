import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import encoder_decoder
from corpus_utils import tokenize_sentence, LanguageIndex
from data import BEGIN_TAG, END_TAG
from utils import load_trained_model, max_length
import data
import random
from sklearn.metrics.pairwise import cosine_similarity
import os
from embedding_utils import get_embedding_dim, get_GloVe_embeddings
import time

EPISODES = 1000
BATCH_SIZE = 64
TOP_K = 4
USE_GLOVE = True

EMBEDDING_DIM = get_embedding_dim(USE_GLOVE)

UNITS = 512


def main():
    tf.enable_eager_execution()

    questions1, answers1 = data.load_conv_text()
    # questions2, answers2 = data.load_opensubtitles_text()

    questions = list(questions1)
    answers = list(answers1)

    inp_lang, targ_lang = LanguageIndex(questions), LanguageIndex(answers)

    input_tensor = [[inp_lang.word2idx[token]
                     for token in tokenize_sentence(question)] for question in questions]
    target_tensor = [[targ_lang.word2idx[token]
                      for token in tokenize_sentence(answer)] for answer in answers]
    max_length_inp, max_length_tar = max_length(
        input_tensor), max_length(target_tensor)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    model: encoder_decoder.Seq2Seq = load_trained_model(
        BATCH_SIZE, EMBEDDING_DIM, UNITS, tf.train.AdamOptimizer())

    # sentimental_words = ["absolutely","abundant","accept","acclaimed","accomplishment","achievement","action","active","activist","acumen","adjust","admire","adopt","adorable","adored","adventure","affirmation","affirmative","affluent","agree","airy","alive","alliance","ally","alter","amaze","amity","animated","answer","appreciation","approve","aptitude","artistic","assertive","astonish","astounding","astute","attractive","authentic","basic","beaming","beautiful","believe","benefactor","benefit","bighearted","blessed","bliss","bloom","bountiful","bounty","brave","bright","brilliant","bubbly","bunch","burgeon","calm","care","celebrate","certain","change","character","charitable","charming","cheer","cherish","clarity","classy","clean","clever","closeness","commend","companionship","complete","comradeship","confident","connect","connected","constant","content","conviction","copious","core","coupled","courageous","creative","cuddle","cultivate","cure","curious","cute","dazzling","delight","direct","discover","distinguished","divine","donate","each","day","eager","earnest","easy","ecstasy","effervescent","efficient","effortless","electrifying","elegance","embrace","encompassing","encourage","endorse","energized","energy","enjoy","enormous","enthuse","enthusiastic","entirely","essence","established","esteem","everyday","everyone","excited","exciting","exhilarating","expand","explore","express","exquisite","exultant","faith","familiar","family","famous","feat","fit","flourish","fortunate","fortune","freedom","fresh","friendship","full","funny","gather","generous","genius","genuine","give","glad","glow","good","gorgeous","grace","graceful","gratitude","green","grin","group","grow","handsome","happy","harmony","healed","healing","healthful","healthy","heart","hearty","heavenly","helpful","here","highest","good","hold","holy","honest","honor","hug","i","affirm","i","allow","i","am","willing","i","am.","i","can","i","choose","i","create","i","follow","i","know","i","know,","without","a","doubt","i","make","i","realize","i","take","action","i","trust","idea","ideal","imaginative","increase","incredible","independent","ingenious","innate","innovate","inspire","instantaneous","instinct","intellectual","intelligence","intuitive","inventive","joined","jovial","joy","jubilation","keen","key","kind","kiss","knowledge","laugh","leader","learn","legendary","let","go","light","lively","love","loveliness","lucidity","lucrative","luminous","maintain","marvelous","master","meaningful","meditate","mend","metamorphosis","mind-blowing","miracle","mission","modify","motivate","moving","natural","nature","nourish","nourished","novel","now","nurture","nutritious","one","open","openhanded","optimistic","paradise","party","peace","perfect","phenomenon","pleasure","plenteous","plentiful","plenty","plethora","poise","polish","popular","positive","powerful","prepared","pretty","principle","productive","project","prominent","prosperous","protect","proud","purpose","quest","quick","quiet","ready","recognize","refinement","refresh","rejoice","rejuvenate","relax","reliance","rely","remarkable","renew","renowned","replenish","resolution","resound","resources","respect","restore","revere","revolutionize","rewarding","rich","robust","rousing","safe","secure","see","sensation","serenity","shift","shine","show","silence","simple","sincerity","smart","smile","smooth","solution","soul","sparkling","spirit","spirited","spiritual","splendid","spontaneous","still","stir","strong","style","success","sunny","support","sure","surprise","sustain","synchronized","team","thankful","therapeutic","thorough","thrilled","thrive","today","together","tranquil","transform","triumph","trust","truth","unity","unusual","unwavering","upbeat","value","vary","venerate","venture","very","vibrant","victory","vigorous","vision","visualize","vital","vivacious","voyage","wealthy","welcome","well","whole","wholesome","willing","wonder","wonderful","wondrous","xanadu","yes","yippee","young","youth","youthful","zeal","zest","zing","zip"]
    
    sentimental_words = ["good", "excellent", "well"]
    targ_lang_embd = get_GloVe_embeddings(targ_lang.vocab, EMBEDDING_DIM)
    sentimental_words_embd = get_GloVe_embeddings(
        sentimental_words, EMBEDDING_DIM)
    sim_scores = np.dot(sentimental_words_embd, np.transpose(targ_lang_embd))
    print(sim_scores.shape)
    #max_prob_ids = np.argmax(sim_scores, axis=1)
    # print(max_prob_ids)
    # print(targ_lang.word2idx)
    # print(targ_lang.idx2word(max_prob_ids[1]))

    optimizer = tf.train.AdamOptimizer()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)

    for episode in range(EPISODES):

        # Start of Episode
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:

                hidden = tf.zeros((BATCH_SIZE, UNITS))
                enc_hidden = model.encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims(
                    [targ_lang.word2idx[BEGIN_TAG]] * BATCH_SIZE, 1)

                loss = 0  # loss for decoder
                pg_loss = 0  # loss for semantic

                result = ''
                for t in range(1, targ.shape[1]):
                    actions = []
                    probs = []
                    rewards = []
                    predictions, dec_hidden = model.decoder(
                        dec_input, dec_hidden)
                    '''
                    predicted_id = tf.argmax(predictions[0]).numpy()
                    if targ_lang.idx2word[predicted_id] == END_TAG:
                        print("result: ", result)
                    else:
                        result += ' ' + targ_lang.idx2word[predicted_id]
                    '''
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
                    for ps in predictions:
                        # action = tf.distributions.Categorical(ps).sample(1)[0]
                        top_k_indices = tf.nn.top_k(ps, TOP_K).indices.numpy()
                        action = np.random.choice(top_k_indices, 1)[0]
                        actions.append(action)
                        prob = ps.numpy()[action]
                        probs.append(prob)
                        reward = np.max(sim_scores[1:, action])
                        print(targ_lang.idx2word[action], reward)
                        # print(targ_lang.idx2word[action], reward)
                        rewards.append(reward)

                        # normalize reward
                        reward_mean = np.mean(rewards)
                        reward_std = np.std(rewards)
                        norm_rewards = [(r - reward_mean) /
                                        reward_std for r in rewards]

                    if targ_lang.idx2word[actions[0]] == END_TAG:
                        print("result: ", result)
                    else:
                        result += ' ' + targ_lang.idx2word[actions[0]]

                    onehot_labels = tf.keras.utils.to_categorical(
                        y=actions, num_classes=len(targ_lang.word2idx))

                    norm_rewards = tf.convert_to_tensor(
                        norm_rewards, dtype="float32")
                    # print(onehot_labels.shape)
                    # print(predictions.shape)
                    loss += model.loss_function(targ[:, t], predictions)
                    # print("------")
                    # print(loss)
                    # print(probs)
                    #pg_loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=onehot_labels, logits=targ[:, t]))
                    pg_loss_cross = model.loss_function(
                        targ[:, t],  )
                    # pg_loss_cross = tf.reduce_mean(
                    #     pg_loss_cross * norm_rewards)
                    pg_loss_cross = tf.reduce_mean(
                        pg_loss_cross * rewards)
                    # print(pg_loss_cross)
                    # print("------")
                    # print(pg_loss_cross)
                    pg_loss += pg_loss_cross
                # End of Episode
                # Update policy
                batch_loss = ((loss + pg_loss) / int(targ.shape[1]))
                total_loss += batch_loss
                variables = model.encoder.variables + model.decoder.variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                if batch % 10 == 0:
                    print('batch {} training loss {:.4f}'.format(
                        batch, total_loss.numpy()))

        # saving (checkpoint) the model every 100 epochs
        #if (episode + 1) % 100 == 0:
            #checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for {} episode {} sec\n'.format(
            episode, time.time() - start))


main()
