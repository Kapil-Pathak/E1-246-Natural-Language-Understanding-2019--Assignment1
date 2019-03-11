import numpy as np
import nltk
nltk.download('reuters')
print("ch")
from nltk.corpus import reuters
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
englishStemmer=SnowballStemmer("english")
nltk.download('wordnet')

training_words=[]
testing_words=[]
for idx in reuters.fileids():
    if idx.split('/')[0]=='training':
        for word in reuters.words('training/'+idx.split('/')[1]):
            training_words.append(word)
    #training_words.append(reuters.words('training/'+id.split('/')[1]))
    else:
        for word in reuters.words('test/'+idx.split('/')[1]):
            testing_words.append(word)

            #english_stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
englishStemmer=SnowballStemmer("english")
            #filtered_words = [englishStemmer.stem(w.lower()) for w in list(training_words) if not w in english_stopwords if len(w)>1 if w.isalpha()]
filtered_words = [w.lower() for w in list(training_words) if w.isalpha()]

vocab_size=len(filtered_words)
from collections import Counter
dictionary = dict()
count = [('UNK', -1)]
count.extend(Counter(filtered_words).most_common(vocab_size - 1))
file = open('vocab_4.txt', 'w')

index = 0
for word, cnt in count:
        dictionary[word] = index
        index += 1
        file.write(word+','+str(cnt)+'\n')
ind2words = dict(zip(dictionary.values(), dictionary.keys()))
file.close()

import pickle
with open('dictionary_4.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('ind2words_4.pickle', 'wb') as handle:
    pickle.dump(ind2words, handle, protocol=pickle.HIGHEST_PROTOCOL)

counts_dict = dict()
for i in filtered_words:
    counts_dict[i] = counts_dict.get(i, 0) + 1

Sub_sampling=dict()
for w in counts_dict:
    Sub_sampling[w]=1-np.sqrt(1/counts_dict[w])

import random
Subsampled_FreqW=[]
for i in filtered_words:
    if random.random()<=Sub_sampling[i]:
        Subsampled_FreqW.append(i)

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    file = open('center_context_pair_4.txt', 'w')
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        # get a random target before the center word
        for target1 in index_words[max(0, index - context_window_size): index]:
            file.write(str(center)+","+str(target1)+"\n")

        # get a random target after the center wrod
        for target2 in index_words[index + 1: index + context_window_size + 1]:
            file.write(str(center)+","+str(target2)+"\n")
    file.close()

words2ind=convert_words_to_index(Subsampled_FreqW, dictionary)

generate_sample(words2ind, 4)

file1=open('center_context_pair_4.txt','r')
i=0
context_dict=dict()
for i in range(vocab_size):
    context_dict[i]=set()
with open('center_context_pair_4.txt') as f:
    content = f.readlines()
    for line in content:
        context_dict[int(line.split(',')[0])].add(int(line.split(',')[1].split('\n')[0]))

import random
all_words=len(Subsampled_FreqW)
distinct_vocab=set(filtered_words)

distinct_vocab_size=len(distinct_vocab)

negative_sampling=[]
for word in distinct_vocab:
    unigram=(counts_dict[word])**(0.75)
    for i in range(int(round(unigram))):
        negative_sampling.append(word)

dimension=300
word_embeddings = np.random.uniform(0,1,[distinct_vocab_size+1, dimension])
context_embeddings = np.random.uniform(0,1,[distinct_vocab_size+1, dimension])

def getNegativeSamples(target,K1,context_dict,dictionary,negative_sampling):
    """ Samples K indexes which are not the target """
    indices = [None] * K1
    for k1 in range(K1):
        newidx =random.randint(0,len(negative_sampling)-1)
        while negative_sampling[newidx] in context_dict[target]:
            newidx =random.randint(0,len(negative_sampling))
        indices[k1] = dictionary[negative_sampling[newidx]]
    return indices

def sigmoid(x):
    return 1/(1+np.exp(-x))

from tqdm import tqdm
MAX_EPOCH=10
neg_samples=10
lr=0.001
total_loss2=[]
for epoch in tqdm(range(MAX_EPOCH)):
    with open('center_context_pair_4.txt') as train_read:
        train=train_read.readlines()
        total_ex=len(train)
        loss_per_epoch=0
        for line in tqdm(train):
            neg_loss=0
            w_ce=word_embeddings[int(line.split(',')[0]),:]
            w_co=context_embeddings[int(line.split(',')[1]),:]
            indx_neg=getNegativeSamples(int(line.split(',')[0]),neg_samples,context_dict,dictionary,negative_sampling)
            for k in range(neg_samples):
                neg_loss+=np.log(sigmoid(-context_embeddings[int(indx_neg[k]),:].dot(w_ce)))
            w_ce_grad=(1-sigmoid(w_ce.dot(w_co)))*w_co
            loss_per_epoch+=np.log(sigmoid(w_ce.dot(w_co)))+neg_loss
            for k in range(neg_samples):
                w_ce_grad-=(1-sigmoid(-context_embeddings[int(indx_neg[k]),:].dot(w_ce)))*context_embeddings[int(indx_neg[k]),:]
            word_embeddings[int(line.split(',')[0]),:]+=lr*w_ce_grad
            w_co_grad=(1-sigmoid(w_ce.dot(w_co)))*w_ce
            context_embeddings[int(line.split(',')[1]),:]+=lr*w_co_grad
            for k in range(neg_samples):
                context_embeddings[int(indx_neg[k]),:]-=lr*(1-sigmoid(-context_embeddings[int(indx_neg[k]),:].dot(w_ce)))*w_ce
    total_loss2.append(loss_per_epoch/total_ex)

print(total_loss2)

np.save('word_embeddings_d300_w4.npy',word_embeddings)
np.save('context_embeddings_d300_w4.npy',context_embeddings)
