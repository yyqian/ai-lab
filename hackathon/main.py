#!/usr/bin/env python3
from gensim.models import doc2vec
from gensim import corpora, models
import jieba
import os
from gensim.similarities.docsim import Similarity
import re


def remove_stopwords(word_list, stopwords):
    cleaned_word_list = []
    for word in word_list:
        if word not in stopwords:
            cleaned_word_list.append(word)
    return cleaned_word_list


def read_file(f):
    return f.read().replace(' ', '').replace('\t', '').replace(
        '\r\n', '').replace('\r', '').replace('\n', '')


# 读取数据
raw_documents = []
for root, p, files in os.walk('./testdata/'):
    for file in files:
        f = open(root+file, encoding='utf8')
        s = read_file(f)
        raw_documents.append(s)
        f.close()
print('data ok!')
# 加载停用词
stopwords = set([line.strip() for line in open('stopword.dic').readlines()])
print('停词表大小', len(stopwords))
# 准备语料
corpora_documents = []
#corpora_documents2 = []
for i, item_text in enumerate(raw_documents):
    words_list = list(jieba.cut(item_text))
    print(i, '分词数量', len(words_list))
    # 过滤停用词
    cleaned_words_list = remove_stopwords(words_list, stopwords)
    print(i, '过滤停用词之后分词数量', len(cleaned_words_list))
    document = doc2vec.TaggedDocument(words=cleaned_words_list, tags=[i])
    corpora_documents.append(cleaned_words_list)
    # corpora_documents2.append(document)
# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
corpus = [dictionary.doc2bow(text) for text in corpora_documents]
# 测试数据
f = open('target.txt', 'r')
test_data = read_file(f)
f.close()
test_cut_raw = list(jieba.cut(test_data))
print('测试文本分词数量', len(test_cut_raw))
test_cut_raw = remove_stopwords(test_cut_raw, stopwords)
print('测试文本过滤停用词之后分词数量', len(test_cut_raw))

# 转化成tf-idf向量
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = [tfidf_model[doc] for doc in corpus]
# 转化成lsi向量
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
corpus_lsi = [lsi[doc] for doc in corpus]
print('corpus_lsi', corpus_lsi)
similarity_lsi = Similarity('Similarity-Lsi-index',
                            corpus_lsi, num_features=1600, num_best=5)
test_corpus = dictionary.doc2bow(test_cut_raw)  # 2.转换成bow向量
print('test_corpus', test_corpus)
test_corpus_tfidf = tfidf_model[test_corpus]  # 3.计算tfidf值
print('test_corpus_tfidf', test_corpus_tfidf)
test_corpus_lsi = lsi[test_corpus_tfidf]  # 4.计算lsi值
print('test_corpus_lsi', test_corpus_lsi)
# lsi.add_documents(test_corpus_lsi) #更新LSI的值
print('——————————————lsi———————————————')
print(similarity_lsi[test_corpus_lsi])
