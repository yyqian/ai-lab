#!/usr/bin/env python3
from jieba import analyse
import operator
import word2vec

input_file = './source.txt'

text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

with open(input_file, 'r') as f:
    text = f.read()
#print(text, '\n')

topK = 40

tfidf = analyse.extract_tags
keywords0 = tfidf(text, topK)
print("\nkeywords by tf-idf:")
print(keywords0)

textrank = analyse.textrank
keywords1 = textrank(text, topK)
print("\nkeywords by textrank:")
print(keywords1)

# 合并结果
score = {}
max = topK * 2
for i in range(0, topK):
    if keywords0[i] in score:
        score[keywords0[i]] += max - 2 * i
    else:
        score[keywords0[i]] = max - 2 * i
    if keywords1[i] in score:
        score[keywords1[i]] += max - 2 * i
    else:
        score[keywords1[i]] = max - 2 * i

sorted_score = sorted(score.items(), key=operator.itemgetter(1))
sorted_score.reverse()
keywords_combined = []
for tup in sorted_score:
    keywords_combined.append(tup[0])
keywords_combined = keywords_combined[0 : int(topK / 2)]
print('\nkeywords combined:')
print(keywords_combined)