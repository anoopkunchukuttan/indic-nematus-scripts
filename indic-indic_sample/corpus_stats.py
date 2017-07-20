import codecs, sys
import numpy as np

with codecs.open(sys.argv[1],'r','utf-8') as infile: 

    word_set=set()
    sent_len=[]
    wc=0

    for line in infile: 
        words=line.strip().split(u' ')
        wc+=len(words)
        sent_len.append(len(words))
        word_set.update(words)

    sent_len_arr=np.array(sent_len)
    labels = [ 'Max', 'Min', 'Mean', 'Median' ]
    sent_stats = [ np.max(sent_len_arr), np.min(sent_len_arr), np.mean(sent_len_arr), np.median(sent_len_arr) ]

    print 'Number of words: {}'.format(wc)
    print 'Vocabulary size: {}'.format(len(word_set))
    print 'Sentence statistics' 
    for l,s in zip(labels,sent_stats): 
        print '{}: {}'.format(l,s)

    print 'Number of sentences: {}'.format(len(sent_len_arr))
    print 'Number of sentences <= 50: {}'.format(np.sum(sent_len_arr<=50))
    print 'Number of sentences <= 80: {}'.format(np.sum(sent_len_arr<=80))
    print 'Number of sentences <= 100: {}'.format(np.sum(sent_len_arr<=100))
