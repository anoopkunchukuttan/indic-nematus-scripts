import sys, codecs

writer=codecs.getwriter('utf-8')(sys.stdout)

for line in codecs.getreader('utf-8')(sys.stdin): 

    fields=line.split(u'|||')
    fields.insert(2,u' Distortion0= -1 LM0= -1 WordPenalty0= -1 PhrasePenalty0= -1 TranslationModel0= -1 -1 -1 -1 ')

    writer.write(u'|||'.join(fields))

