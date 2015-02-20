import sys
import nltk
import math
import string
from collections import Counter
from nltk.corpus import brown

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []

    temp = []

    for sentence in wbrown:
        temp += sentence

    dictionary = dict(Counter(temp))

    for key, value in dictionary.iteritems():
        if value > 5:
            knownwords.append(key)

    # print knownwords

    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    rare = []

    for sentence in brown:
        temp = []
        for word in sentence:
            if word in knownwords:
                temp.append(word)
            else:
                temp.append('_RARE_')
        rare.append(temp)

    # print rare

    return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')

    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    qvalues = {}

    unigram = {}
    bigram = {}
    trigram = {}

    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    li_uni = []
    li_bi = []
    li_tri = []

    uni_count = 0

    for sentence in tbrown:
        li_uni = sentence[2:]
        li_bi = sentence[1:]
        li_tri = sentence

        #calculate unigram
        for word in li_uni:
            uni_count += 1
            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1

        #calculate bigram
        bigram_tuples = tuple(nltk.bigrams(li_bi))
        for item in bigram_tuples:
            if item in bigram:
                bigram[item] += 1
            else:
                bigram[item] = 1

        #calculate trigram
        trigram_tuples = tuple(nltk.trigrams(li_tri))
        for item in trigram_tuples:
            if item in trigram:
                trigram[item] += 1
            else:
                trigram[item] = 1

    for word in unigram:
        temp = [word]
        unigram_p[tuple(temp)] = math.log(float(unigram[word])/uni_count, 2)

    for word in bigram:
        if word[0] == '*':
            bigram_p[tuple(word)] = math.log(float(bigram[word])/unigram[('STOP')], 2)
        else:
            bigram_p[tuple(word)] = math.log(float(bigram[word])/unigram[word[0]], 2)

    for word in trigram:
        if word[0] == '*' and word[1] == '*':
            trigram_p[tuple(word)] = math.log(float(trigram[word])/unigram[('STOP')], 2)
        else:
            trigram_p[tuple(word)] = math.log(float(trigram[word])/bigram[(word[0], word[1])], 2)

    qvalues = trigram_p
   
    return qvalues

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    evalues = {}
    taglist = []

    tagcount = {}
    wordtag = {}

    for sentence, tags in zip(wbrown, tbrown):
        for word, tag in zip(sentence, tags):
            if tag in tagcount:
                tagcount[tag] += 1
            else:
                tagcount[tag] = 1

            if (word, tag) in wordtag:
                wordtag[(word, tag)] += 1
            else:
                wordtag[(word, tag)] = 1

    for (word, tag) in wordtag:
        # print word, tag
        prob = math.log(float(wordtag[(word, tag)])/tagcount[tag], 2)
        evalues[(word, tag)] = prob

    for tag in tagcount:
        taglist.append(tag)

    # print evalues, taglist

    return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams
#evalues is from the return of calc_emissions()
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    tagged = []
    
    for sentence in brown:
    
        # initialization
        pi = {}
        bp = {}

        tags = []

        temp_sentence = list(sentence)
        result = ''

        # replace all the low frequency tag with _RARE_
        for k in range(len(sentence)):
            if sentence[k] not in knownwords:
                sentence[k] = '_RARE_'

        pi[(0, '*', '*')] = 0
        pi[(1, '*', '*')] = 0

        for k in range(2, len(sentence) - 1):

            # first word of the sentence, the tuple is ('*', '*', v), only iterate over v
            if k == 2:

                for v in taglist:
                    prob = -1000
                    tup = ('*', '*', v) 
                    wordtag = (sentence[k], v)

                    if tup in qvalues and wordtag in evalues and qvalues[tup] + evalues[wordtag] >= prob:
                        prob = qvalues[tup] + evalues[wordtag]

                    bp[(k, '*', v)] = '*'                 
                    pi[(k, '*', v)] = prob

            # secend word of the sentence, the tuple is ('*', u, v), iterate over u and v
            elif k == 3:

                for u in taglist:
                    for v in taglist:
                        trace = taglist[0]
                        prob = -1000

                        wordtag = (sentence[k], v)

                        tup = ('*', u, v)

                        if tup in qvalues and wordtag in evalues and (k - 1, '*', u) in pi:
                            temp = pi[(k - 1, '*', u)] + qvalues[tup] + evalues[wordtag]
                            if temp >= prob:
                                prob = temp
                                trace = '*'

                        bp[(k, u, v)] = trace                 
                        pi[(k, u, v)] = prob

            # otherwise, iterate over w, u and v
            else:
                for u in taglist:
                    for v in taglist:
                        trace = taglist[0]
                        prob = -1000

                        wordtag = (sentence[k], v)

                        for w in taglist:
                             
                            tup = (w, u, v)

                            if tup in qvalues and wordtag in evalues and (k - 1, w, u) in pi:
                                temp = pi[(k - 1, w, u)] + qvalues[tup] + evalues[wordtag]
                                if temp >= prob:
                                    prob = temp
                                    trace = w

                        bp[(k, u, v)] = trace                 
                        pi[(k, u, v)] = prob
            
        # find last two tags of the sentence
        temp = -1000

        for u in taglist:
            for v in taglist:
                tup = (u, v, 'STOP')
                if tup in qvalues and qvalues[tup] + pi[(len(sentence) - 2, u, v)] >= temp:
                    temp = qvalues[tup] + pi[(len(sentence) - 2, u, v)]
                    max_tup = tup

        # initialize the tags list
        tags = [None] * len(sentence)

        tags[0] = max_tup[2]
        tags[1] = max_tup[1]
        tags[2] = max_tup[0]

        # work backward with back pointer to find best tags
        for k in range(len(sentence) - 5):
            tags[k + 3] = bp[len(sentence) - 2 - k, tags[k + 2], tags[k + 1]]

        # attach words with tags
        for k in range(len(sentence) - 3):
            result += temp_sentence[k + 2] + '/' + tags[len(sentence) - 3 - k] + ' '

        result = result[:-1]
        result += '\n'

        # print result
        tagged.append(result)

    return tagged

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of lists of tokens in the WORD/TAG format.
def nltk_tagger(brown):
    tagged = []


    return tagged

def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n'
        outfile.write(output)
    outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
    wbrown = []
    tbrown = []

    for sentence in brown_train:
        sentence = '*/* */* ' + sentence + ' STOP/STOP'
        tokens = sentence.split()
        #print tokens

        ws = []
        ts = []

        for item in tokens:
            loc = item.rfind('/')
            ws.append(item[:loc])
            ts.append(item[loc+len('/'):].upper())

        wbrown.append(ws)
        tbrown.append(ts)

    # print wbrown
    # print tbrown

    return wbrown, tbrown

def main():
    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)

    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)

    #question 2 output
    q2_output(qvalues)

    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)

    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)

    #question 3 output
    q3_output(wbrown_rare)

    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)

    #question 4 output
    q4_output(evalues)

    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    #format Brown development data here
    li = []

    for sentence in brown_dev:
        sentence = '* * ' + sentence + ' STOP'
        tokens = nltk.word_tokenize(sentence)
        li.append(tokens)
    
    brown_dev = li

    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    #do nltk tagging here
    

    nltk_tagged = nltk_tagger(brown_dev)

    #question 6 output
    q6_output(nltk_tagged)
if __name__ == "__main__": main()