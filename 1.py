import sys
import nltk
import math

from nltk.corpus import brown

#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []
    count = {}

    # Iterate through all tokens of the wbrowns and create a dictionary to count
    # the frequency of each token
    for token in wbrown:
        if token in count:
            count[token] += 1
        else:
            count[token] = 1

    # Only output the tokens which have frequency > 5
    for key in count:
        if count[key] > 5:
            knownwords.append(key)

    return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    rare = []
    sentence = []
    knownwords_set = set(knownwords)

    # Replace the tokens in the brown that doesn't appear in "knownwords" 
    # with "_Rare_"
    for token in brown:
        if token in knownwords_set:
            sentence.append(token)
        else:
            sentence.append("_RARE_")

        # "STOP" indicates the end of a sentence, rare is a list that contains a
        # list of tokens
        if token == "STOP":
            rare.append(sentence)
            sentence = []
            # print sentence

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
    bigram_count = {}
    trigram_count = {}

    brown_len = 0

    # Tokenize the tbrown as bigram and trigram tuples since trigram probability
    # depends on bigram count
    bigram_tuple = tuple(nltk.bigrams(tbrown))
    trigram_tuple = tuple(nltk.trigrams(tbrown))

    # Count the frequency of trigram tuple
    for tup in trigram_tuple:
        # (?, "STOP", "*") and ("STOP", "*", "*") are not valid trigram tuples
        if tup[1:3] != ("STOP", "*") and tup[1:3] != ("*", "*"):
            # print tup
            if tup in trigram_count:
                trigram_count[tup] += 1
            else:
                trigram_count[tup] = 1

    # Count the frequency of bigram tuple
    for tup in bigram_tuple:
        if tup == ("STOP", "*"):
            continue # skip this tuple

        if tup == ('*', '*'):
            brown_len += 1
            continue # skip to the next tuple
        elif tup in bigram_count:
            bigram_count[tup] += 1
        else:
            bigram_count[tup] = 1

    # Calculate trigram probabilty based on bigram count
    for tup in trigram_count:
        if tup[0:2] == ('*', '*'):
            qvalues[tup] = math.log(trigram_count[tup]/float(brown_len),2)
        else:
            qvalues[tup] = math.log(trigram_count[tup]/float(bigram_count[tup[0:2]]),2)


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
    ecount = {}
    taglist = []
    tag_list_r = []
    tag_sentence = []
    tag_count = {}


    for tag in tbrown:
        # Since wbrown is a list that contains a lot of words that form the sentence
        # we format the taglist in the same way
        tag_sentence.append(tag)
        if tag == "STOP":
            taglist.append(tag_sentence)
            tag_sentence = []

        # Counting the frequency of tags at the same time
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1 

    # Extract the total possible tags from the tag_count dictionary
    for tag in tag_count:
        tag_list_r.append(tag)

    # print tag_list_r

    # Iterate through the sentence and its tags at the same time to produce the
    # count for calculating emission probability
    for sentence, tag_sentence in zip(wbrown, taglist):
        for word, tag in zip(sentence, tag_sentence):
            if (word, tag) in ecount:
                ecount[(word, tag)] += 1
            else:
                ecount[(word, tag)] = 1

    # Computing emission probability
    for tup in ecount:
        evalues[tup] = math.log(ecount[tup]/float(tag_count[tup[1]]), 2)

    return evalues, tag_list_r

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
    brown_rare = []

    for sentence in brown:
        sentence2 = replace_rare(sentence, knownwords)
        sentence2[0].pop()      # get rid of the "STOP" at the end of the sentence
        brown_rare += sentence2
        # brown_rare.append(rare)

    # print brown_rare

    tag_dic = {}
    tag_index = 0

    # Eliminate the "STOP" and "*" tags from the taglist
    for tag in taglist:
        if tag != "STOP" and tag != "*":
            tag_dic[tag_index] = tag
            tag_index += 1

    # Viterbi initialization

    sentence_index = 0

    for sentence in brown_rare:
        # Initialize the dictionary for viterbi algorithm
        pi = {}
        bp = {}

        pi[0, '*', '*'] = 1

        # k is the index of the word, 0 is the ('*','*')
        # iterate through all the words in a sentence
        for k in range(1, len(sentence)+1):
            if k == 1:
                for c in range(len(tag_dic)):
                    if ('*', '*', tag_dic[c]) in qvalues:
                        pi[k, '*', tag_dic[c]] = qvalues[('*', '*', tag_dic[c])]
                    else:
                        pi[k, '*', tag_dic[c]] = -1000

                    if (sentence[k-1], tag_dic[c]) in evalues:
                        pi[k, '*', tag_dic[c]] += evalues[(sentence[k-1], tag_dic[c])]
                    else:
                        pi[k, '*', tag_dic[c]] = -1000
            elif k == 2:
                for b in range(len(tag_dic)):
                    for c in range(len(tag_dic)):
                        pi[k, tag_dic[b], tag_dic[c]] = pi[k-1, '*', tag_dic[b]]
                        if ('*', tag_dic[b], tag_dic[c]) in qvalues:
                            pi[k, tag_dic[b], tag_dic[c]] += qvalues[('*', tag_dic[b], tag_dic[c])]
                        else:
                            pi[k, tag_dic[b], tag_dic[c]] += -1000

                        if (sentence[k-1], tag_dic[c]) in evalues:
                            pi[k, tag_dic[b], tag_dic[c]] += evalues[(sentence[k-1], tag_dic[c])]
                        else:
                            pi[k, tag_dic[b], tag_dic[c]] += -1000
            else:
                for b in range(len(tag_dic)):
                    for c in range(len(tag_dic)):
                        for a in range(len(tag_dic)):
                            sum_p = pi[k-1, tag_dic[a], tag_dic[b]]
                            if (tag_dic[a], tag_dic[b], tag_dic[c]) in qvalues:
                                sum_p += qvalues[(tag_dic[a], tag_dic[b], tag_dic[c])]
                            else:
                                sum_p += -1000

                            if (sentence[k-1], tag_dic[c]) in evalues:
                                sum_p += evalues[(sentence[k-1], tag_dic[c])]
                            else:
                                sum_p += -1000

                            if (k, tag_dic[b], tag_dic[c]) not in pi:
                                pi[k, tag_dic[b], tag_dic[c]] = sum_p
                                bp[k, tag_dic[b], tag_dic[c]] = tag_dic[a]
                            elif pi[k, tag_dic[b], tag_dic[c]] < sum_p:
                                pi[k, tag_dic[b], tag_dic[c]] = sum_p
                                bp[k, tag_dic[b], tag_dic[c]] = tag_dic[a]

        # print pi, bp
        # print sentence

        # Termination Step
        final_p = float('-inf')
        back_tag = []
        for b in range(len(tag_dic)):
            for c in range(len(tag_dic)):
                sum_p = pi[len(sentence), tag_dic[b], tag_dic[c]]
                if (tag_dic[b], tag_dic[c], "STOP") in qvalues:
                    sum_p += qvalues[(tag_dic[b], tag_dic[c], "STOP")]
                else:
                    sum_p += -1000

                if final_p < sum_p:
                    final_p = sum_p
                    back_tag = [tag_dic[b], tag_dic[c]]

        # Reproduce the tag sequence
        for tag_index in range(len(sentence), 2, -1):
            previous_tag = bp[tag_index, back_tag[0], back_tag[1]]
            back_tag.insert(0, previous_tag)

        sentence_tag = ""
        for word, tag in zip(brown[sentence_index][:-1], back_tag):
            sentence_tag += word + "/" + tag + " "
        tagged.append(sentence_tag) 
        sentence_tag = ""

        sentence_index += 1



    return tagged

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
        outfile.write("\n")
    outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of tagged sentences the WORD/TAG format. Each sentence is a string with a terminal newline rather than a list of tokens.
def nltk_tagger(brown1):
    tagged = []

    training=brown.tagged_sents(tagset = 'universal')

    #Create Unigram, Bigram, Trigram taggers based on the training set.
    unigram_tagger = nltk.UnigramTagger(training)
    bigram_tagger = nltk.BigramTagger(training)
    trigram_tagger = nltk.TrigramTagger(training)

    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    doc = []
    for sentence in brown1:
        sentence_tag = trigram_tagger.tag(sentence)
        doc.append(sentence_tag)

    tagged_sen = []
    for sentence in doc:
        for tup in sentence:
            word_tag = tup[0] + "/" + tup[1]
            tagged_sen.append(word_tag)
        tagged.append(tagged_sen)
        tagged_sen = []

    # print tagged

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

    word_index = 0 

    # Seperate tags and tokens each each token
    for sentence in brown_train:
        # Contruct the sentence as if we are using trigram
        sentence = "*/* */* " + sentence + " STOP/STOP"
        # Split each token/tag pair by space first
        word_tags = sentence.split(" ")
        # iterate through each token/tag pair
        for token in word_tags:
            # Now split the token/tag pair by "/"
            word_tag = token.split('/')
            # if the array word_tag is > 1, then it is a valid token/tag pair
            if len(word_tag) > 1:
                # First, add the first entry of word_tag in the word list
                wbrown.append(word_tag[0])
                # Only the last entry of the word_tag array is the tag, anything
                # before that is part of the word
                for index in range(1, len(word_tag)):
                    if index == (len(word_tag)-1):
                        tbrown.append(word_tag[index])
                    else:
                        wbrown[word_index] =  wbrown[word_index] + '/' + word_tag[index]
            # print word_index
                word_index += 1 # increament to count the number of words in the text

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
    # Tokenize the brown_dev text
    brown_dev_token = []
    brown_dev_sen = []
    tokens_tri = []
    for sentence in brown_dev:
        sentence = sentence + " STOP"
        tokens_tri += nltk.word_tokenize(sentence)

    # For testing purpose
    # for index in range(0, len(brown_dev)):
    #     if (index % 5) == 0:
    #         sentence = brown_dev[index]
    #         sentence = sentence + " STOP"
    #         tokens_tri += nltk.word_tokenize(sentence)

    for token in tokens_tri:
        brown_dev_token.append(token)
        if token == "STOP":
            brown_dev_sen.append(brown_dev_token)
            brown_dev_token = []

    # print brown_dev_sen

    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev_sen, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    #do nltk tagging here
    brown_dev_token2 = []
    brown_dev_sen2 = []
    tokens_tri2 = []
    for sentence in brown_dev:
        sentence = sentence + " STOP"
        tokens_tri2 += nltk.word_tokenize(sentence)

    for token in tokens_tri2:
        if token == "STOP":
            brown_dev_sen2.append(brown_dev_token2)
            brown_dev_token2 = []
        else:
            brown_dev_token2.append(token)

    nltk_tagged = nltk_tagger(brown_dev_sen2)

    #question 6 output
    q6_output(nltk_tagged)
if __name__ == "__main__": main()
