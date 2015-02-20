import nltk
import math

#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
def calc_probabilities(brown):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    unigram = {}
    bigram = {}
    trigram = {}

    uni_count = 0

    for sentence in brown:
        sentence += ' STOP'
        tokens = nltk.word_tokenize(sentence)

        # build unigram dictionary
        for word in tokens:
            uni_count += 1
            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1

        # build bigram dictionary, it should add a '*' to the beginning of the sentence first
        tokens = ['*'] + tokens
        bigram_tuples = tuple(nltk.bigrams(tokens))
        # bicount = dict(Counter(bigram_tuples))
        # fdist = nltk.FreqDist(bi)
        for item in bigram_tuples:
            if item in bigram:
                bigram[item] += 1
            else:
                bigram[item] = 1

        # build trigram dictionary, it should add another '*' to the beginning of the sentence
        tokens = ['*'] + tokens
        trigram_tuples = tuple(nltk.trigrams(tokens))
        # tricount = dict(Counter(trigram_tuples))
        # fdist = nltk.FreqDist(tri)
        for item in trigram_tuples:
            if item in trigram:
                trigram[item] += 1
            else:
                trigram[item] = 1

    # calculate unigram probability
    for word in unigram:
        temp = [word]
        unigram_p[tuple(temp)] = math.log(float(unigram[word])/uni_count, 2)

    # calculate bigram probability
    for word in bigram:
        if word[0] == '*':
            bigram_p[tuple(word)] = math.log(float(bigram[word])/unigram[('STOP')], 2)
        else:
            bigram_p[tuple(word)] = math.log(float(bigram[word])/unigram[word[0]], 2)

    # calculate trigram probability
    for word in trigram:
        if word[0] == '*' and word[1] == '*':
            trigram_p[tuple(word)] = math.log(float(trigram[word])/unigram[('STOP')], 2)
        else:
            trigram_p[tuple(word)] = math.log(float(trigram[word])/bigram[(word[0], word[1])], 2)
    
    #print unigram_p
    #print bigram_p
    #print trigram_p
    return unigram_p, bigram_p, trigram_p

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):
    scores = []

    for sentence in data:
        total_score = 0
        #calculate unigram
        if n == 1:
            sentence += ' STOP'
            tokens = nltk.word_tokenize(sentence)
            for word in tokens:
                #print ngram_p[tuple([word])]
                if tuple([word]) in ngram_p:
                    total_score += ngram_p[tuple([word])]
            scores.append(total_score)
        
        #calculate bigram
        if n == 2:
            sentence = '* ' + sentence + ' STOP'
            tokens = nltk.word_tokenize(sentence)
            for word1,word2 in zip(tokens[0::1], tokens[1::1]):
                #print ngram_p[(word1, word2)]
                if (word1, word2) in ngram_p:
                    total_score += ngram_p[(word1, word2)]
            scores.append(total_score)

        #calculate trigram
        if n == 3:
            sentence = '* * ' + sentence + ' STOP'
            tokens = nltk.word_tokenize(sentence)
            for word1,word2,word3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
                #print ngram_p[(word1, word2, word3)]
                if (word1, word2, word3) in ngram_p:
                    total_score += ngram_p[(word1, word2, word3)]
            scores.append(total_score)

    return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):
    scores = []

    for sentence in brown:
        total_score = 0
        mark = 0

        sentence = '* * ' + sentence + ' STOP'
        tokens = nltk.word_tokenize(sentence)

        # for all the (word1, word2, word3) tuple in sentence, calculate probabilities
        for word1, word2, word3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
            word_score = 0

            # the first tuple is ('*', '*', WORD), so we begin unigram with word3
            uni_score = 0
            if tuple([word3]) in unigrams:
                uni_score = 2**unigrams[tuple([word3])]

            bi_score = 0
            if (word2, word3) in bigrams:
                bi_score = 2**bigrams[(word2, word3)]

            tri_score = 0
            if (word1, word2, word3) in trigrams:
                tri_score = 2**trigrams[(word1, word2, word3)]

            # if all the unigram, bigram, trigram scores are 0 then the sentence's probability should be -1000
            if uni_score != 0 or bi_score != 0 or tri_score != 0:
                word_score = math.log((uni_score + bi_score + tri_score), 2) + math.log(1, 2) - math.log(3, 2)
                total_score += word_score
            else:
                mark = 1

        if mark == 1:
            total_score = -1000

        scores.append(total_score)

    return scores

def main():
    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)

    #question 1 output
    q1_output(unigrams, bigrams, trigrams)

    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

    #open Sample1 and Sample2 (question 5)
    infile = open('Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open('Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, 'Sample1_scored.txt')
    score_output(sample2scores, 'Sample2_scored.txt')

if __name__ == "__main__": main()
