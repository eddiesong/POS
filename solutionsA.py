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
        #print sentence
        tokens = nltk.word_tokenize(sentence)
        #print tokens
        for word in tokens:
            uni_count += 1
            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1

        #find bigram probability
        tokens = ['*'] + tokens
        bi = nltk.bigrams(tokens)
        fdist = nltk.FreqDist(bi)
        for word,value in fdist.items():
            if word in bigram:
                bigram[word] += 1
            else:
                bigram[word] = 1

        #find trigram probability
        tokens = ['*'] + tokens
        tri = nltk.trigrams(tokens)
        fdist = nltk.FreqDist(tri)
        for word,value in fdist.items():
            if word in trigram:
                trigram[word] += 1
            else:
                trigram[word] = 1
    
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
                total_score += ngram_p[tuple([word])]
            scores.append(total_score)
        
        #calculate bigram
        if n == 2:
            sentence = '* ' + sentence + ' STOP'
            tokens = nltk.word_tokenize(sentence)
            for word1,word2 in zip(tokens[0::1], tokens[1::1]):
                #print ngram_p[(word1, word2)]
                total_score += ngram_p[(word1, word2)]
            scores.append(total_score)

        #calculate trigram
        if n == 3:
            sentence = '* * ' + sentence + ' STOP'
            tokens = nltk.word_tokenize(sentence)
            for word1,word2,word3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
                #print ngram_p[(word1, word2, word3)]
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
        uni_score = 0
        bi_score = 0
        tri_score = 0
        total_score = 0

        #calculate unigram
        sentence += ' STOP'
        tokens = nltk.word_tokenize(sentence)
        for word in tokens:
            #print unigrams[tuple([word])]
            uni_score += unigrams[tuple([word])]
        total_score += float(1)/3 * uni_score

        #calculate bigram
        tokens = ['*'] + tokens
        for word1,word2 in zip(tokens[0::1], tokens[1::1]):
            #print bigrams[(word1, word2)]
            bi_score += bigrams[(word1, word2)]
        total_score += float(1)/3 * bi_score

        #calculate trigram
        tokens = ['*'] + tokens
        for word1,word2,word3 in zip(tokens[0::1], tokens[1::1], tokens[2::1]):
            #print trigrams[(word1, word2, word3)]
            tri_score += trigrams[(word1, word2, word3)]
        total_score += float(1)/3 * tri_score

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
