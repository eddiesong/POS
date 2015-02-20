UNI: ps2839
run time: 60.6s for Solution A and 354.6s for Solution B, so it's near 7 mins for both


Solution A

perplexity of A2.uni.txt: 1104.83292814
perplexity of A2.bi.txt: 57.2215464238
perplexity of A2.tri.txt: 5.89521267642

If we have more information in a language model, we have lower perplexity.

perplexity of A3.txt: 13.0759217039

As a linear interpolated model, it combines unigram, bigram and trigram model and gives them equal weights. It does significantly well compares to unigram and bigram.

perplexity of Sample1_scored.txt:11.6492786046
perplexity of Sample2_scored.txt:1611241155.03

Probably all the sentence in Sample2 receive a -1000 log based probability, so it indicates that Sample 1 is an excerpt of the Brown dataset. The tokens from Sample2 are nearly all unseen tokens.


Solition B

pos.py output for B5.txt: 93.6359584096 (the reference implementation get 93.7008827776)

It's pretty close to the reference implementation, maybe it's because different way of treating corner case.

pos.py output for B6.txt: 95.3123637315 (the reference implementation get 96.9354729304)

It received a higher score compares to the HMM tagger, maybe it's because it is a combination of Unigram, Bigram and Trigram taggers. When tagging fails, it can back off to another tagger.

