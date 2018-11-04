#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:07:36 2018

@Name    : Xiaotian Zhan
@Uniqname: zhanxia
 
"""
#-----------------------------------#
#--- Import and Global Variables ---#
#-----------------------------------#
import re
import sys
#import pdb
import collections
import numpy as np
Debug = 0
if Debug:
    Trainfile = "/home/billzhan/OneDrive/Academic/Umich/2018Fall/SI561/asgmt3/POS.train" 
    Testfile = "/home/billzhan/OneDrive/Academic/Umich/2018Fall/SI561/asgmt3/POS.test"
    Outputfile = "/home/billzhan/OneDrive/Academic/Umich/2018Fall/SI561/asgmt3/POS.test.out"
else:
    Trainfile = sys.argv[1]
    Testfile = sys.argv[2]
    Outputfile = "POS.test.out"
#-------------#
#--- Class ---#
#-------------#

class ViterbiTagger(object):
    def __init__(self, trainfile, testfile):
        # files
        self.trainfile = trainfile
        self.testfile = testfile
        # train related data
        self.train_results = self.read_file(trainfile)
        self.train_pairs = self.train_results[0]
        self.tags_list = self.train_results[-1]
        # test related
        self.test_results = self.read_file(testfile)
        self.test_pairs = self.test_results[0]
        self.test_rawwords = self.test_results[-2]
        # global variables (from train)
        self.word2ix_dict, self.tag2ix_dict = self.train_results[1], self.train_results[2]  
        self.word2ix_dict['UNK'] = 0  #add a token 'UNK' in the begining
        self.ix2tag_dict = {v:k for k,v in self.tag2ix_dict.items()}
        self.uniqwords = list(self.word2ix_dict.keys())  #in insertion order
        self.uniqtags = list(self.tag2ix_dict.keys())  #in insertion order
        self.sentence_counts = len(self.train_pairs)  #number of BOS
        self.W, self.T = len(self.uniqwords), len(self.uniqtags)
        self.tagunigram_counts = self.count_tagunigram()
        # 3 probability distribution
        self.initial_vec = self.cal_init_states()
        self.transition_mat = self.cal_transition() 
        self.emission_mat = self.cal_emission()
    
    #--- preparations
    # read files
    def read_file(self, file):
        """
        read file and split into words and tags
        ---------------------------------------
        @return results: tuple 3 lists: (word,tag) pairs by sentence, words, tags
        """
        pairs_list, raw_sentences, word_toix, tag_toix, tags_list = [], [], {}, {}, []
        wix, tix = 1, 0  #index zero in word2ix remaining for UNK
        with open(file) as f:
            for line in f.readlines():
                if re.search(r'\\|\.\.\./|//',line):  #bad input
                    continue
                if re.search(r'/\w+&\w+/', line):  #some bad formatted &
                    line = re.sub(r'&', ' ', line)
                if re.search(r'/\w+-\w+/', line):  #some bad formatted -
                    line = re.sub(r'-', ' ', line)
                line_content = line.strip().split(' ')
                wordtag_list = [(token.split('/')[0],token.split('/')[1]) 
                                for token in line_content if re.search(r'/', token)]  #store word/tag into a tuple
                line_rawwords = [t[0] for t in wordtag_list]
                raw_sentences.append(line_rawwords)
                line_words = [t[0].lower() for t in wordtag_list]  #append all words(lowered) to words_list
                line_tags = [t[1].split('|')[0] if re.search(r'|',t[1]) else t[1]  
                                    for t in wordtag_list]  #append all tags to tags_list, if encounter |, pick first one
                tags_list.extend(line_tags)
                for word in set(line_words):
                    if word_toix.get(word) is None:
                        word_toix[word] = wix
                        wix += 1
                for tag in set(line_tags):
                    if tag_toix.get(tag) is None:  # dont use not xx, because we have value zero
                        tag_toix[tag] = tix  
                        tix += 1
                pairs_list.append(list(zip(line_words, line_tags)))  #zip the two processed lists
        results = (pairs_list, word_toix, tag_toix, raw_sentences,tags_list)
        return results
    # get word index
    def get_wordix(self, word):
        return self.word2ix_dict.setdefault(word, 0)
    
    # calculate initial states(tags) distribution
    def cal_init_states(self):
        """
        calculate probability distribution for initial states, tags at first positions here
        -----------------------------------------------------------------------------------
        @param train_contents: parsed training data, we have (word, tag) tuples
        
        @return PI: dictionary where keys sorted in alphabetic order, values are initial probability
        """
        data = self.train_pairs
        firsttag_list = [s[0][1] for s in data]  #s[0][1] means the first pair, the second element, i.e. the tag
        firsttag_counter = collections.Counter(firsttag_list)
        # calculate initial probability distribution
        first_counts = np.zeros(self.T)
        for tag in self.uniqtags:
            # add one smoothing
            if firsttag_counter.get(tag):
                counts = firsttag_counter.get(tag)+1
            else:
                counts = 1
            first_counts[self.tag2ix_dict[tag]] = counts
#            pr = counts/(self.sentence_counts+self.T)
#            PI[tag] = pr
        PI = first_counts/(self.sentence_counts+self.T)

        return PI
    
    # count tag unigrams
    def count_tagunigram(self):
        tags_counter = collections.Counter(self.tags_list)  #c(ti)
        counts_vector = np.zeros([self.T,1])
        for tag in self.uniqtags:
            ix = self.tag2ix_dict[tag]
            counts_vector[ix,0] = tags_counter[tag]
#        counts_vector = np.array([tags_counter[tag] for tag in self.uniqtags]).reshape([self.T,1])  #generate a column vector of c(ti)        
        return counts_vector

    # calculate transition matrix of tags
    def cal_transition(self):
        """
        calculate tags transition matrix, i.e., pr(ti|ti-1), using add-one smooth
        -------------------------------------------------------------------------
        @return transition_probs: (T,T) np array, (i,j) is p(tj|ti)
        """
        # count bigrams
        tagbigram_counts = np.zeros([self.T,self.T])
        for row in self.train_pairs:
            tagindex_inrow = [self.tag2ix_dict.get(pair[1]) for pair in row]  #get the index of tags in this row
            bigrams1_ix = [tagindex_inrow[i] for i in range(len(row)-1)]  #index of 1st ele in bigram
            bigrams2_ix = [tagindex_inrow[j] for j in range(1,len(row))]  #index of 2nd ele in bigram
#            pdb.set_trace()
            tagbigram_counts[bigrams1_ix,bigrams2_ix] += 1  #all (i,j) entries plus one, i from 1_ix, j from 2_ix

        tagsunigram_mat = np.repeat(self.tagunigram_counts,self.T,axis=1)
        transition_probs = np.divide((tagbigram_counts+1),(tagsunigram_mat+self.T))
        
        return transition_probs
    
    # calculate emission matrix from tag to word
    def cal_emission(self):
        """
        calculate tag to word emission matrix, i.e., pr(Ww|Tt) using add-one smooth
        ---------------------------------------------------------------------------
        @return emission_probs: (T,W) matrix, (i,j) is p(wi|tj)
        """
        tagtoword_counts = np.zeros([self.T,self.W])
        for row in self.train_pairs:
            tagindex_inrow = [self.tag2ix_dict.get(pair[1]) for pair in row]
            wordindex_inrow = [self.word2ix_dict.get(pair[0]) for pair in row]
            tagtoword_counts[tagindex_inrow,wordindex_inrow] += 1
#        pdb.set_trace()
        tagsunigram_mat = np.repeat(self.tagunigram_counts,self.W,axis=1)
        emission_probs = np.divide((tagtoword_counts+1),(tagsunigram_mat+self.W))
        
        return emission_probs
    
    #--- tagging methods
    def viterbi_tagging(self, sentence):
        """
        POS using viterbi algorithm
        ---------------------------
        @param sentence: a list of words in a sentene
        
        @return pos_list: a list of corresponding tags 
        """
        # initialization step
        W = len(sentence)  #number of words in sentence
        w1_ix = self.get_wordix(sentence[0])
        Score, BackPtr = np.zeros([self.T,W]), np.zeros([self.T,W])
#        pdb.set_trace()
        Score[:,0] = np.multiply(self.emission_mat[:,w1_ix],self.initial_vec)
        BackPtr[:,0] = 0
        # interation step
        for i in range(1,W):
            cur_ix = self.get_wordix(sentence[i])  #w word index in our dictionary
            Score_wprev = np.repeat(Score[:,i-1].reshape([self.T,1]),self.T,axis=1)  #tiled col vector of Score with previous word 
            wprev_times_trans = np.multiply(Score_wprev, self.transition_mat)  #each column is Score[:,w-1] dot mul trans[:,j]
            max_vec = np.max(wprev_times_trans, axis=0)
            argmax_vec = np.argmax(wprev_times_trans, axis=0)
#            pdb.set_trace()
            Score[:,i] = np.multiply(self.emission_mat[:,cur_ix], max_vec.T)
            BackPtr[:,i] = argmax_vec.T
        # sequence identification
        Seq = [0 for _ in range(W)]
        Seq[-1] = np.argmax(Score[:,-1])
        loop_range = list(range(W-1))[::-1]
        for w in loop_range:  #loop in reversed order
#            pdb.set_trace()
            Seq[w] = int(BackPtr[Seq[w+1],w+1])
        
        pos_list = [self.ix2tag_dict[t] for t in Seq] 
        
        return pos_list
    def baseline_tagging(self, sentence):
        wordix_list = [self.get_wordix(word) for word in sentence]
        Seq = [np.argmax(self.emission_mat[:,w]) for w in wordix_list]
        pos_list = [self.ix2tag_dict[t] for t in Seq]
        return pos_list
    
    # make prediction
    def predict(self, output_file,test_pairs=None, cal_acc=True):
        # initialize
        test_data = test_pairs or self.test_pairs
        sys_correct = bsl_correct = test_size = 0
        # predict every sentence
        with open(output_file,'w') as f:
            for i in range(len(test_data)):  #each row in the data, (word, tag)
                row = test_data[i]
                # system predict and calculate accuracy
                row_len = len(row)
                sentence = [pair[0] for pair in row]
                gold_tags = [pair[1] for pair in row]
                pred_tags = self.viterbi_tagging(sentence)
                bsl_tags = self.baseline_tagging(sentence)
                sys_row_correct = len([i for i in range(row_len) if pred_tags[i]==gold_tags[i]])
                bsl_row_correct = len([i for i in range(row_len) if bsl_tags[i]==gold_tags[i]])
                sys_correct += sys_row_correct
                bsl_correct += bsl_row_correct
                test_size += row_len
                # write to file
                raw_words = self.test_rawwords[i]
                line_content = ' '.join([raw_words[i]+'/'+pred_tags[i] for i in range(row_len)])
                f.writelines(line_content+'\n')
        print('Prediction output file {} is created'.format(output_file))
        if cal_acc:
            sys_acc = sys_correct/test_size
            bsl_acc = bsl_correct/test_size
            print('Tag-level accuracy of [baseline] is: {}%'.format(round(bsl_acc*100,3)))
            print('Tag-level accuracy of [this system] is: {}%'.format(round(sys_acc*100,3)))
            return sys_acc
        return

#--- debug
#viterbi_tagger = ViterbiTagger(trainfile=Trainfile, testfile=Testfile)
#test_sentence0 = [pair[0] for pair in viterbi_tagger.test_pairs[0]]
#test_tags0 = [pair[1] for pair in viterbi_tagger.test_pairs[0]]
#correct = 0
#pred_tags0 = viterbi_tagger.viterbi_tagging(test_sentence0)    
#for i in range(len(pred_tags0)):
#    if pred_tags0[i]==test_tags0[i]:
#        correct += 1  
#sys_acc = viterbi_tagger.predict(output_file=Outputfile)

#------------#
#--- Main ---#
#------------#
def main():
    viterbi_tagger = ViterbiTagger(trainfile=Trainfile, testfile=Testfile)
    sys_acc = viterbi_tagger.predict(output_file=Outputfile)

if __name__=='__main__':
    main()




