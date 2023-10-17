# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:23:33 2022

@author: Xu Yaping
"""

import re, os, math, random
from argparse import ArgumentParser
from Map_Candidate_set import Candidate_set

import numpy as np
from scipy import stats
'''
from multiprocessing import Pool
from functools import partial
'''

def Mkdir(mk_path):
    folder = os.path.exists(mk_path)
    if not folder:
        os.makedirs(mk_path)

class Simulate_wet_pipeline():
    def __init__(self, K: int, deep: int, rsK = 41, rsN = 45):
        """
        Initialize the coder of CRC_Matirx.
        """
        self.K, self.rsK, self.rsN = K, rsK, rsN
        self.rsT = (self.rsN - self.rsK)//2
        self.read_size = 150

        self.Symbol2Cpdna = { 6: 1, 8: 1, 10: 1 }
        self.Pld_dna_size = self.rsK * self.Symbol2Cpdna[self.K]
        self.Block_dna_size = self.rsN * self.Symbol2Cpdna[self.K]
        
        self.encode_cpdna_list = self.CodeK2Rule(self.K)
        self.fmol, self.deep = 300000000000, deep
        self.Oligo_Order = ['A', 'C', 'G', 'T']


    def Simulate_read(self, pdl_infr):
        '''#To syn and seq each cpdna'''
        sequence_read = []
        for syn_cpdna in pdl_infr[-1]: sequence_read.append(Simulate_wet_pipeline.Simulate_CpDNA(self, syn_cpdna, pdl_infr[1]))
        sequence_readT = np.array(sequence_read).T #the syn details for one payload
        sequence_read_filt = [l for l in sequence_readT if len(l) == self.read_size]
        
        '''To infer composite letter from observed frequencies with MAP'''
        norm2CpDNA = []
        for l in range(self.read_size):
            Oligo_count = {ll: 0 for ll in self.Oligo_Order }
            for ll in range(len(sequence_read_filt)): Oligo_count[sequence_read_filt[ll][l]] += 1
            Clo_NormCpDNA = Simulate_wet_pipeline.Sample2Infer(self, [Oligo_count[ll] for ll in self.Oligo_Order])
            norm2CpDNA.append(Clo_NormCpDNA)
        return [pdl_infr[0], pdl_infr[1], norm2CpDNA]


    '''For one composite DNA'''
    def Simulate_CpDNA(self, CpDNA_refer, dp_refer):
        oligo_val = list(map(int, re.findall(r'\d+', CpDNA_refer)))
        k_ratio =  { i : r/self.K for i,r in zip(self.Oligo_Order, oligo_val) }
        cpdna_fmol_val = { r: int(k_ratio[r]*self.fmol) for r in k_ratio }
        sumfmol = sum(list(cpdna_fmol_val.values()))
        if sumfmol != self.fmol :
            maxoligo = sorted(k_ratio.items(), key=lambda d: d[1], reverse=True)[0][0]
            cpdna_fmol_val[maxoligo] += self.fmol-sumfmol if sumfmol < self.fmol else sumfmol-self.fmol
        sum2fmol = [cpdna_fmol_val['A'],cpdna_fmol_val['A']+cpdna_fmol_val['C'],cpdna_fmol_val['A']+cpdna_fmol_val['C']+cpdna_fmol_val['G'],self.fmol]
        
        #To shape ramdom seed
        seedoligo = random.sample(range(self.fmol), dp_refer)
        '''To insert the Oligo error, e.g mismatch, insert and delete'''
        mismatch_err, indel_err = np.array([1/3, 1/3, 1/3]), np.array([0.2, 0,2, 0.2, 0.2, 0.2]) 
        pdl_oligo = []
        for r in seedoligo :
            simul_oligo = ''
            if r < sum2fmol[0]:
                simul_oligo = self.Oligo_Order[0]
            elif r >= sum2fmol[0] and r < sum2fmol[1]:
                simul_oligo = self.Oligo_Order[1]
            elif r >= sum2fmol[1] and r < sum2fmol[2]:
                simul_oligo = self.Oligo_Order[2]
            elif r >= sum2fmol[2] and r < self.fmol:
                simul_oligo = self.Oligo_Order[-1]
            else:
                print('Error in countoligo.',r,sum2fmol)
                continue
            
            err_seed = random.sample(range(1, 101), 1)[0]  #oligo_err = 0.1
            if err_seed == 50:
                err_seed = random.sample(range(1, 101), 1)
                simul_oligo = np.random.choice(['A', 'C', 'G', 'T', ''], p=indel_err.ravel()) if err_seed == 50 else np.random.choice([l for l in self.Oligo_Order if l != simul_oligo], p=mismatch_err.ravel())
            pdl_oligo.append(simul_oligo)
        random.shuffle(pdl_oligo)
        return pdl_oligo

    def Sample2Infer(self, sample_observed):
        br_nr_cpdna = Simulate_wet_pipeline.Brute_Norm(self, sample_observed)
        if br_nr_cpdna in self.encode_cpdna_list:
            candidate_set = [br_nr_cpdna]
        else:
            candidate_set = []
        candidate_set.extend(Candidate_set(self.K, ':'.join(list(map(str, br_nr_cpdna)))))
        candidate_set_fre_val = []
        for sub_cpdna in candidate_set:
            candidate_set_fre_val.append([ sample_observed,  sub_cpdna, Simulate_wet_pipeline.Simulate_Fre(self, sub_cpdna, sample_observed) ])
        '''to sort by the simulate_fre_val'''
        candidate_set_fre_val.sort(key=lambda l:l[-1], reverse=True)
        candidate_set_fre_val = candidate_set_fre_val[0]
        
        sample_norm = [ candidate_set_fre_val[0][l]*self.K/sum(candidate_set_fre_val[0])  for l in range(4)]
        distance_O = [ pow(candidate_set_fre_val[1][ll]-sample_norm[ll], 2) for ll in range(4)]
        Max_fre_sample = [ ':'.join(list(map(str, candidate_set_fre_val[0]))), ':'.join(list(map(str, candidate_set_fre_val[1]))), sum(distance_O)]
        return Max_fre_sample

    def Brute_Norm(self, ton_dp):
        '''#firstly to normalize dp_list into cpdna_list --- 等比缩放#'''
        div = sum(ton_dp)/self.K
        '''to certain the sm of [quot_sum] is equal to [K]'''
        quot_mod = {}
        quot_sum = 0
        for nn in range(4):
            quot_mod[nn] = [int(ton_dp[nn]//div), ton_dp[nn]%div]
            quot_sum += quot_mod[nn][0]
    
        '''#按照余数排序: oligosorted :  [(0, [3, 9]), (3, [3, 7]), (2, [2, 3]), (1, [1, 1])] from high to low value'''
        oligosorted = sorted(quot_mod.items(),key=lambda l:(l[1][1], l[1][0]),reverse=True)
    
        '''to deal in futher'''    
        diff_K_val = self.K-quot_sum
        for d_val in range(diff_K_val):
            oligosorted[d_val][1][0] += 1
        
        '''#rel : [[0, 4], [3, 3], [2, 2], [1, 1]] , 加和为K, rel = [[oligo_index, quot], [oligo_index, quot], ..., ...]'''
        rel = [[oligosorted[l][0],oligosorted[l][1][0]] for l in range(4)] 
        rel.sort(key=lambda l:l[0])
        multirel = [ rel[l][1] for l in range(4)]
        return multirel

    def Simulate_Fre(self, sim_cpdna, sim_dp):
        '''to insert the error into fmol_sample'''
        err_rat = 0.01
        dp2Fmol = [ int( self.fmol*( l/self.K + err_rat*(1-4*(l/self.K))/3 ) ) for l in sim_cpdna ]
        if sum(dp2Fmol) != self.fmol:
            dp2Fmol[dp2Fmol.index(max(dp2Fmol))] += self.fmol - sum(dp2Fmol)
        
        '''to simulate the fre_value in the sr_cpdna'''
        sub_norm_log_val = [Simulate_wet_pipeline.Factorial(sim_dp[l], dp2Fmol[l]) for l in range(4)]
        sub_norm_log_val.sort()
        norm_rat_sum = sum(sub_norm_log_val)
        return norm_rat_sum

    def Shape_Fmol_Err(self):
        cpdna_list = []
        for a in range(self.K+1):
            for c in range(self.K+1):
                for g in range(self.K+1):
                    for t in range(self.K+1):
                        if a + t + c + g == self.K:
                            cpdna_list.append([a,t,c,g])

        ratioFmol_Set = {}
        err_rat = 0.01
        for ratioK in cpdna_list:
            ratioFmol = [ int( self.fmol*( K_r/self.K + err_rat*(1-4*(K_r/self.K))/3 ) ) for K_r in ratioK ]
            if sum(ratioFmol) != self.fmol:
                ratioFmol[ratioFmol.index(max(ratioFmol))] += self.fmol - sum(ratioFmol)
            a_str = ':'.join(list(map(str, ratioK)))
            ratioFmol_Set[a_str] = ratioFmol
        return ratioFmol_Set

    @staticmethod
    def Factorial(sample, fmol_part):
        sum1 = 0
        sum2 = 0
        if sample <= fmol_part/2:
            sample = fmol_part - sample
        else:
            sample = sample
        for f in range(sample+1,fmol_part+1):
            sum1 += math.log10(f)
        for f in range(1,fmol_part-sample+1):
            sum2 += math.log10(f)
        return sum1 - sum2

    @staticmethod
    def CodeK2Rule(K):
        K2CpDNA = {
            10: [[1, 4, 4, 1], [1, 8, 0, 1], [0, 8, 2, 0], [1, 1, 2, 6], [2, 4, 4, 0], [5, 1, 3, 1], [5, 0, 5, 0], [6, 2, 1, 1], [7, 0, 2, 1], [3, 0, 1, 6], [2, 5, 0, 3], [0, 4, 4, 2], [1, 1, 7, 1], [3, 5, 1, 1], [5, 3, 2, 0], [0, 0, 4, 6], [6, 1, 1, 2], [2, 2, 0, 6], [0, 3, 5, 2], [0, 0, 10, 0], [8, 0, 1, 1], [0, 1, 0, 9], [1, 1, 1, 7], [1, 4, 5, 0], [2, 1, 1, 6], [0, 9, 1, 0], [6, 3, 1, 0], [9, 1, 0, 0], [2, 4, 3, 1], [0, 2, 6, 2], [2, 0, 5, 3], [0, 5, 5, 0], [1, 5, 3, 1], [5, 1, 0, 4], [1, 5, 1, 3], [5, 3, 0, 2], [0, 0, 1, 9], [1, 0, 5, 4], [0, 3, 6, 1], [1, 1, 8, 0], [4, 4, 1, 1], [0, 3, 1, 6], [4, 0, 0, 6], [0, 2, 3, 5], [4, 5, 1, 0], [2, 6, 2, 0], [1, 6, 0, 3], [2, 3, 5, 0], [0, 1, 4, 5], [0, 10, 0, 0], [5, 0, 4, 1], [5, 0, 2, 3], [2, 2, 5, 1], [5, 0, 0, 5], [4, 1, 0, 5], [0, 1, 3, 6], [1, 4, 3, 2], [2, 1, 6, 1], [3, 5, 0, 2], [3, 1, 1, 5], [0, 6, 3, 1], [4, 2, 4, 0], [2, 1, 5, 2], [1, 0, 4, 5], [2, 5, 2, 1], [2, 3, 4, 1], [1, 6, 3, 0], [8, 0, 2, 0], [2, 4, 1, 3], [4, 1, 4, 1], [0, 4, 1, 5], [2, 0, 0, 8], [4, 0, 6, 0], [6, 0, 1, 3], [5, 0, 1, 4], [0, 0, 9, 1], [4, 0, 1, 5], [0, 9, 0, 1], [6, 1, 3, 0], [0, 5, 4, 1], [1, 0, 6, 3], [3, 1, 4, 2], [0, 2, 0, 8], [1, 3, 1, 5], [1, 1, 3, 5], [2, 5, 3, 0], [3, 1, 6, 0], [3, 0, 2, 5], [6, 0, 2, 2], [0, 7, 3, 0], [0, 4, 2, 4], [0, 6, 4, 0], [0, 3, 2, 5], [2, 4, 0, 4], [7, 0, 1, 2], [1, 4, 1, 4], [1, 3, 0, 6], [7, 2, 0, 1], [0, 4, 0, 6], [0, 7, 2, 1], [5, 4, 1, 0], [1, 1, 5, 3], [0, 8, 1, 1], [7, 0, 3, 0], [7, 1, 2, 0], [5, 3, 1, 1], [1, 4, 2, 3], [4, 0, 4, 2], [2, 6, 0, 2], [10, 0, 0, 0], [0, 2, 1, 7], [1, 4, 0, 5], [0, 0, 2, 8], [3, 7, 0, 0], [2, 2, 1, 5], [0, 7, 0, 3], [0, 2, 2, 6], [1, 0, 3, 6], [1, 0, 0, 9], [0, 1, 6, 3], [5, 5, 0, 0], [7, 0, 0, 3], [4, 1, 3, 2], [3, 1, 2, 4], [4, 0, 5, 1], [2, 5, 1, 2], [2, 1, 0, 7], [0, 3, 0, 7], [7, 1, 1, 1], [2, 7, 0, 1], [0, 2, 7, 1], [3, 6, 0, 1], [5, 1, 4, 0], [1, 2, 5, 2], [8, 0, 0, 2], [0, 4, 6, 0], [2, 0, 8, 0], [3, 2, 1, 4], [8, 1, 1, 0], [6, 0, 4, 0], [5, 0, 3, 2], [4, 5, 0, 1], [6, 0, 3, 1], [0, 1, 1, 8], [3, 2, 5, 0], [0, 1, 8, 1], [6, 1, 0, 3], [6, 3, 0, 1], [2, 1, 4, 3], [0, 6, 1, 3], [2, 0, 7, 1], [6, 1, 2, 1], [2, 8, 0, 0], [5, 1, 1, 3], [2, 3, 1, 4], [4, 1, 2, 3], [1, 2, 6, 1], [3, 2, 0, 5], [9, 0, 0, 1], [1, 6, 2, 1], [2, 6, 1, 1], [1, 9, 0, 0], [3, 0, 6, 1], [9, 0, 1, 0], [5, 2, 1, 2], [1, 0, 8, 1], [3, 2, 4, 1], [0, 5, 1, 4], [0, 0, 7, 3], [2, 1, 7, 0], [1, 5, 2, 2], [0, 5, 3, 2], [6, 2, 2, 0], [0, 8, 0, 2], [0, 6, 2, 2], [0, 5, 0, 5], [3, 0, 7, 0], [3, 6, 1, 0], [4, 6, 0, 0], [0, 3, 7, 0], [0, 6, 0, 4], [2, 0, 3, 5], [0, 5, 2, 3], [1, 3, 4, 2], [2, 0, 4, 4], [2, 0, 2, 6], [2, 0, 1, 7], [4, 2, 0, 4], [7, 2, 1, 0], [1, 2, 7, 0], [1, 0, 9, 0], [1, 2, 3, 4], [0, 0, 5, 5], [1, 3, 2, 4], [1, 5, 0, 4], [1, 2, 0, 7], [1, 2, 4, 3], [6, 2, 0, 2], [2, 3, 0, 5], [3, 5, 2, 0], [8, 1, 0, 1], [7, 3, 0, 0], [1, 1, 4, 4], [0, 0, 3, 7], [1, 8, 1, 0], [1, 1, 0, 8], [0, 1, 9, 0], [1, 3, 6, 0], [1, 0, 1, 8], [3, 0, 5, 2], [1, 3, 5, 1], [8, 2, 0, 0], [3, 4, 2, 1], [0, 2, 5, 3], [1, 6, 1, 2], [2, 1, 2, 5], [7, 1, 0, 2], [0, 2, 8, 0], [0, 1, 5, 4], [4, 4, 2, 0], [5, 2, 0, 3], [1, 7, 0, 2], [1, 0, 2, 7], [3, 1, 0, 6], [0, 7, 1, 2], [0, 0, 8, 2], [3, 0, 0, 7], [1, 2, 2, 5], [0, 4, 5, 1], [1, 5, 4, 0], [5, 2, 2, 1], [3, 4, 1, 2], [3, 1, 5, 1], [1, 1, 6, 2], [0, 0, 0, 10], [6, 4, 0, 0], [2, 1, 3, 4], [2, 0, 6, 2], [1, 7, 1, 1], [0, 2, 4, 4], [5, 1, 2, 2], [0, 0, 6, 4], [4, 1, 1, 4], [2, 2, 6, 0], [1, 7, 2, 0], [4, 4, 0, 2], [4, 1, 5, 0], [2, 7, 1, 0], [0, 1, 2, 7], [1, 0, 7, 2], [1, 2, 1, 6], [6, 0, 0, 4], [4, 0, 2, 4], [5, 4, 0, 1], [0, 1, 7, 2], [5, 2, 3, 0]],
            8: [[0, 4, 4, 0], [0, 7, 1, 0], [2, 1, 5, 0], [6, 0, 1, 1], [3, 1, 3, 1], [5, 1, 1, 1], [3, 0, 1, 4], [3, 4, 0, 1], [0, 5, 2, 1], [0, 1, 5, 2], [5, 2, 0, 1], [1, 1, 6, 0], [6, 0, 0, 2], [0, 0, 2, 6], [6, 2, 0, 0], [0, 3, 1, 4], [4, 1, 0, 3], [0, 3, 5, 0], [0, 0, 3, 5], [1, 0, 5, 2], [0, 3, 4, 1], [6, 0, 2, 0], [0, 0, 7, 1], [2, 0, 0, 6], [0, 4, 0, 4], [4, 2, 1, 1], [1, 0, 7, 0], [1, 1, 0, 6], [0, 1, 6, 1], [3, 4, 1, 0], [1, 3, 4, 0], [7, 0, 0, 1], [1, 1, 5, 1], [3, 0, 0, 5], [5, 0, 3, 0], [5, 0, 2, 1], [0, 0, 0, 8], [1, 2, 5, 0], [0, 1, 4, 3], [0, 2, 0, 6], [3, 1, 1, 3], [4, 3, 1, 0], [0, 0, 6, 2], [4, 0, 4, 0], [4, 0, 1, 3], [0, 5, 1, 2], [1, 2, 1, 4], [1, 6, 1, 0], [2, 4, 1, 1], [1, 3, 3, 1], [0, 5, 3, 0], [6, 1, 1, 0], [1, 1, 3, 3], [1, 0, 2, 5], [0, 6, 1, 1], [1, 4, 1, 2], [0, 1, 3, 4], [0, 1, 2, 5], [1, 4, 0, 3], [2, 6, 0, 0], [1, 0, 1, 6], [0, 6, 2, 0], [5, 3, 0, 0], [4, 1, 1, 2], [1, 0, 3, 4], [4, 1, 2, 1], [1, 5, 2, 0], [5, 1, 0, 2], [3, 3, 1, 1], [0, 4, 1, 3], [2, 1, 4, 1], [4, 0, 0, 4], [0, 1, 1, 6], [2, 0, 1, 5], [3, 5, 0, 0], [3, 1, 0, 4], [0, 1, 0, 7], [1, 1, 4, 2], [8, 0, 0, 0], [0, 0, 4, 4], [1, 3, 0, 4], [0, 6, 0, 2], [1, 6, 0, 1], [1, 2, 4, 1], [1, 5, 1, 1], [1, 0, 4, 3], [4, 4, 0, 0], [7, 1, 0, 0], [0, 0, 5, 3], [1, 4, 3, 0], [4, 0, 3, 1], [0, 0, 1, 7], [0, 3, 0, 5], [1, 3, 1, 3], [4, 1, 3, 0], [2, 1, 1, 4], [0, 2, 6, 0], [5, 0, 0, 3], [1, 2, 0, 5], [4, 3, 0, 1], [1, 5, 0, 2], [5, 1, 2, 0], [3, 0, 4, 1], [1, 7, 0, 0], [5, 2, 1, 0], [2, 0, 6, 0], [1, 4, 2, 1], [3, 0, 5, 0], [0, 4, 3, 1], [0, 0, 8, 0], [7, 0, 1, 0], [6, 1, 0, 1], [3, 1, 4, 0], [0, 5, 0, 3], [0, 1, 7, 0], [2, 5, 1, 0], [1, 1, 2, 4], [5, 0, 1, 2], [0, 7, 0, 1], [1, 0, 0, 7], [0, 2, 1, 5], [2, 1, 0, 5], [1, 1, 1, 5], [0, 8, 0, 0], [1, 0, 6, 1], [2, 0, 5, 1], [0, 2, 5, 1], [2, 5, 0, 1]],
            6: [[2, 4, 0, 0], [1, 1, 0, 4], [6, 0, 0, 0], [4, 0, 2, 0], [1, 5, 0, 0], [1, 1, 1, 3], [5, 0, 0, 1], [0, 3, 0, 3], [3, 3, 0, 0], [0, 0, 3, 3], [1, 0, 3, 2], [0, 2, 4, 0], [0, 0, 6, 0], [0, 2, 0, 4], [0, 0, 1, 5], [0, 0, 5, 1], [0, 0, 2, 4], [0, 6, 0, 0], [0, 4, 2, 0], [0, 1, 0, 5], [1, 1, 3, 1], [1, 4, 0, 1], [0, 1, 1, 4], [4, 1, 0, 1], [1, 3, 2, 0], [1, 0, 5, 0], [5, 1, 0, 0], [3, 0, 0, 3], [0, 2, 3, 1], [1, 2, 0, 3], [3, 0, 3, 0], [2, 0, 0, 4], [1, 0, 4, 1], [0, 2, 1, 3], [0, 0, 0, 6], [1, 4, 1, 0], [0, 5, 0, 1], [4, 0, 0, 2], [0, 4, 0, 2], [1, 3, 0, 2], [4, 2, 0, 0], [0, 1, 3, 2], [2, 0, 3, 1], [4, 1, 1, 0], [0, 1, 5, 0], [1, 3, 1, 1], [0, 4, 1, 1], [0, 3, 2, 1], [1, 0, 1, 4], [0, 0, 4, 2], [0, 1, 2, 3], [1, 1, 4, 0], [1, 0, 2, 3], [4, 0, 1, 1], [0, 3, 3, 0], [2, 0, 4, 0], [0, 3, 1, 2], [1, 2, 3, 0], [2, 0, 1, 3], [0, 5, 1, 0], [5, 0, 1, 0], [1, 0, 0, 5], [3, 1, 1, 1], [0, 1, 4, 1]],
    
            }
        '''#to certain the process of runnign code'''
        if K in [10,8,6]:
            return K2CpDNA[K]
        else :
            print('Error! This suit of code is set to the K in 10,8,6,4,3,2. ')
            print('If the K out of range , then the code suit have dificiency in GFint and Reed Selemon part.')
            exit(1)
        return 0


    def NormDepth(self, file_en):
        '''line count of encode file'''
        encode_number = 0
        f_en = open(file_en, 'r')
        for line in f_en:
            encode_number += 1
        f_en.close()
        depth_set = np.arange(int(self.deep*0.5) ,int(self.deep*1.5))
        self.deep_U, self.deep_L = depth_set + 0.5, depth_set - 0.5 
        prob = stats.norm.cdf(self.deep_U,loc=self.deep, scale = 100) - stats.norm.cdf(self.deep_L,loc=self.deep, scale = 100)
        prob = prob / prob.sum()
        nums_set = np.random.choice(depth_set, size = encode_number//2, p = prob)
        for n in range(len(nums_set)):
            real_deep = int(max(self.deep*0.1, self.K)) 
            if nums_set[n] <= real_deep:
                nums_set[n] = random.sample(range(real_deep,self.deep),1)[0]
        return nums_set


    def Syn_Norm(self, file_en: str, file_wet: str, file_wet_col: str, need_logs: bool = True):
        contig_dp = Simulate_wet_pipeline.NormDepth(self, file_en)
        write_col_cnt = 0
         
        f_en = open(file_en,'r')
        f_nr = open(file_wet, 'w')
        f_nr_col= open(file_wet_col, 'w')
        syn_result_matrix, syn_result_matrix_dp = [], []
        for line in f_en :
            if re.search(r'^>',line):
                pdl_id = int(re.findall('\d+', line)[0])
            elif re.search(r'^\d',line):
                line = line.strip().split(',')
                syn_result = Simulate_wet_pipeline.Simulate_read(self, [pdl_id, contig_dp[pdl_id-1], line])
                syn_result_matrix.append(syn_result[-1])
                syn_result_matrix_dp.append(str(syn_result[1]))
                f_nr.write('>contig{}\t{}'.format(syn_result[0], syn_result[1]) + '\n')
                Row_read_str = [ l[1] for l in syn_result[-1]]
                f_nr.write('{}'.format(','.join(Row_read_str)) + '\n')
                
                if pdl_id % self.Block_dna_size == 0:
                    for l in range(self.read_size):
                        block_cpdna = [syn_result_matrix[ll][l] for ll in range(self.Block_dna_size)]
                        write_col_cnt += 1
                        f_nr_col.write('>conitg{}'.format(write_col_cnt) + '\n')
                        f_nr_col.write('CpDNA:\t{}'.format(','.join([ll[1] for ll in block_cpdna])) + '\n')
                        f_nr_col.write('Frequ:\t{}'.format(','.join([ll[0] for ll in block_cpdna])) + '\n')
                        f_nr_col.write('Depth:\t{}'.format(','.join(syn_result_matrix_dp)) + '\n')
                        f_nr_col.write('EDist:\t{}'.format(','.join([str(ll[-1]) for ll in block_cpdna])) + '\n')
                    syn_result_matrix, syn_result_matrix_dp = [], []
            else:
                print('ERROR IN LINE: ', line)
                break
        f_en.close()
        f_nr.close()
        f_nr_col.close()
        return 0 


def read_args():
    """
    Read arguments from the command line.

    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value in 6(64), 8(128), 10(256).")
    parser.add_argument("-d", "--deep", required=True, type=str,
                        help="the inferred sequences path")
    parser.add_argument("-i", "--encoded_path", required=True, type=str,
                        help="the encoding seqeuences path with Derrick-cp.")
    parser.add_argument("-s1", "--saved_path", required=True, type=str,
                        help="the simulated seqeuences path")
    parser.add_argument("-s2", "--saved_pretreat_path", required=True, type=str,
                        help="the simulated seqeuences wirh pretreatment path")

    return parser.parse_args()

if __name__ == '__main__':
    '''#the global variance#'''
    params = read_args()
    print("The parameters are:")
    print("k = ", params.resolution) #-k
    print("Sequencing depth = ", params.deep)  #-d
    print("Encoded sequences path = ", params.encoded_path)   #-i
    print("Simulated sequences path = ", params.saved_path)   #-s
    print("Simulated sequences wirh pretreatment path = ", params.saved_pretreat_path)   #-s

    Simulate_wet = Simulate_wet_pipeline(params.resolution, params.deep)
    Simulate_wet.Syn_Norm(params.encoded_path, params.saved_path, params.saved_pretreat_path)
