# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:23:33 2022

@author: Xu Yaping
"""

import numpy as np
import re, math, random, time
from scipy import stats
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser
from Map_Candidate_set import Candidate_set

class Simulate_wet_pipeline():
    def __init__(self, k: int, alp_size: int, deep: int, err_rate: float, sample_val: float, rsK: int = 41, rsN: int = 45):
        '''# TODO 0-0: Initialize the coder of Matirx consisting of ten consecutive blocks and CRC32.'''
        self.k, self.alp_size, self.matrix_size, self.read_size , self.crc= k, alp_size, 10, 150, 32
        self.fmol, self.deep = 3*pow(10, 11), deep
        self.Oligo_Order = ['A', 'C', 'G', 'T']

        '''# TODO 0-1:  Two conversions --- Bits and GFint, Bits and letters'''
        self.Cpdna_set, self.Symbol2bit, self.GFint, self.CpDNA2bit = Simulate_wet_pipeline.Basic_infr(self)
        self.Cpdna_list = list(self.Cpdna_set.values())
        
        '''# TODO 0-2:  Simualte the padding number of bits, to comlish the mapping between bits and letters.'''
        padding_val = divmod(rsN * self.Symbol2bit[1], self.CpDNA2bit[1])
        self.Block_dna_size = padding_val[0]*self.CpDNA2bit[0] if padding_val[1] == 0 else (padding_val[0]+1)*self.CpDNA2bit[0]
        
        '''# TODO 0-3: Set the parameter about the sampling and nucleotide error'''
        self.error_rate = float(err_rate)
        self.error_seed = random.sample(range( 1, int(1/self.error_rate) + 1), 1)[0]
        self.sample_val = float(sample_val)

    def Syn_Norm(self, file_en: str, file_wet: str, file_wet_col: str, need_logs: bool = True):
        print("Encoded sequences path = ", file_en)   #-i
        print("Simulated sequences path = ", file_wet)   #-o
        print("Simulated sequences wirh pretreatment path = ", file_wet_col)   #-p

        '''#TODO 1-0: Deep follows the normal distribution#'''
        contig_dp = Simulate_wet_pipeline.NormDepth(self, file_en)

        '''#TODO 1-1: Extracting the encoded read#'''
        Encode_infr = []
        f_en = open(file_en,'r')
        for line in f_en:
            if re.search(r'^>',line):
                read_id = int(re.findall('\d+', line)[0])
            elif re.search(r'^\d',line):
                line = line.strip().split(',')
                Encode_infr.append([read_id, contig_dp[read_id-1], line])
            else:
                print('ERROR IN LINE: ', line)
                break
        f_en.close()
        
        col_no, Pool_cnt = 0, 20
        f_nr, f_nr_col = open(file_wet, 'w'), open(file_wet_col, 'w')
        for l in range(0, len(Encode_infr), Pool_cnt*self.Block_dna_size*10):
            Pool_read = Encode_infr[l: l + Pool_cnt*self.Block_dna_size*10]
            Pool_infr = [Pool_read[l: l+self.Block_dna_size] for l in range(0, Pool_cnt*self.Block_dna_size*10, self.Block_dna_size)]
            
            '''#TODO 1-2: Simulate each matrix containg in multi matrix_crc32 in multithread#'''
            p = Pool(Pool_cnt)
            matirx_results = p.map(Simulate_wet.Simulate_matrix, Pool_infr)
            p.close()
            p.join()
            
            '''#TODO 1-2: Simulate each matrix_crc32 in multithread#'''
            for sub_matrix in matirx_results:
                if sub_matrix != []:
                    syn_result_matrix, syn_result_matrix_dp = [], []
                    for syn_result in sub_matrix:
                        #[ [ Max_fre_sample: str, cpdna: str, distance: float ], ..., ...]
                        syn_result_matrix.append(syn_result[-1])
                        syn_result_matrix_dp.append(str(syn_result[1]))
                        
                        f_nr.write('>contig{}\t{}\t{}\n'.format(syn_result[0], syn_result[1], syn_result[2]))
                        Row_read_str = [ l[1] for l in syn_result[-1]]
                        f_nr.write('{}\n'.format(','.join(Row_read_str)))
                    
                    '''#TODO 1-3: Output the infered letter read#'''
                    for l in range(self.read_size):
                        block_cpdna = [syn_result_matrix[ll][l] for ll in range(self.Block_dna_size)]
                        col_no += 1
                        f_nr_col.write('>conitg{}\n'.format(col_no))
                        f_nr_col.write('CpDNA:\t{}\n'.format(','.join([ll[1] for ll in block_cpdna])))
                        f_nr_col.write('Dprat:\t{}\n'.format(','.join([ll[0] for ll in block_cpdna])))
                        f_nr_col.write('Depth:\t{}\n'.format(','.join(syn_result_matrix_dp)))
                        f_nr_col.write('EDist:\t{}\n'.format(','.join([str(ll[-1]) for ll in block_cpdna])) )
        f_en.close()
        f_nr.close()
        f_nr_col.close()
        return 0 
    

    def NormDepth(self, file_en):
        '''#TODO 3-0: Simulate each matrix_crc32 in multithread#'''
        encode_no = 0 #-1
        for encode_no, line in enumerate(open(file_en,'r')):
            encode_no+=1
        
        #The vast majority of depth is within four standard deviations, that is, from (1-self.sample_val)*depth to (1+self.sample_val)*depth
        depth_set = np.arange(int(self.deep*(1-self.sample_val)) ,int(self.deep*(1+self.sample_val)))
        self.deep_U, self.deep_L = depth_set + 0.5, depth_set - 0.5 
        prob = stats.norm.cdf(self.deep_U,loc=self.deep, scale = 100) - stats.norm.cdf(self.deep_L,loc=self.deep, scale = 100)
        prob = prob / prob.sum()
        nums_set = np.random.choice(depth_set, size = encode_no//2, p = prob)
        for n in range(len(nums_set)):
            real_deep = int(max(self.deep*0.1, self.k)) 
            if nums_set[n] <= real_deep:
                nums_set[n] = random.sample(range(real_deep,self.deep),1)[0]
        return nums_set
    

    def Basic_infr(self): #->dict, list, list, list
        '''#The composite dna set filtered by their accuracy and used to the digital data storage (DDS), and the sizes of Cpdna set are convenient to converse between binary information and composite DNA letters#
            #k=10(400): {'0:0:0:10': 0.99999995, '0:0:10:0': 0.99999995, '0:10:0:0': 0.99999995, '10:0:0:0': 0.99999995, '1:9:0:0': 0.99740026, '0:0:1:9': 0.99740026, '0:0:9:1': 0.99740026, '0:1:0:9': 0.99740026, '0:1:9:0': 0.99740026, '0:9:0:1': 0.99740026, '0:9:1:0': 0.99740026, '1:0:0:9': 0.99740026, '1:0:9:0': 0.99740026, '9:0:0:1': 0.99740026, '9:0:1:0': 0.99740026, '9:1:0:0': 0.99740026, '0:1:1:8': 0.99546847, '0:1:8:1': 0.99546847, '0:8:1:1': 0.99546847, '1:0:1:8': 0.99546847, '1:0:8:1': 0.99546847, '1:1:0:8': 0.99546847, '1:1:8:0': 0.99546847, '1:8:0:1': 0.99546847, '1:8:1:0': 0.99546847, '8:0:1:1': 0.99546847, '8:1:0:1': 0.99546847, '8:1:1:0': 0.99546847, '1:1:1:7': 0.99406382, '1:7:1:1': 0.99406382, '7:1:1:1': 0.99406382, '1:1:7:1': 0.99406382, '0:0:2:8': 0.98716647, '0:0:8:2': 0.98716647, '0:2:0:8': 0.98716647, '0:8:0:2': 0.98716647, '0:8:2:0': 0.98716647, '2:0:0:8': 0.98716647, '0:2:8:0': 0.98716647, '2:0:8:0': 0.98716647, '2:8:0:0': 0.98716647, '8:0:0:2': 0.98716647, '8:0:2:0': 0.98716647, '8:2:0:0': 0.98716647, '6:2:1:1': 0.98628256, '6:1:2:1': 0.98628256, '1:1:2:6': 0.98628256, '1:6:2:1': 0.98628256, '6:1:1:2': 0.98628256, '1:1:6:2': 0.98628256, '1:2:1:6': 0.98628256, '1:2:6:1': 0.98628256, '1:6:1:2': 0.98628256, '2:6:1:1': 0.98628256, '2:1:1:6': 0.98628256, '2:1:6:1': 0.98628256, '0:1:7:2': 0.98620412, '0:2:1:7': 0.98620412, '1:7:0:2': 0.98620412, '1:7:2:0': 0.98620412, '2:0:1:7': 0.98620412, '2:0:7:1': 0.98620412, '2:1:0:7': 0.98620412, '2:1:7:0': 0.98620412, '2:7:0:1': 0.98620412, '2:7:1:0': 0.98620412, '7:0:1:2': 0.98620412, '7:0:2:1': 0.98620412, '7:1:0:2': 0.98620412, '7:1:2:0': 0.98620412, '7:2:0:1': 0.98620412, '7:2:1:0': 0.98620412, '0:1:2:7': 0.98620412, '0:2:7:1': 0.98620412, '0:7:1:2': 0.98620412, '0:7:2:1': 0.98620412, '1:0:2:7': 0.98620412, '1:0:7:2': 0.98620412, '1:2:0:7': 0.98620412, '1:2:7:0': 0.98620412, '1:2:5:2': 0.98174905, '1:2:2:5': 0.98174905, '2:1:2:5': 0.98174905, '2:1:5:2': 0.98174905, '2:2:1:5': 0.98174905, '2:2:5:1': 0.98174905, '2:5:1:2': 0.98174905, '2:5:2:1': 0.98174905, '5:1:2:2': 0.98174905, '5:2:1:2': 0.98174905, '5:2:2:1': 0.98174905, '1:5:2:2': 0.98174905, '4:2:2:2': 0.98081195, '2:4:2:2': 0.98081195, '2:2:2:4': 0.98081195, '2:2:4:2': 0.98081195, '0:2:2:6': 0.97927898, '2:0:2:6': 0.97927898, '0:2:6:2': 0.97927898, '2:0:6:2': 0.97927898, '2:2:0:6': 0.97927898, '2:2:6:0': 0.97927898, '2:6:0:2': 0.97927898, '2:6:2:0': 0.97927898, '0:6:2:2': 0.97927898, '6:0:2:2': 0.97927898, '6:2:0:2': 0.97927898, '6:2:2:0': 0.97927898, '2:2:3:3': 0.97764945, '2:3:2:3': 0.97764945, '2:3:3:2': 0.97764945, '3:2:2:3': 0.97764945, '3:2:3:2': 0.97764945, '3:3:2:2': 0.97764945, '5:3:1:1': 0.97596792, '5:1:3:1': 0.97596792, '1:1:3:5': 0.97596792, '1:3:1:5': 0.97596792, '1:3:5:1': 0.97596792, '1:5:3:1': 0.97596792, '3:1:1:5': 0.97596792, '5:1:1:3': 0.97596792, '1:1:5:3': 0.97596792, '3:5:1:1': 0.97596792, '1:5:1:3': 0.97596792, '3:1:5:1': 0.97596792, '4:1:2:3': 0.97475491, '4:1:3:2': 0.97475491, '4:3:1:2': 0.97475491, '3:4:1:2': 0.97475491, '4:2:1:3': 0.97475491, '4:2:3:1': 0.97475491, '1:2:3:4': 0.97475491, '1:3:2:4': 0.97475491, '1:3:4:2': 0.97475491, '1:4:2:3': 0.97475491, '1:4:3:2': 0.97475491, '2:1:3:4': 0.97475491, '2:3:1:4': 0.97475491, '2:3:4:1': 0.97475491, '2:4:1:3': 0.97475491, '3:1:2:4': 0.97475491, '3:1:4:2': 0.97475491, '3:2:1:4': 0.97475491, '3:2:4:1': 0.97475491, '3:4:2:1': 0.97475491, '4:3:2:1': 0.97475491, '1:2:4:3': 0.97475491, '2:1:4:3': 0.97475491, '2:4:3:1': 0.97475491, '1:3:3:3': 0.97331835, '3:1:3:3': 0.97331835, '3:3:1:3': 0.97331835, '3:3:3:1': 0.97331835, '0:3:1:6': 0.97257357, '0:3:6:1': 0.97257357, '0:1:3:6': 0.97257357, '0:6:1:3': 0.97257357, '1:0:3:6': 0.97257357, '3:0:1:6': 0.97257357, '3:0:6:1': 0.97257357, '3:1:6:0': 0.97257357, '3:6:0:1': 0.97257357, '3:6:1:0': 0.97257357, '0:1:6:3': 0.97257357, '1:0:6:3': 0.97257357, '1:3:0:6': 0.97257357, '1:6:0:3': 0.97257357, '3:1:0:6': 0.97257357, '6:0:1:3': 0.97257357, '6:0:3:1': 0.97257357, '6:1:0:3': 0.97257357, '6:3:0:1': 0.97257357, '0:6:3:1': 0.97257357, '1:3:6:0': 0.97257357, '1:6:3:0': 0.97257357, '6:1:3:0': 0.97257357, '6:3:1:0': 0.97257357, '4:1:1:4': 0.9715781, '1:4:1:4': 0.9715781, '1:4:4:1': 0.9715781, '4:1:4:1': 0.9715781, '4:4:1:1': 0.9715781, '1:1:4:4': 0.9715781, '0:7:3:0': 0.9696671, '3:0:0:7': 0.9696671, '3:0:7:0': 0.9696671, '3:7:0:0': 0.9696671, '7:0:0:3': 0.9696671, '7:0:3:0': 0.9696671, '7:3:0:0': 0.9696671, '0:0:3:7': 0.9696671, '0:0:7:3': 0.9696671, '0:3:0:7': 0.9696671, '0:7:0:3': 0.9696671, '0:3:7:0': 0.9696671, '2:0:3:5': 0.96946634, '3:2:0:5': 0.96946634, '0:2:3:5': 0.96946634, '0:3:2:5': 0.96946634, '3:0:2:5': 0.96946634, '0:2:5:3': 0.96946634, '0:3:5:2': 0.96946634, '0:5:2:3': 0.96946634, '0:5:3:2': 0.96946634, '2:0:5:3': 0.96946634, '2:3:0:5': 0.96946634, '2:3:5:0': 0.96946634, '2:5:0:3': 0.96946634, '2:5:3:0': 0.96946634, '3:0:5:2': 0.96946634, '3:2:5:0': 0.96946634, '3:5:0:2': 0.96946634, '3:5:2:0': 0.96946634, '5:0:2:3': 0.96946634, '5:0:3:2': 0.96946634, '5:2:0:3': 0.96946634, '5:2:3:0': 0.96946634, '5:3:0:2': 0.96946634, '5:3:2:0': 0.96946634, '2:4:4:0': 0.9654088, '0:2:4:4': 0.9654088, '0:4:2:4': 0.9654088, '0:4:4:2': 0.9654088, '2:0:4:4': 0.9654088, '2:4:0:4': 0.9654088, '4:0:2:4': 0.9654088, '4:0:4:2': 0.9654088, '4:2:0:4': 0.9654088, '4:2:4:0': 0.9654088, '4:4:0:2': 0.9654088, '4:4:2:0': 0.9654088, '0:1:4:5': 0.96282785, '0:1:5:4': 0.96282785, '1:4:0:5': 0.96282785, '1:4:5:0': 0.96282785, '4:0:1:5': 0.96282785, '0:5:1:4': 0.96282785, '0:5:4:1': 0.96282785, '1:0:4:5': 0.96282785, '4:1:0:5': 0.96282785, '5:0:4:1': 0.96282785, '5:4:0:1': 0.96282785, '0:4:1:5': 0.96282785, '1:0:5:4': 0.96282785, '1:5:0:4': 0.96282785, '1:5:4:0': 0.96282785, '4:0:5:1': 0.96282785, '4:1:5:0': 0.96282785, '4:5:0:1': 0.96282785, '4:5:1:0': 0.96282785, '5:0:1:4': 0.96282785, '5:1:4:0': 0.96282785, '5:4:1:0': 0.96282785, '0:4:5:1': 0.96282785, '5:1:0:4': 0.96282785, '0:4:3:3': 0.96272354, '0:3:3:4': 0.96272354, '0:3:4:3': 0.96272354, '3:0:3:4': 0.96272354, '3:0:4:3': 0.96272354, '3:3:0:4': 0.96272354, '3:3:4:0': 0.96272354, '3:4:0:3': 0.96272354, '4:0:3:3': 0.96272354, '4:3:0:3': 0.96272354, '4:3:3:0': 0.96272354, '3:4:3:0': 0.96272354, '0:0:6:4': 0.95921278, '0:0:4:6': 0.95921278, '0:4:0:6': 0.95921278, '0:4:6:0': 0.95921278, '0:6:0:4': 0.95921278, '0:6:4:0': 0.95921278, '4:0:0:6': 0.95921278, '4:0:6:0': 0.95921278, '6:0:0:4': 0.95921278, '6:0:4:0': 0.95921278, '6:4:0:0': 0.95921278, '4:6:0:0': 0.95921278, '5:0:5:0': 0.95252961, '5:5:0:0': 0.95252961, '0:5:5:0': 0.95252961, '0:5:0:5': 0.95252961, '5:0:0:5': 0.95252961, '0:0:5:5': 0.95252961}
            #k=8(300):  {'0:8:0:0': 0.99999997, '8:0:0:0': 0.99999997, '0:0:0:8': 0.99999997, '0:0:8:0': 0.99999997, '0:0:1:7': 0.99739989, '0:1:0:7': 0.99739989, '0:1:7:0': 0.99739989, '1:0:7:0': 0.99739989, '1:7:0:0': 0.99739989, '7:0:0:1': 0.99739989, '7:0:1:0': 0.99739989, '7:1:0:0': 0.99739989, '0:0:7:1': 0.99739989, '0:7:0:1': 0.99739989, '0:7:1:0': 0.99739989, '1:0:0:7': 0.99739989, '0:1:1:6': 0.99548424, '0:1:6:1': 0.99548424, '0:6:1:1': 0.99548424, '1:0:1:6': 0.99548424, '1:0:6:1': 0.99548424, '1:1:0:6': 0.99548424, '1:1:6:0': 0.99548424, '1:6:0:1': 0.99548424, '1:6:1:0': 0.99548424, '6:0:1:1': 0.99548424, '6:1:0:1': 0.99548424, '6:1:1:0': 0.99548424, '1:1:1:5': 0.99460764, '1:1:5:1': 0.99460764, '1:5:1:1': 0.99460764, '5:1:1:1': 0.99460764, '2:2:2:2': 0.99154185, '1:4:1:2': 0.98945246, '2:4:1:1': 0.98945246, '4:1:1:2': 0.98945246, '1:1:4:2': 0.98945246, '1:2:4:1': 0.98945246, '2:1:1:4': 0.98945246, '2:1:4:1': 0.98945246, '4:2:1:1': 0.98945246, '1:4:2:1': 0.98945246, '4:1:2:1': 0.98945246, '1:1:2:4': 0.98945246, '1:2:1:4': 0.98945246, '1:2:2:3': 0.98836451, '2:1:2:3': 0.98836451, '2:1:3:2': 0.98836451, '2:2:3:1': 0.98836451, '1:2:3:2': 0.98836451, '2:2:1:3': 0.98836451, '2:3:1:2': 0.98836451, '2:3:2:1': 0.98836451, '1:3:2:2': 0.98836451, '3:1:2:2': 0.98836451, '3:2:1:2': 0.98836451, '3:2:2:1': 0.98836451, '5:1:0:2': 0.98780231, '5:1:2:0': 0.98780231, '0:2:5:1': 0.98780231, '0:5:1:2': 0.98780231, '0:5:2:1': 0.98780231, '2:0:5:1': 0.98780231, '2:5:0:1': 0.98780231, '2:5:1:0': 0.98780231, '5:0:1:2': 0.98780231, '5:0:2:1': 0.98780231, '5:2:0:1': 0.98780231, '5:2:1:0': 0.98780231, '0:1:2:5': 0.98780231, '0:1:5:2': 0.98780231, '0:2:1:5': 0.98780231, '1:0:2:5': 0.98780231, '1:0:5:2': 0.98780231, '1:2:0:5': 0.98780231, '1:2:5:0': 0.98780231, '1:5:0:2': 0.98780231, '1:5:2:0': 0.98780231, '2:0:1:5': 0.98780231, '2:1:0:5': 0.98780231, '2:1:5:0': 0.98780231, '0:0:2:6': 0.98736367, '0:0:6:2': 0.98736367, '0:2:0:6': 0.98736367, '0:2:6:0': 0.98736367, '2:0:0:6': 0.98736367, '2:0:6:0': 0.98736367, '2:6:0:0': 0.98736367, '0:6:0:2': 0.98736367, '0:6:2:0': 0.98736367, '6:0:0:2': 0.98736367, '6:0:2:0': 0.98736367, '6:2:0:0': 0.98736367, '1:3:1:3': 0.98602879, '3:1:1:3': 0.98602879, '3:1:3:1': 0.98602879, '1:1:3:3': 0.98602879, '3:3:1:1': 0.98602879, '1:3:3:1': 0.98602879, '0:2:4:2': 0.9838565, '2:0:2:4': 0.9838565, '2:0:4:2': 0.9838565, '2:2:0:4': 0.9838565, '2:2:4:0': 0.9838565, '2:4:0:2': 0.9838565, '2:4:2:0': 0.9838565, '4:0:2:2': 0.9838565, '4:2:0:2': 0.9838565, '4:2:2:0': 0.9838565, '0:2:2:4': 0.9838565, '0:4:2:2': 0.9838565, '3:3:2:0': 0.98029011, '0:3:3:2': 0.98029011, '2:0:3:3': 0.98029011, '2:3:0:3': 0.98029011, '2:3:3:0': 0.98029011, '3:0:3:2': 0.98029011, '3:2:3:0': 0.98029011, '3:0:2:3': 0.98029011, '3:2:0:3': 0.98029011, '3:3:0:2': 0.98029011, '0:2:3:3': 0.98029011, '0:3:2:3': 0.98029011, '0:1:4:3': 0.97857201, '0:4:1:3': 0.97857201, '1:4:0:3': 0.97857201, '1:4:3:0': 0.97857201, '3:0:1:4': 0.97857201, '3:1:4:0': 0.97857201, '4:0:1:3': 0.97857201, '4:0:3:1': 0.97857201, '4:1:0:3': 0.97857201, '4:1:3:0': 0.97857201, '4:3:0:1': 0.97857201, '4:3:1:0': 0.97857201, '0:1:3:4': 0.97857201, '0:3:1:4': 0.97857201, '0:3:4:1': 0.97857201, '0:4:3:1': 0.97857201, '1:0:3:4': 0.97857201, '1:0:4:3': 0.97857201, '1:3:0:4': 0.97857201, '1:3:4:0': 0.97857201, '3:0:4:1': 0.97857201, '3:1:0:4': 0.97857201, '3:4:0:1': 0.97857201, '3:4:1:0': 0.97857201, '0:0:3:5': 0.97451882, '0:0:5:3': 0.97451882, '0:3:0:5': 0.97451882, '0:3:5:0': 0.97451882, '0:5:0:3': 0.97451882, '0:5:3:0': 0.97451882, '3:0:0:5': 0.97451882, '3:0:5:0': 0.97451882, '3:5:0:0': 0.97451882, '5:0:0:3': 0.97451882, '5:0:3:0': 0.97451882, '5:3:0:0': 0.97451882, '0:0:4:4': 0.9701977, '0:4:0:4': 0.9701977, '4:0:0:4': 0.9701977, '4:0:4:0': 0.9701977, '4:4:0:0': 0.9701977, '0:4:4:0': 0.9701977}
            #k=6(150):  {'0:0:0:6': 0.99999734, '0:0:6:0': 0.99999734, '0:6:0:0': 0.99999734, '6:0:0:0': 0.99999734, '0:0:1:5': 0.99113196, '0:0:5:1': 0.99113196, '0:1:0:5': 0.99113196, '0:5:1:0': 0.99113196, '1:0:0:5': 0.99113196, '1:0:5:0': 0.99113196, '0:1:5:0': 0.99113196, '0:5:0:1': 0.99113196, '1:5:0:0': 0.99113196, '5:0:0:1': 0.99113196, '5:0:1:0': 0.99113196, '5:1:0:0': 0.99113196, '1:1:3:1': 0.98696775, '1:3:1:1': 0.98696545, '1:1:1:3': 0.98687725, '3:1:1:1': 0.98687725, '0:1:4:1': 0.98686702, '1:0:4:1': 0.98686702, '1:4:1:0': 0.98686702, '1:4:0:1': 0.98686702, '0:4:1:1': 0.98686702, '4:1:1:0': 0.98686702, '0:1:1:4': 0.98686702, '1:0:1:4': 0.98686702, '1:1:0:4': 0.98686702, '1:1:4:0': 0.98686702, '4:0:1:1': 0.98686702, '4:1:0:1': 0.98686702, '1:1:2:2': 0.98474756, '1:2:2:1': 0.98425314, '1:2:1:2': 0.98425036, '2:1:2:1': 0.98373195, '2:1:1:2': 0.98372945, '2:2:1:1': 0.9836936, '2:0:2:2': 0.9736908, '2:2:0:2': 0.9736908, '2:2:2:0': 0.9736908, '0:2:2:2': 0.9736908, '1:0:3:2': 0.97318602, '1:3:0:2': 0.97318602, '1:3:2:0': 0.97318602, '2:3:1:0': 0.97318602, '0:1:3:2': 0.97318602, '2:3:0:1': 0.97318602, '0:2:3:1': 0.97318602, '2:0:3:1': 0.97318602, '1:2:3:0': 0.97318602, '1:0:2:3': 0.97318602, '1:2:0:3': 0.97318602, '3:2:0:1': 0.97318602, '3:2:1:0': 0.97318602, '0:1:2:3': 0.97318602, '3:0:2:1': 0.97318602, '0:3:2:1': 0.97318602, '2:1:0:3': 0.97318602, '2:1:3:0': 0.97318602, '3:0:1:2': 0.97318602, '3:1:0:2': 0.97318602, '3:1:2:0': 0.97318602, '0:2:1:3': 0.97318602, '0:3:1:2': 0.97318602, '2:0:1:3': 0.97318602, '0:4:0:2': 0.9711819, '4:0:0:2': 0.9711819, '4:0:2:0': 0.9711819, '0:0:2:4': 0.9711819, '0:0:4:2': 0.9711819, '0:2:4:0': 0.9711819, '0:2:0:4': 0.9711819, '0:4:2:0': 0.9711819, '4:2:0:0': 0.9711819, '2:0:0:4': 0.9711819, '2:0:4:0': 0.9711819, '2:4:0:0': 0.9711819, '3:0:3:0': 0.96197842, '3:3:0:0': 0.96197842, '0:0:3:3': 0.96197842, '0:3:0:3': 0.96197842, '0:3:3:0': 0.96197842, '3:0:0:3': 0.96197842}
        '''
        Cpdna_set = {
            10: {0: '10:0:0:0', 1: '0:10:0:0', 2: '0:0:10:0', 3: '0:0:0:10', 4: '9:1:0:0', 5: '9:0:1:0', 6: '9:0:0:1', 7: '1:9:0:0', 8: '1:0:9:0', 9: '1:0:0:9', 10: '0:9:1:0', 11: '0:9:0:1', 12: '0:1:9:0', 13: '0:1:0:9', 14: '0:0:9:1', 15: '0:0:1:9', 16: '8:2:0:0', 17: '8:0:2:0', 18: '8:0:0:2', 19: '2:8:0:0', 20: '2:0:8:0', 21: '2:0:0:8', 22: '0:8:2:0', 23: '0:8:0:2', 24: '0:2:8:0', 25: '0:2:0:8', 26: '0:0:8:2', 27: '0:0:2:8', 28: '7:3:0:0', 29: '7:0:3:0', 30: '7:0:0:3', 31: '3:7:0:0', 32: '3:0:7:0', 33: '3:0:0:7', 34: '0:7:3:0', 35: '0:7:0:3', 36: '0:3:7:0', 37: '0:3:0:7', 38: '0:0:7:3', 39: '0:0:3:7', 40: '6:4:0:0', 41: '6:0:4:0', 42: '6:0:0:4', 43: '4:6:0:0', 44: '4:0:6:0', 45: '4:0:0:6', 46: '0:6:4:0', 47: '0:6:0:4', 48: '0:4:6:0', 49: '0:4:0:6', 50: '0:0:6:4', 51: '0:0:4:6', 52: '5:5:0:0', 53: '5:0:5:0', 54: '5:0:0:5', 55: '0:5:5:0', 56: '0:5:0:5', 57: '0:0:5:5', 58: '8:1:1:0', 59: '8:1:0:1', 60: '8:0:1:1', 61: '1:8:1:0', 62: '1:8:0:1', 63: '1:1:8:0', 64: '1:1:0:8', 65: '1:0:8:1', 66: '1:0:1:8', 67: '0:8:1:1', 68: '0:1:8:1', 69: '0:1:1:8', 70: '7:2:1:0', 71: '7:2:0:1', 72: '7:1:2:0', 73: '7:1:0:2', 74: '7:0:2:1', 75: '7:0:1:2', 76: '2:7:1:0', 77: '2:7:0:1', 78: '2:1:7:0', 79: '2:1:0:7', 80: '2:0:7:1', 81: '2:0:1:7', 82: '1:7:2:0', 83: '1:7:0:2', 84: '1:2:7:0', 85: '1:2:0:7', 86: '1:0:7:2', 87: '1:0:2:7', 88: '0:7:2:1', 89: '0:7:1:2', 90: '0:2:7:1', 91: '0:2:1:7', 92: '0:1:7:2', 93: '0:1:2:7', 94: '6:3:1:0', 95: '6:3:0:1', 96: '6:1:3:0', 97: '6:1:0:3', 98: '6:0:3:1', 99: '6:0:1:3', 100: '3:6:1:0', 101: '3:6:0:1', 102: '3:1:6:0', 103: '3:1:0:6', 104: '3:0:6:1', 105: '3:0:1:6', 106: '1:6:3:0', 107: '1:6:0:3', 108: '1:3:6:0', 109: '1:3:0:6', 110: '1:0:6:3', 111: '1:0:3:6', 112: '0:6:3:1', 113: '0:6:1:3', 114: '0:3:6:1', 115: '0:3:1:6', 116: '0:1:6:3', 117: '0:1:3:6', 118: '5:4:1:0', 119: '5:4:0:1', 120: '5:1:4:0', 121: '5:1:0:4', 122: '5:0:4:1', 123: '5:0:1:4', 124: '4:5:1:0', 125: '4:5:0:1', 126: '4:1:5:0', 127: '4:1:0:5', 128: '4:0:5:1', 129: '4:0:1:5', 130: '1:5:4:0', 131: '1:5:0:4', 132: '1:4:5:0', 133: '1:4:0:5', 134: '1:0:5:4', 135: '1:0:4:5', 136: '0:5:4:1', 137: '0:5:1:4', 138: '0:4:5:1', 139: '0:4:1:5', 140: '0:1:5:4', 141: '0:1:4:5', 142: '6:2:2:0', 143: '6:2:0:2', 144: '6:0:2:2', 145: '2:6:2:0', 146: '2:6:0:2', 147: '2:2:6:0', 148: '2:2:0:6', 149: '2:0:6:2', 150: '2:0:2:6', 151: '0:6:2:2', 152: '0:2:6:2', 153: '0:2:2:6', 154: '5:3:2:0', 155: '5:3:0:2', 156: '5:2:3:0', 157: '5:2:0:3', 158: '5:0:3:2', 159: '5:0:2:3', 160: '3:5:2:0', 161: '3:5:0:2', 162: '3:2:5:0', 163: '3:2:0:5', 164: '3:0:5:2', 165: '3:0:2:5', 166: '2:5:3:0', 167: '2:5:0:3', 168: '2:3:5:0', 169: '2:3:0:5', 170: '2:0:5:3', 171: '2:0:3:5', 172: '0:5:3:2', 173: '0:5:2:3', 174: '0:3:5:2', 175: '0:3:2:5', 176: '0:2:5:3', 177: '0:2:3:5', 178: '4:4:2:0', 179: '4:4:0:2', 180: '4:2:4:0', 181: '4:2:0:4', 182: '4:0:4:2', 183: '4:0:2:4', 184: '2:4:4:0', 185: '2:4:0:4', 186: '2:0:4:4', 187: '0:4:4:2', 188: '0:4:2:4', 189: '0:2:4:4', 190: '7:1:1:1', 191: '1:7:1:1', 192: '1:1:7:1', 193: '1:1:1:7', 194: '6:2:1:1', 195: '6:1:2:1', 196: '6:1:1:2', 197: '2:6:1:1', 198: '2:1:6:1', 199: '2:1:1:6', 200: '1:6:2:1', 201: '1:6:1:2', 202: '1:2:6:1', 203: '1:2:1:6', 204: '1:1:6:2', 205: '1:1:2:6', 206: '5:3:1:1', 207: '5:1:3:1', 208: '5:1:1:3', 209: '3:5:1:1', 210: '3:1:5:1', 211: '3:1:1:5', 212: '1:5:3:1', 213: '1:5:1:3', 214: '1:3:5:1', 215: '1:3:1:5', 216: '1:1:5:3', 217: '1:1:3:5', 218: '4:4:1:1', 219: '4:1:4:1', 220: '4:1:1:4', 221: '1:4:4:1', 222: '1:4:1:4', 223: '1:1:4:4', 224: '5:2:2:1', 225: '5:2:1:2', 226: '5:1:2:2', 227: '2:5:2:1', 228: '2:5:1:2', 229: '2:2:5:1', 230: '2:2:1:5', 231: '2:1:5:2', 232: '2:1:2:5', 233: '1:5:2:2', 234: '1:2:5:2', 235: '1:2:2:5', 236: '4:1:3:2', 237: '4:1:2:3', 238: '3:4:2:1', 239: '3:4:1:2', 240: '3:2:4:1', 241: '3:2:1:4', 242: '3:1:4:2', 243: '3:1:2:4', 244: '2:4:3:1', 245: '2:4:1:3', 246: '2:3:4:1', 247: '2:3:1:4', 248: '2:1:4:3', 249: '2:1:3:4', 250: '1:4:3:2', 251: '1:4:2:3', 252: '1:3:4:2', 253: '1:3:2:4', 254: '1:2:4:3', 255: '1:2:3:4', 256: '4:3:3:0', 257: '4:3:0:3', 258: '4:0:3:3', 259: '3:4:3:0', 260: '3:4:0:3', 261: '3:3:4:0', 262: '3:3:0:4', 263: '3:0:4:3', 264: '3:0:3:4', 265: '0:4:3:3', 266: '0:3:4:3', 267: '0:3:3:4', 268: '4:3:2:1', 269: '4:3:1:2', 270: '4:2:3:1', 271: '4:2:1:3', 272: '3:3:3:1', 273: '3:3:1:3', 274: '3:1:3:3', 275: '1:3:3:3', 276: '4:2:2:2', 277: '2:4:2:2', 278: '2:2:4:2', 279: '2:2:2:4', 280: '3:3:2:2', 281: '3:2:3:2', 282: '3:2:2:3', 283: '2:3:3:2', 284: '2:3:2:3', 285: '2:2:3:3'},
            8:  {0: '8:0:0:0', 1: '0:8:0:0', 2: '0:0:8:0', 3: '0:0:0:8', 4: '7:1:0:0', 5: '7:0:1:0', 6: '7:0:0:1', 7: '1:7:0:0', 8: '1:0:7:0', 9: '1:0:0:7', 10: '0:7:1:0', 11: '0:7:0:1', 12: '0:1:7:0', 13: '0:1:0:7', 14: '0:0:7:1', 15: '0:0:1:7', 16: '6:2:0:0', 17: '6:0:2:0', 18: '6:0:0:2', 19: '2:6:0:0', 20: '2:0:6:0', 21: '2:0:0:6', 22: '0:6:2:0', 23: '0:6:0:2', 24: '0:2:6:0', 25: '0:2:0:6', 26: '0:0:6:2', 27: '0:0:2:6', 28: '5:3:0:0', 29: '5:0:3:0', 30: '5:0:0:3', 31: '3:5:0:0', 32: '3:0:5:0', 33: '3:0:0:5', 34: '0:5:3:0', 35: '0:5:0:3', 36: '0:3:5:0', 37: '0:3:0:5', 38: '0:0:5:3', 39: '0:0:3:5', 40: '4:4:0:0', 41: '4:0:4:0', 42: '4:0:0:4', 43: '0:4:4:0', 44: '0:4:0:4', 45: '0:0:4:4', 46: '6:1:1:0', 47: '6:1:0:1', 48: '6:0:1:1', 49: '1:6:1:0', 50: '1:6:0:1', 51: '1:1:6:0', 52: '1:1:0:6', 53: '1:0:6:1', 54: '1:0:1:6', 55: '0:6:1:1', 56: '0:1:6:1', 57: '0:1:1:6', 58: '5:2:1:0', 59: '5:2:0:1', 60: '5:1:2:0', 61: '5:1:0:2', 62: '5:0:2:1', 63: '5:0:1:2', 64: '2:5:1:0', 65: '2:5:0:1', 66: '2:1:5:0', 67: '2:1:0:5', 68: '2:0:5:1', 69: '2:0:1:5', 70: '1:5:2:0', 71: '1:5:0:2', 72: '1:2:5:0', 73: '1:2:0:5', 74: '1:0:5:2', 75: '1:0:2:5', 76: '0:5:2:1', 77: '0:5:1:2', 78: '0:2:5:1', 79: '0:2:1:5', 80: '0:1:5:2', 81: '0:1:2:5', 82: '4:3:1:0', 83: '4:3:0:1', 84: '4:1:3:0', 85: '4:1:0:3', 86: '4:0:3:1', 87: '4:0:1:3', 88: '3:4:1:0', 89: '3:4:0:1', 90: '3:1:4:0', 91: '3:1:0:4', 92: '3:0:4:1', 93: '3:0:1:4', 94: '1:4:3:0', 95: '1:4:0:3', 96: '1:3:4:0', 97: '1:3:0:4', 98: '1:0:4:3', 99: '1:0:3:4', 100: '0:4:3:1', 101: '0:4:1:3', 102: '0:3:4:1', 103: '0:3:1:4', 104: '0:1:4:3', 105: '0:1:3:4', 106: '5:1:1:1', 107: '1:5:1:1', 108: '1:1:5:1', 109: '1:1:1:5', 110: '4:2:1:1', 111: '4:1:2:1', 112: '4:1:1:2', 113: '2:4:1:1', 114: '2:1:4:1', 115: '2:1:1:4', 116: '1:4:2:1', 117: '1:4:1:2', 118: '1:2:4:1', 119: '1:2:1:4', 120: '1:1:4:2', 121: '1:1:2:4', 122: '3:3:1:1', 123: '3:1:3:1', 124: '3:1:1:3', 125: '1:3:3:1', 126: '1:3:1:3', 127: '1:1:3:3', 128: '4:2:2:0', 129: '4:2:0:2', 130: '4:0:2:2', 131: '2:4:2:0', 132: '2:4:0:2', 133: '2:2:4:0', 134: '2:2:0:4', 135: '2:0:4:2', 136: '2:0:2:4', 137: '0:4:2:2', 138: '0:2:4:2', 139: '0:2:2:4', 140: '3:3:2:0', 141: '3:3:0:2', 142: '3:2:3:0', 143: '3:2:0:3', 144: '3:0:3:2', 145: '3:0:2:3', 146: '2:3:3:0', 147: '2:3:0:3', 148: '2:0:3:3', 149: '0:3:3:2', 150: '0:3:2:3', 151: '0:2:3:3', 152: '3:2:2:1', 153: '3:2:1:2', 154: '3:1:2:2', 155: '2:3:2:1', 156: '2:3:1:2', 157: '2:2:3:1', 158: '2:2:1:3', 159: '2:1:3:2', 160: '2:1:2:3', 161: '1:3:2:2', 162: '1:2:3:2', 163: '1:2:2:3', 164: '2:2:2:2'},
            6:  {0: '6:0:0:0', 1: '0:6:0:0', 2: '0:0:6:0', 3: '0:0:0:6', 4: '5:1:0:0', 5: '5:0:1:0', 6: '5:0:0:1', 7: '1:5:0:0', 8: '1:0:5:0', 9: '1:0:0:5', 10: '0:5:1:0', 11: '0:5:0:1', 12: '0:1:5:0', 13: '0:1:0:5', 14: '0:0:5:1', 15: '0:0:1:5', 16: '4:2:0:0', 17: '4:0:2:0', 18: '4:0:0:2', 19: '2:4:0:0', 20: '2:0:4:0', 21: '2:0:0:4', 22: '0:4:2:0', 23: '0:4:0:2', 24: '0:2:4:0', 25: '0:2:0:4', 26: '0:0:4:2', 27: '0:0:2:4', 28: '3:3:0:0', 29: '3:0:3:0', 30: '3:0:0:3', 31: '0:3:3:0', 32: '0:3:0:3', 33: '0:0:3:3', 34: '4:1:1:0', 35: '4:1:0:1', 36: '4:0:1:1', 37: '1:4:1:0', 38: '1:4:0:1', 39: '1:1:4:0', 40: '1:1:0:4', 41: '1:0:4:1', 42: '1:0:1:4', 43: '0:4:1:1', 44: '0:1:4:1', 45: '0:1:1:4', 46: '2:0:3:1', 47: '2:0:1:3', 48: '1:3:2:0', 49: '1:3:0:2', 50: '1:2:3:0', 51: '1:2:0:3', 52: '1:0:3:2', 53: '1:0:2:3', 54: '0:3:2:1', 55: '0:3:1:2', 56: '0:2:3:1', 57: '0:2:1:3', 58: '0:1:3:2', 59: '0:1:2:3', 60: '3:1:1:1', 61: '1:3:1:1', 62: '1:1:3:1', 63: '1:1:1:3', 64: '3:2:1:0', 65: '3:2:0:1', 66: '3:1:2:0', 67: '3:1:0:2', 68: '3:0:2:1', 69: '3:0:1:2', 70: '2:3:1:0', 71: '2:3:0:1', 72: '2:1:3:0', 73: '2:1:0:3', 74: '2:2:2:0', 75: '2:2:0:2', 76: '2:0:2:2', 77: '0:2:2:2', 78: '2:2:1:1', 79: '2:1:2:1', 80: '2:1:1:2', 81: '1:2:2:1', 82: '1:2:1:2', 83: '1:1:2:2'}
        }
        Cpdna_k_set = { l: Cpdna_set[self.k][l] for l in range(self.alp_size)}
        
        '''The transform relationship within symbol,bit,CpDNA in differernt resolution k'''
        Symbol2bit = { 64: [1, 6], 84: [1, 8], 128: [1, 7], 258: [1, 8] }
        GFint = [2, Symbol2bit[self.alp_size][1]]
        CpDNA2bit = { 64: [1, 6], 84:[3, 19], 128: [1, 7], 258: [1, 8] }        
        
        return Cpdna_k_set, Symbol2bit[self.alp_size], GFint, CpDNA2bit[self.alp_size]


    def Simulate_matrix(self, matrix_infr: list) -> list:
        '''#TODO: Simulate in one matrix containing multi-matrix_crc32''' 
        matirx_infr = []
        for read_list in matrix_infr:
            rel = Simulate_wet_pipeline.Simulate_read(self, read_list)
            matirx_infr.append(rel)
        return matirx_infr


    def Simulate_read(self, read_infr: list) -> list:
        """
        #TODO: Simulate the one read contain 150 letters
        :param read_infr = [read_id: int, read_depth: int, read_cpdna: list]
        :return: parameters: list = [read_id: int, read_depth: int, real_depth: int, normcpdna: list].
        :return: normcpdna: [ Max_fre_sample: str, cpdna: str, distance: float ]
        """
        sequence_read = []
        for syn_cpdna in read_infr[-1]: 
            sequence_read.append(Simulate_wet_pipeline.Simulate_CpDNA(self, syn_cpdna, read_infr[1]))
        sequence_readT = np.array(sequence_read).T

        sequence_read_filt = [l for l in sequence_readT]
        real_dp = len(sequence_read_filt)
        
        '''#TODO: To infer composite letter from observed frequencies with MAP'''
        norm2CpDNA = []
        for l in range(self.read_size):
            Oligo_count = {ll: 0 for ll in self.Oligo_Order }
            for ll in range(real_dp): Oligo_count[sequence_read_filt[ll][l]] += 1
            Clo_NormCpDNA = Simulate_wet_pipeline.Sample2Infer_MAP(self, [Oligo_count[ll] for ll in self.Oligo_Order]) #[ Max_fre_sample: str, cpdna: str, distance: float ]
            norm2CpDNA.append(Clo_NormCpDNA)
        return [read_infr[0], read_infr[1], real_dp, norm2CpDNA]


    def Simulate_CpDNA(self, cpdna_refer: str, dp_refer: int) -> list:
        """
        #TODO: Simulate the one composite letter
        :param cpdna_refer: str, such as '1:2:3:4'
        :param dp_refer: int, sequencing depth
        :return: simulate oligo in wet pipeline.
        """
        cpdna_val = list(map(int, re.findall(r'\d+', cpdna_refer)))
        ratio =  { i : r/self.k for i,r in zip(self.Oligo_Order, cpdna_val) }
        fmol_val = { r: int(ratio[r]*self.fmol) for r in ratio }
        sumfmol = sum(list(fmol_val.values()))
        if sumfmol != self.fmol:
            maxoligo = sorted(ratio.items(), key=lambda l: l[1], reverse=True)[0][0]
            fmol_val[maxoligo] += self.fmol-sumfmol if sumfmol < self.fmol else sumfmol-self.fmol
        sum2fmol = [ sum([fmol_val[self.Oligo_Order[ll]] for ll in range(l+1)]) for l in range(4) ]
        
        seedoligo_val = random.sample(range(self.fmol), dp_refer)
        
        '''ToDO: Introduce the nucleotide error, e.g mismatch, insert and delete'''            
        del_err, insert_err = 7.96*pow(10, 4), 3.43*pow(10, 4) #self-setup; Calculated from real data
        total_err = del_err + insert_err + self.err_rate
        
        mismatch_allot, insert_allot = np.array([1/3, 1/3, 1/3]), np.array([insert_err/4, insert_err/4, insert_err/4, insert_err/4])
        pdl_oligo = []
        for val in seedoligo_val :
            for rr in range(4):
                if val < sum2fmol[rr]: break
            simul_oligo = self.Oligo_Order[rr]
            simul_oligo_1 = self.Oligo_Order[rr]
            
            if random.uniform(0, 100) < total_err:
                if random.uniform(0, 100) < insert_err: 
                    simul_oligo = np.random.choice(['A', 'C', 'G', 'T'], p=insert_allot.ravel())
                elif random.uniform(0, 100) < insert_err + del_err:
                    simul_oligo = ''
                else:
                    simul_oligo = np.random.choice([ll for ll in self.Oligo_Order if ll != simul_oligo_1], p=mismatch_allot.ravel())
                
            pdl_oligo.append(simul_oligo)
        random.shuffle(pdl_oligo)
        return pdl_oligo


    def Sample2Infer_MAP(self, sample_observed: list) -> list:
        """
        #TODO: Select several letters with higher probability, form a set, and select the most likely letter
        :param sample_observed: list contianing the count of oligo in self.Oligo_order, like [8, 19, 32, 41]
        :return: Max_fre_sample = [ Max_fre_sample: str, cpdna: str, distance: float ]
        """
        br_nr_cpdna = Simulate_wet_pipeline.Brute_Norm(self, sample_observed)
        if ':'.join(list(map(str, br_nr_cpdna))) in self.Cpdna_list:
            candi_set = [br_nr_cpdna]
        else:
            candi_set = []
        candi_set.extend(Candidate_set(self.k, ':'.join(list(map(str, br_nr_cpdna)))))
        
        candi_fre_val = [] #[sample:list, cpdna:list, fre_val: float]
        for sub_cpdna in candi_set:
            candi_fre_val.append([ sample_observed, sub_cpdna, Simulate_wet_pipeline.Simulate_Fre(self, sub_cpdna, sample_observed) ])
        
        '''TODO: Select the letter with the MAP probability as the inferred letter'''
        candi_fre_val.sort(key=lambda l:l[-1], reverse=True)
        candi_fre_val = candi_fre_val[0]
        
        '''TODO: Simulate the Euclidean distance'''
        real_dp = sum(candi_fre_val[0])
        sample_norm = [ candi_fre_val[0][l]*self.k/real_dp  for l in range(4) ]
        distance = [ pow(candi_fre_val[1][ll]-sample_norm[ll], 2) for ll in range(4)]
        Max_fre_sample = [ ':'.join(list(map(str, candi_fre_val[0]))), ':'.join(list(map(str, candi_fre_val[1]))), sum(distance)]
        
        return Max_fre_sample


    def Brute_Norm(self, ton_dp):
        '''#TODO: Infer the most probable original letter that generates the observed frequencies via MAP #'''
        div = sum(ton_dp)/self.k
        quot_mod = {}
        quot_sum = 0
        for nn in range(4):
            quot_mod[nn] = [int(ton_dp[nn]//div), ton_dp[nn]%div]
            quot_sum += quot_mod[nn][0]
    
        oligosorted = sorted(quot_mod.items(),key=lambda l:(l[1][1], l[1][0]),reverse=True)

        diff_K_val = self.k-quot_sum
        for d_val in range(diff_K_val):
            oligosorted[d_val][1][0] += 1
        
        '''#rel : [[0, 4], [3, 3], [2, 2], [1, 1]] , 加和为K, rel = [[oligo_index, quot], [oligo_index, quot], ..., ...]'''
        rel = [[oligosorted[l][0],oligosorted[l][1][0]] for l in range(4)] 
        rel.sort(key=lambda l:l[0])
        multirel = [ rel[l][1] for l in range(4)]
        return multirel


    def Simulate_Fre(self, sim_cpdna: list, sim_dp: list) -> float:
        """
        #TODO: simulate the frequency from population to sample#
        :param sim_cpdna: list, like [1,2,3,4]
        :param sim_dp: list, like [8, 19, 32, 41]
        :return: frequent value
        """
        '''#TODO: Insert the error into fmol_sample in simulation'''
        dp2Fmol = [ int( self.fmol*( l/self.k + self.error_rate*(1-4*(l/self.k))/3 ) ) for l in sim_cpdna ]
        if sum(dp2Fmol) != self.fmol:
            dp2Fmol[dp2Fmol.index(max(dp2Fmol))] += self.fmol - sum(dp2Fmol)
        
        sub_norm_log_val = [Simulate_wet_pipeline.Combinatorial(sim_dp[l], dp2Fmol[l]) for l in range(4)]
        sub_norm_log_val.sort()
        norm_rat_sum = sum(sub_norm_log_val)
        return norm_rat_sum
    

    def Shape_Fmol_Err(self):
        cpdna_list = []
        for a in range(self.k+1):
            for c in range(self.k+1):
                for g in range(self.k+1):
                    for t in range(self.k+1):
                        if a + t + c + g == self.k:
                            cpdna_list.append([a,t,c,g])

        ratioFmol_Set = {}
        err_rat = 0.01
        for ratioK in cpdna_list:
            ratioFmol = [ int( self.fmol*( K_r/self.k + err_rat*(1-4*(K_r/self.k))/3 ) ) for K_r in ratioK ]
            if sum(ratioFmol) != self.fmol:
                ratioFmol[ratioFmol.index(max(ratioFmol))] += self.fmol - sum(ratioFmol)
            a_str = ':'.join(list(map(str, ratioK)))
            ratioFmol_Set[a_str] = ratioFmol
        return ratioFmol_Set


    @staticmethod
    def Combinatorial(sample, fmol_part):
        '''#TODO: combination number from fmol_part to sample'''
        sum1, sum2 = 0, 0
        if sample <= fmol_part/2:
            sample = fmol_part - sample
        else:
            sample = sample
            
        for f in range(sample+1,fmol_part+1):
            sum1 += math.log10(f)
            
        for f in range(1,fmol_part-sample+1):
            sum2 += math.log10(f)
            
        return sum1 - sum2



def read_args():
    """
    #TODO: Read arguments from the command line.
    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value in 6(64), 6(84), 8(128), 10(256) et al")
    parser.add_argument("-c", "--alphabet_size", required=True, type=int,
                        help="the size of alphabet count in k(=6,8,10 et al.)")
    parser.add_argument("-d", "--deep", required=True, type=int,
                        help="the inferred sequences path")
    
    parser.add_argument("-i", "--encoded_path", required=True, type=str,
                        help="the encoding seqeuences path with Derrick-cp.")
    parser.add_argument("-o", "--saved_path", required=True, type=str,
                        help="the simulated seqeuences path")
    parser.add_argument("-p", "--saved_pretreat_path", required=True, type=str,
                        help="the simulated seqeuences wirh pretreatment path")

    #required = False
    parser.add_argument("-e", "--error_rate", required=False, type=float,
                        help="The total error rate of oligo in process of DNA stroage.")
    parser.add_argument("-s", "--sample_val", required=False, type=float,
                        help="The value of sampling quality.")

    return parser.parse_args()


if __name__ == '__main__':
    '''#TODO: global variance#'''
    params = read_args()
    print("The parameters are:")
    print('Begin, simulating...')
    print("k = ", params.resolution)
    print("Depth = ", params.deep)
    print("Error rate = ", params.error_rate)  #-e 0.01 ~ 2
    print("Sample_val = ", params.sample_val)  #-s 0.1 ~ 0.9

    print()
    time_start = time.time()
    Simulate_wet = Simulate_wet_pipeline(params.resolution, params.alphabet_size, params.deep, params.error_rate, params.sample_val, rsK=41, rsN=45)
    Simulate_wet.Syn_Norm(params.encoded_path, params.saved_path, params.saved_pretreat_path)
    time_end = time.time()
    time_interval = time_end - time_start
    print('End, simulating... with {} sec.'.format(round(time_interval, 4)))

