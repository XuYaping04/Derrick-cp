# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:36:29 2023

@author: Xu Yaping
"""
import re, os, time, itertools
from numpy import packbits, unpackbits
import numpy as np
from zlib import crc32
import reedsolo as rs
from SoftRule import Soft_Rule, Soft_Coll
from copy import deepcopy
from argparse import ArgumentParser

class Seq2Digital_Decode():
    
    def __init__(self, k: int, alp_size: int, rsK: int = 41, rsN: int = 45):
        '''# TODO 0-0: Initialize the coder of Matirx consisting of ten consecutive blocks and CRC32.'''
        self.k, self.alp_size, self.matrix_size, self.read_size , self.crc= k, alp_size, 10, 150, 32
        self.rsK, self.rsN = rsK, rsN
        self.rsD = self.rsN - self.rsK
        self.rsT = self.rsD//2

        '''# TODO 0-1:  Two conversions --- Bits and GFint, Bits and letters'''
        self.Cpdna_set, self.Symbol2bit, self.GFint, self.CpDNA2bit = Seq2Digital_Decode.Basic_infr(self.k, self.alp_size)
        self.Bit2CpDNA_set, self.CpDNA2Bit_set = Seq2Digital_Decode.Trans_CpDNA2Bit(list(self.Cpdna_set.values()), self.CpDNA2bit[0])
        self.Max_Sr_Symbol, self.Max_Location, self.Max_Comloc = Seq2Digital_Decode.Basic_infr_SR(self.k)


        '''# TODO 0-2:  Simualte the padding number of bits, to comlish the mapping between bits and letters.'''
        padding_val = divmod(self.rsN * self.Symbol2bit[1], self.CpDNA2bit[1])
        if padding_val[1] != 0:
            self.Block_cpdna_size = (padding_val[0] + 1)*self.CpDNA2bit[0]
            self.padding_bits = self.CpDNA2bit[1] - padding_val[1]
        else:
            self.Block_cpdna_size = (padding_val[0])*self.CpDNA2bit[0]
            self.padding_bits = 0
        del padding_val


        #Enconding: The size of bits in one whole matrix
        self.Pld_bit_size = self.rsK * self.Symbol2bit[1]
        self.Block_bit_size = self.rsN * self.Symbol2bit[1] + self.padding_bits
        self.Matrix_bit_size = self.Pld_bit_size * self.matrix_size - self.crc
        #assert self.read_size % self.matrix_size == 0
        self.MultiMatrix_bit_size = self.Matrix_bit_size * (self.read_size//self.matrix_size)
        
        '''
        # TODO 0-3:  The basic informatio of Reed-solomon code
        #Encode
            mes_ecc = rs.rs_encode_msg(mes_en, self.rsD, gen=self.generator[self.rsD])
        #Decode
            rmes, recc, errata_pos = rs.rs_correct_msg(mes_err, nsym) #纠错后的序列rmes， 及其相应的纠错码recc， 以及其纠正的位置errata_pos
            rmes, recc, errata_pos = rs.rs_correct_msg(mes_err, nsym, erase_pos=[3, 6]) #纠错后的序列rmes， 及其相应的纠错码recc， 以及其纠正的位置errata_pos
        #Py2: 
            self.coder = rs.RSCoder(GFint=self.GFint, k=self.rsK, n=self.rsN)            
        '''
        self.prim = rs.find_prime_polys(generator = self.GFint[0], c_exp=self.GFint[1], fast_primes=True, single=True)
        self.table_rel = rs.init_tables(generator = self.GFint[0], c_exp=self.GFint[1], prim=self.prim)
        self.generator = rs.rs_generator_poly_all(self.rsN)
        
        '''# TODO 0-4: Recording '''
        self.Block_Depth, self.decode_ID = [], 0
    
    
    def Coder_Matrix(self, Matirx_Norm: list, Depth_list: list, need_log=False):
        """
        #TODO: Decode one matrix_crc32
            Read arguments from the command line.
        : param: Matirx_Norm = [ [matirx_id: int, block_cpdna: list(int), block_distance: list(float)], [..., ..., ...], ...  ]
        : param: Depth_list: list(int)
        """
        '''#TODO -0: Decode with the hard decision in RS(N, K)'''
        self.decode_ID, self.Block_Depth, Mat_Deal = 0, Depth_list, 'Failure(soft)'
        Mat_H_rel = Seq2Digital_Decode.Coder_Matrix_Hard(self, Matirx_Norm) #[ID, Decoded_result(str), [nr_pdl(list), sr_pdl(list)], Euclidean(list)]
        Mat_H_deal = [ l[1] for l in Mat_H_rel ]
        Mat_H_letter = [ l[2][1] for l in Mat_H_rel ]
        Mat_Decode = [l for l in Mat_H_letter]

        if need_log:    
            print('#Decoder result in hard decision:\t{}\t{}'.format(len(Mat_H_deal), Mat_H_deal) )
        
        if 'UnDe' not in Mat_H_deal:
            '''#TODO -1: Identify with CRC_32'''
            Mat_CRC, Mat_bits = Seq2Digital_Decode.Coder_CRC32(self, Mat_H_letter)
            if Mat_CRC:
                Mat_TF, Mat_Deal = True, 'Success(hard)'
            else:
                if 'DeEq2Coll' in Mat_H_deal:
                    '''#TODO -2: DeEq2Coll#'''
                    Mat_TF, Mat_Deal, Mat_Decode, Mat_bits = Seq2Digital_Decode.Coder_Matrix_Soft(self, ['DeEq2Coll'],  Mat_H_letter, Mat_H_rel, need_log=False)                    
                else:
                    '''#TODO -3: DeEq2Rt#'''
                    Mat_TF, Mat_Deal, Mat_Decode, Mat_bits = Seq2Digital_Decode.Coder_Matrix_Soft(self, ['DeEq2Rt'],  Mat_H_letter, Mat_H_rel, need_log=False)    
        else:
            '''#TODO -4: DeEq2Coll, UnDe, or more DeEq2Rt'''            
            Mat_TF, Mat_Deal, Mat_Decode, Mat_bits = Seq2Digital_Decode.Coder_Matrix_Soft(self, ['UnDe', 'DeEq2Coll'],  Mat_H_letter, Mat_H_rel, need_log=False)    
            
        '''#TODO -5: Transform the letter into bits'''
        if Mat_TF:
            Mat_Decode_bit = Mat_bits
        else:
            Mat_Decode_bit = '0'*self.Matrix_bit_size
        return Mat_TF, Mat_Deal, Mat_Decode, Mat_Decode_bit


    def Coder_Matrix_Hard(self,  Mat_H_letter: list):
        '''
        TODO: Decode one matrix containing ten RS blocks with Reed-Solomon(Hard decision)
        : param: Mat_H_letter = [ [matirx_id: int, block_cpdna: list(int), block_distance: list(float)], [..., ..., ...], ...  ]
        '''
        hard_rel = []
        for hard_block in Mat_H_letter:
            #Decoded_result: 'DeRt', 'DeIn', 'DeEq2Rt', DeEq2Coll', 'UnDe'#
            self.decode_ID = hard_block[0]
            block_rel = Seq2Digital_Decode.Coder_Block_Hard(self, hard_block[1])
            #[ID, Decoded_result(str), [nr_pdl(list), sr_pdl(list)], dist(list)]
            hard_rel.append( [self.decode_ID, block_rel[0], [hard_block[1], block_rel[1]], hard_block[2]] )
        return hard_rel


    def Coder_Block_Hard(self, rs_block: list):
        '''
        : param: rs_block: list(int)
        : retrun: [decoded_result(str), decoded_letter(list)]
        '''
        block_nr_GF = Seq2Digital_Decode.Letter2GFint(self, rs_block)

        '''#TODO -0: Decode each rs block with hard decision;'''
        try:
            rmes, recc, errata_pos = rs.rs_correct_msg(block_nr_GF, self.rsD)
        except:
            rmes, recc, errata_pos = [], [], []
        if rmes != []:
            block_de_GF_byte = rmes + recc
            block_de_GF = [ block_de_GF_byte[l] for l in range(self.rsN)]
            de_block_rel = Seq2Digital_Decode.GFint2Letter(self, block_de_GF)

            if len(errata_pos) == self.rsT:
                '''#TODO -1: Certain the RS block whether or not in collision#'''
                diff_nr_de_letter = [ [rs_block[l], de_block_rel[l]] for l in range(self.rsN) if de_block_rel[l] != rs_block[l]]

                SR_Oligo_IN = True
                for diff_Cpdna in diff_nr_de_letter:
                    cpdan_sr_set = Soft_Coll( self.k, diff_Cpdna[0] )
                    if diff_Cpdna[1] not in cpdan_sr_set:
                        SR_Oligo_IN = False
                        break
                                
                if SR_Oligo_IN: 
                    '''#True'''
                    return ['DeEq2Rt', de_block_rel]
                else:
                    '''#False'''
                    return ['DeEq2Coll', de_block_rel]
            else:
                '''len(errata_pos) < self.rsT:'''
                if len(errata_pos) == 0:
                    return ['DeRt', de_block_rel]
                else:
                    return ['DeIn', de_block_rel]
        else:
            return ['UnDe', []]


    def Coder_Matrix_Soft(self, Mat_H_deal: list, Mat_H_letter: list, Mat_H_rel: list, need_log=False):        
        '''#TODO -0: Decode each block individually'''
        Mat_record, Mat_Infr = {}, []
        for l in Mat_H_rel:
            if l[1] in Mat_H_deal:
                Mat_Infr.append([ l[0], l[3], l[2][0] ])
                if l[1] == 'UnDe': 
                    Mat_record[l[0]] = {}
                else: 
                    Mat_record[l[0]] = { 0: {','.join( l[2][-1]): self.rsD} }
        
        Mat_TF, Mat_decoded_ID, Mat_loc_pause, Mat_record, Sr_Mat_letter, Sr_Mat_bits = Seq2Digital_Decode.Soft_Failure(self, Mat_record, Mat_Infr, Mat_H_letter, need_log)
        if Mat_TF:
            return Mat_TF, 'Success(soft)', Sr_Mat_letter, Sr_Mat_bits
        elif not Mat_TF and 'UnDe' in Mat_H_deal:
            '''#TODO -1: Further decoding #Failure; Update the corronping information'''
            Mat_Infr, Mat_H_letter = [], []
            for l in Mat_H_rel:
                Mat_H_letter.append(l[2][1])
                if l[1] in ['UnDe', 'DeEq2Coll', 'DeEq2Rt']:
                    Mat_Infr.append([l[0], l[3], l[2][0]])
                    if l[1] == 'DeEq2Rt':
                        Mat_record[l[0]] = { 0: {','.join( l[2][-1]): 2} }
                        Mat_loc_pause[l[0]] = {}
            
            '''#TODO -2: Decoder failure and decode more rs block in soft decision. Now update the record, infr and location et al.'''
            if need_log:    print('2.2:\t{}\t{}\t{} and ID:\t{}'.format('UnDe', 'DeEq2Coll', 'DeEq2Rt', list(Mat_record.keys()) ))
            Mat_TF, Mat_loc_pause, Mat_record, Sr_Mat_letter, Sr_Mat_bits = Seq2Digital_Decode.Soft_MoreBlock_MoreErr(self, 1, self.Max_Sr_Symbol, \
                                                                                                            Mat_loc_pause, Mat_record, Mat_Infr, Mat_H_letter)
            if Mat_TF:
                Mat_TF, Mat_Deal = True, 'Success(soft)'
            else:
                '''Failure'''
                Mat_TF, Mat_Deal = False, 'Failure(soft)'
                
            return Mat_TF, Mat_Deal, Sr_Mat_letter, Sr_Mat_bits
        else:
            return False, 'Failure(soft)', Sr_Mat_letter, Sr_Mat_bits
    


    def Soft_Failure(self, Mat_record, Mat_Infr, Mat_letter, need_log=False):
        '''#TODO -0: ---RS_Block(decoder fuilure and error) = 1---'''
        if len(Mat_Infr) == 1:
            if need_log:    print('---RS_Block(decoder fuilure and error) = 1---' )
            Mat_TF, Mat_record, Mat_Sr_Block, Mat_bits = Seq2Digital_Decode.Soft_OneBlock(self, 1, Mat_record, Mat_Infr[0], Mat_letter)
            if Mat_TF:
                return True, [Mat_Infr[0][0]], {}, {}, Mat_Sr_Block, Mat_bits
            else:
                Mat_loc_pause = {}
                Mat_loc_pause[Mat_Infr[0][0]] = { l: 'End' for l in Mat_record[Mat_Infr[0][0]]}
                return False, [], Mat_loc_pause, Mat_record, Mat_letter, Mat_bits
        else:
            '''#TODO -1: ---RS_Block(decoder fuilure and error) > 1---
               #To fileter out the rs_block exsiting the self.rsT+1 error symbols, in order to narrow the error rs_block'''
            if need_log:    print('---RS_Block(decoder fuilure and error) > 1---')
            Mat_TF, Mat_ID, mat_record_list, Mat_letter, Mat_bits = Seq2Digital_Decode.Soft_MoreBlock_ErrOne(self, Mat_record, Mat_Infr, Mat_letter)
            if Mat_TF:
                return True, Mat_ID, {}, {}, Mat_letter, Mat_bits
            else: 
                '''#TODO -2: Filter the decoded rs_block with one error code , and Updata the baisc information, such as record, infr et al.'''
                Mat_record = mat_record_list[0]
                Mat_letter_1 = deepcopy(Mat_letter)
                Mat_record_1, Mat_loc_pause,  = {}, {}
                Mat_Infr_1, Finish_1_Infr = [], []
                
                for l in Mat_Infr:
                    if len(Mat_record[l[0]][1]) != 1:
                        #the sr_block with more than (self.rsT+1) error symbols
                        Mat_Infr_1.append(l)
                        Mat_record_1[l[0]] = Mat_record[l[0]]
                        Mat_loc_pause[l[0]] = {ll: 'End' for ll in Mat_record_1[l[0]]}
                    else:
                        #the sr_block with (self.rsT+1) error symbols have been decoded by def_Soft_MoreBlock_ErrOne
                        Finish_1_Infr.append(l)
                        Mat_letter_1[l[0]] = list(Mat_record[l[0]][1].keys())[0].split(',')
                        
                '''#In rare cases#'''
                if Mat_Infr_1 == []:
                    Mat_Infr_1 = Mat_Infr
                    Mat_record_1 = Mat_record
                    for l in Mat_record_1:
                        Mat_loc_pause[l] = { ll: 'End' for ll in Mat_record_1[l] if ll != 0}
                
                '''#TODO -3: Decode more block with error more than one'''
                Mat_TF, Mat_loc_pause, Mat_record_1, UnCo_Matrix_Fin, Mat_bits = Seq2Digital_Decode.Soft_MoreBlock_MoreErr(self, 2, self.Max_Sr_Symbol, \
                                                                                Mat_loc_pause, Mat_record_1, Mat_Infr_1, Mat_letter_1, need_log)
                
                if Mat_TF:
                    UnCo_decoded_ID = [l for l in range(len(Mat_letter)) if Mat_letter[l] != UnCo_Matrix_Fin[l]]
                    return True, UnCo_decoded_ID, {}, {}, UnCo_Matrix_Fin, Mat_bits
                else:
                    if Finish_1_Infr != []:
                        if need_log:    print('Decode the filtered block(=1) again' )
                        Mat_Infr_1.extend(Finish_1_Infr)
                        for fin_1 in Finish_1_Infr:
                            Mat_Infr_1.append(fin_1)
                            Mat_loc_pause[fin_1[0]] = { 1: 'End'}
                            Mat_record_1[fin_1[0]] = Mat_record[fin_1[0]]
                            
                        Mat_TF, Mat_loc_pause, Mat_record_1, UnCo_Matrix_Fin, Mat_bits = Seq2Digital_Decode.Soft_MoreBlock_MoreErr(self, 2, self.Max_Sr_Symbol, \
                                                                                                                        Mat_loc_pause, Mat_record_1, Mat_Infr_1, Mat_letter, need_log)
                        if Mat_TF:
                            UnCo_decoded_ID = [l for l in range(len(Mat_letter)) if Mat_letter[l] != UnCo_Matrix_Fin[l]]
                            return True, UnCo_decoded_ID, {}, {}, UnCo_Matrix_Fin, Mat_bits
                    else:
                        return False, [], Mat_loc_pause, Mat_record_1, Mat_letter, Mat_bits
        return False, [], Mat_loc_pause, Mat_record_1, Mat_letter, Mat_bits


    def Soft_OneBlock(self, rec_s, mat_record, mat_Infr, mat_letter, need_log=False):
        '''#TODO -0: Decode one RS block'''
        self.decode_ID = mat_Infr[0]
        for rec_cnt in range(rec_s, self.Max_Sr_Symbol+1):
            if need_log:    
                print('Decode the Block(ID={}) and correct {} error symbol.'.format(mat_Infr[0], rec_cnt) )
                
            mat_record[mat_Infr[0]][rec_cnt], idx = {}, 0
            while idx != 'End':
                idx, rel = Seq2Digital_Decode.Soft_RecCerCnt(self, rec_cnt, idx, mat_Infr[1], mat_Infr[2])
                if rel != []:
                    for sub_rel in rel:
                        '''#TODO -1: Update record, and check with CRC32#'''
                        sub_rel_str = ','.join(sub_rel)
                        if sub_rel_str not in mat_record[mat_Infr[0]][rec_cnt]:
                            mat_record[mat_Infr[0]][rec_cnt][sub_rel_str] = 1
                        else:
                            mat_record[mat_Infr[0]][rec_cnt][sub_rel_str] += 1
                        
                        mat_letter_1 = deepcopy(mat_letter)
                        mat_letter_1[ mat_Infr[0] ] = sub_rel
                        Mat_ctc_T, Mat_bits = Seq2Digital_Decode.Coder_CRC32(self, mat_letter_1)
                        if Mat_ctc_T:
                            return True, mat_record, mat_letter_1, Mat_bits
        return False, mat_record, [], ''


    def Soft_1Block_ErrOne(self, block_infr):
        '''#TODO -0: Decode one read and correct 1 error
            Read arguments from the command line.
            : param: rs_block: list(int)
            block_infr = [Pdl_Index(0), order_list(1), nr_pdl_list(2)]'''
        rec_comindex_set = Seq2Digital_Decode.Combin_Order(self, 1, block_infr[1])
        sr_set = []
        for sub_loc in range(0, len(rec_comindex_set)):
            sub_sr = Seq2Digital_Decode.Soft_CerLocation(self, block_infr[2], rec_comindex_set[sub_loc])
            sr_set.extend(sub_sr)
        
        '''#TODO -1: Further modification'''
        sr_uniq, sr_uniq_over1 = [], []
        if sr_set != []:
            '''To get uniq_sr_pdl and its count'''
            for sub_sr in sr_set:
                if sub_sr not in sr_uniq:
                    sr_uniq.append(sub_sr)
                    if sr_set.count(sub_sr) == self.rsT+1:
                        sr_uniq_over1.append(sub_sr)
            if len(sr_uniq_over1) == 1:
                return True, sr_uniq_over1
            else:
                return False, sr_uniq
        return False, sr_uniq


    def Soft_MoreBlock_ErrOne(self, mat_record, mat_infr, mat_letter, need_log=False):
        '''#TODO -0: Decode multi-block witg one letter error.'''
        if need_log:   
            print('#Decode {} block(ID={}) and correct {} error symbol'.format(len(mat_infr), ','.join(list(map(str, list(mat_record.keys())))), 1) )
            
        mat_ID, mat_record_1 = list(mat_record.keys()), deepcopy(mat_record)
        for sub_infr in mat_infr:
            self.decode_ID = sub_infr[0]
            
            mat_TF, mat_rel_set = Seq2Digital_Decode.Soft_1Block_ErrOne(self, sub_infr)
            
            if mat_TF:
                mat_record[sub_infr[0]][1] = { ','.join(m): self.rsT+1 for m in mat_rel_set}
                mat_record_1[sub_infr[0]][1] = deepcopy(mat_record[sub_infr[0]][1])
            else:
                mat_record[sub_infr[0]][1] = {}
                mat_record_1[sub_infr[0]][1] = { ','.join(m): self.rsT for m in mat_rel_set}
            if mat_rel_set != []:
                '''#TODO -1: To ensure whether or not each sr_pdl have sr_set --- RS_Eq pdl have its raw rs_pdl'''
                sr_set = []
                for M_id in mat_ID:
                    if M_id == sub_infr[0]:
                        Mbl_srblock_set = list(mat_record_1[M_id][1].keys())
                    else:
                        Mbl_srblock_set = []
                        for M_rec_cnt in mat_record_1[M_id]:
                            #mat_record_1 = {ID: {rec_cnt: {'block': cnt, 'block': cnt, ..., ...}, rec_cnt: {}, ...}, ID: {}, ..., ...}
                            Mbl_srblock_set.extend(list(mat_record_1[M_id][M_rec_cnt].keys()))
                
                    Mbl_srblock_set_list = [ l.split(',') for l in Mbl_srblock_set]
                    sr_set.append(Mbl_srblock_set_list)
                if [] not in sr_set:
                    MrPdl_O1_SR_Combin = Seq2Digital_Decode.Comb_SoftRule(sr_set)
                    for M_Sr_Combin in MrPdl_O1_SR_Combin:
                        mat_letter_1 = deepcopy(mat_letter)
                        for l in range(len(mat_ID)):
                            mat_letter_1[mat_ID[l]] = M_Sr_Combin[l]
                        
                        '''#TODO -2: Certify the CRC32 code'''
                        mat_sr_T, mat_bits = Seq2Digital_Decode.Coder_CRC32(self, mat_letter_1)
                        if mat_sr_T:
                            mat_id = []
                            for l in mat_ID:
                                if mat_letter_1[l] != mat_letter[l]:
                                    mat_id.append(l)
                            '''Effect for all UnDe and Coll_Pdl just have one Err_Code'''
                            return True, mat_id, mat_record_1, mat_letter_1, mat_bits
                    
        '''All RS_Eq have been decoded on OverOneCode'''
        return False, [], [mat_record, mat_record_1], mat_letter, ''

            
    def Soft_MoreBlock_MoreErr(self, rec_cnt_s, rec_cnt_e, mat_loc_pause, mat_record, mat_infr, mat_letter, need_log=False):
        '''#TODO -0:  Decode multi-block with error more than one'''
        if need_log:   
            print('#Decode {} block(ID={}) and correct {} ~ {} error symbol'.format(len(mat_infr), ','.join(list(map(str, list(mat_record.keys())))), rec_cnt_s, rec_cnt_e) )

        ID_list = list(mat_loc_pause.keys())
        ID_list.sort()

        for rec in range(rec_cnt_s, rec_cnt_e+1):
            '''#TODO -1: Initail the variance of mat_record and mat_loc_pause which record the candidate decoded result by soft decision and the newest location in soft decision#'''
            for l in mat_infr:
                if rec not in mat_record[l[0]]:
                    mat_record[l[0]][rec] = {}
                    mat_loc_pause[l[0]][rec] = 0
            
            '''#Rec_End_T: judge whether or not one rs_block on the cetained Over_Err_Code_Cnt have finished possible of each index and combinations#'''
            Rec_End_T = [ True if mat_loc_pause[l][rec] == 'End' else False for l in mat_loc_pause ]
            while False in Rec_End_T:
                del Rec_End_T

                for sub_infr in mat_infr: #sub_infr = [block_ID(int)(0), block_location(list)(1), block_nr_cpdna(list)(2)]
                    self.decode_ID = sub_infr[0]
                    sub_loc = mat_loc_pause[sub_infr[0]][rec]
                    
                    if sub_loc != 'End':
                        if need_log:   print('#Decode the block(ID={}) and correct {} error symbol'.format(sub_infr[0], rec))

                        sub_loc, sub_block_set = Seq2Digital_Decode.Soft_RecCerCnt(self, rec, sub_loc, sub_infr[1], sub_infr[2])
                        '''#TODO -3: Update the newest location#'''
                        mat_loc_pause[sub_infr[0]][rec] = sub_loc
                        if sub_block_set != []:
                            '''#Statistics the number of occurrence of each decoded rs_block'''
                            for sub in sub_block_set:
                                sub_str = ','.join(sub)
                                if sub_str in mat_record[sub_infr[0]][rec]:
                                    mat_record[sub_infr[0]][rec][sub_str] += 1
                                else:
                                    mat_record[sub_infr[0]][rec][sub_str]  = 1

                            '''#TODO -4: Shape candidate_Pdl_Pool for each rs_block with decoding failure#'''
                            #ID_reccnt = [[[ID_1, 0], [ID_1, 1]], [[ID_2, 1]], [[ID_3, 1], [ID_3, 2]], ..., ...]
                            ID_reccnt = []
                            ID_Empty_T = False
                            for sub_ID in ID_list:
                                if sub_ID == sub_infr[0]:
                                    '''The current decoded rs_block'''
                                    ID_reccnt.append([[sub_ID, rec]])
                                else:
                                    '''#The others decoded rs_block'''
                                    sub_ID_reccnt = []
                                    sub_Rec_Cnt = list(mat_record[sub_ID].keys())
                                    sub_Rec_Cnt.sort()
                                    for sub_rec in sub_Rec_Cnt:
                                        if mat_record[sub_ID][sub_rec] != {}:
                                            sub_ID_reccnt.append( [sub_ID, sub_rec] )
                                            
                                    '''#If there is one candidte_pool is empty, then we cannot to decode the crc_block and need to decode in futher#'''
                                    if sub_ID_reccnt == []:
                                        ID_Empty_T = True
                                        break
                                    else:
                                        ID_reccnt.append(sub_ID_reccnt)
    
                            '''#TODO -5: Each decoded rs_block have at least one candidate block, and then to shape block_combinations#'''
                            if not ID_Empty_T: 
                                ID_reccomb = Seq2Digital_Decode.Comb_SoftRule(ID_reccnt)
                                '''#Sort the candidata decoded rs_block by the sum of correction counts in positive sequence#'''
                                for t in range(len(ID_reccomb)):
                                    Com_Sum = 0
                                    for tt in ID_reccomb[t]:
                                        Com_Sum += tt[-1]
                                    ID_reccomb[t].append(Com_Sum)
                                ID_reccomb.sort(key=lambda l: l[-1])
                                
                                for t in range(len(ID_reccomb)):
                                    '''#Retain valid information---ID and Correction number  --- [[ID_1, 1], [ID_2, 0], [ID_3, 2]]#'''
                                    ID_reccomb[t] =  ID_reccomb[t][:-1]
                                    
                                    '''Extract the corresponding decoding information form mat_record'''
                                    block_set = []
                                    for tt in ID_reccomb[t]:
                                        M = [ [ l, mat_record[ tt[0] ][ tt[1] ][l] ]  for l in mat_record[ tt[0] ][ tt[1] ]]
                                        block_set.append(M)
                                    block_combin = Seq2Digital_Decode.Comb_SoftRule(block_set)
                                    
                                    '''#Sort the combinations of the decoded rs_block by the sum of occurrence counts in negative sequence#'''
                                    block_com_p = []
                                    for sub in block_combin:
                                        sum_cnt, sub_block_com = 0, []
                                        for tt in sub:
                                            sum_cnt += tt[1]
                                            sub_block_com.append(tt[0])
                                        block_com_p.append([sub_block_com, sum_cnt])
                                    block_com_p.sort(key=lambda l: l[-1], reverse=True)
                                                                    
                                    
                                    '''#TODO -6: Check with CRC32#'''
                                    ID_com = [ l[0] for l in ID_reccomb[t] ]
                                    for tt in range(0, len(block_com_p)):
                                        '''#Retain valid information --- M_Pool_Pdl_Comb_Set = [[sr_block_1, sr_block_2, sr_block_3 [], ...]'''
                                        sub_cer_rel = Seq2Digital_Decode.Cer_CRC_Block(self, block_com_p[tt][0], ID_com, mat_letter)
                                        '''one of pool_pdl_com_rel = [True, SR_Block] or [False, []]'''
                                        if sub_cer_rel[0]:
                                            return True, {}, {}, sub_cer_rel[1], sub_cer_rel[-1]

                '''#Rec_End_T:  Judge whether or not ending decoding in the certained correction Symbols'''
                Rec_End_T = [True if mat_loc_pause[l][rec] == 'End' else False for l in mat_loc_pause ]
            
            '''#TODO -7: Finish the certained correction counts, and then rec++#
                #According to the current decoding situation, then adjust the order of these blocks#'''
            empty = []
            unempty = []
            
            for l in mat_infr:
                Empty_TF = True 
                M_Block_Pass_Cnt = list(mat_loc_pause[l[0]].keys())
                M_Block_Pass_Cnt.sort()
                for ll in M_Block_Pass_Cnt:
                    if mat_record[l[0]][ll] != []:
                        '''UnEmpty'''
                        Empty_TF = False
                        break
                if Empty_TF:    empty.append(l)  #Empty
                else:   unempty.append(l) #UnEmpty
            
            '''#TODO -8: Empty in front, not empty in back#'''
            mat_infr = empty
            mat_infr.extend(unempty)
            del empty, unempty
        return False, mat_loc_pause, mat_record, [], ''


    def Soft_RecOne(self, rec_cnt, rec_start_loc, rec_loc, rec_block ):
        '''#TODO: remedy one error. separete the all candidate into little interval as "range_loc" ,then decode with reed-solomon if crc_block true then return#'''
        rec_comindex_set = Seq2Digital_Decode.Combin_Order(self, rec_cnt, rec_loc)
        pool_rel = []
        for sub_loc in range(rec_start_loc, len(rec_comindex_set)):
            sub_sr = Seq2Digital_Decode.Soft_CerLocation(self, rec_block, rec_comindex_set[sub_loc])
            pool_rel.extend(sub_sr)

        return 'End', pool_rel


    def Soft_RecCerCnt(self, rec_cnt, rec_start_loc, rec_loc, rec_block ):
        rec_comindex_set = Seq2Digital_Decode.Combin_Order(self, rec_cnt, rec_loc)
        '''#TODO: separete the all candidate into little interval as "range_loc" ,then decode with reed-solomon if crc_block true then return#'''
        for sub_loc in range(rec_start_loc, len(rec_comindex_set)):
            sub_sr = Seq2Digital_Decode.Soft_CerLocation(self, rec_block, rec_comindex_set[sub_loc])
            if sub_sr != []:
                if sub_loc+1 < len(rec_comindex_set):   return sub_loc+1, sub_sr
                else:   return 'End', sub_sr
        return 'End', []


    def Soft_CerLocation(self, loc_sr_block, loc_combin):
        '''#TODO: recify the error in several positions sorted and selected by Euclidean distance;
           Extract transition probabilities from transition library(Soft_Rule.py) by potential wrong letters.'''
        loc_sr_refer = [  Soft_Rule(self.k, self.Block_Depth[l], loc_sr_block[l] )  for l in loc_combin ] 
        loc_sr_combin = Seq2Digital_Decode.Comb_SoftRule(loc_sr_refer)
        '''
        #if loc_sr_combin == []:    return []
        '''
        dpindex_rel = []
        for sub_loc_combin in loc_sr_combin:
            loc_sr_block_1 = deepcopy(loc_sr_block)
            for rec_index in range(len(sub_loc_combin)):            
                '''#To change the tosoft_cpdna_1, under index: loc_combin and rec_cpdan: sub_loc_combin'''
                loc_sr_block_1[ loc_combin[rec_index] ] = sub_loc_combin[rec_index]
            
            '''#To decode the tosoft_cpdna_1 with Reed-Solomon#'''
            dpindex_soft_rel = Seq2Digital_Decode.Coder_Block_Hard(self, loc_sr_block_1)
            if dpindex_soft_rel[0] not in  ['UnDe', 'DeEq2Coll']:
                '''#TODO: Make sure the correction item works#'''
                softindex_effect_T = True
                for rec_index in range(len(sub_loc_combin)):
                    if dpindex_soft_rel[1][ loc_combin[rec_index] ] != sub_loc_combin[rec_index]:
                        softindex_effect_T = False
                        break
                if softindex_effect_T:
                    dpindex_rel.append(dpindex_soft_rel[1])
                    
        return dpindex_rel
    

    def Combin_Order(self, com_cnt, com_loc):
        '''#TODO: Combination'''
        limit_loc = self.Max_Location[com_cnt]
        limit_com = self.Max_Comloc[com_cnt]
        '''Reversed order'''
        loc_order = sorted(range(len(com_loc)), key=lambda l : com_loc[l], reverse=True)
        loc_order_part = loc_order[: limit_loc]
        
        combin_loc_set = Seq2Digital_Decode.Combinations_Cnt(loc_order_part, com_cnt)
        combin_loc_set_sort = []
        for sub_com in combin_loc_set: 
            sub_order_val = 0
            for tocom_index in sub_com:
                sub_order_val += com_loc[tocom_index] 
            combin_loc_set_sort.append([sub_com, sub_order_val])
            
        '''Reversed order: The greater the distance, the greater the possibility to be the error location'''
        combin_loc_set_sort.sort(key=lambda l: l[-1], reverse=True)
        tocombin_rel_sort_set = [l[0] for l in combin_loc_set_sort]
        
        return tocombin_rel_sort_set[:limit_com]


    def Cer_CRC_Block(self, Cer_sr_pdl_Com, Cer_ID, cer_Matrix):
        '''#TODO: Verify with crc_32 in multi-block.'''
        cer_sr_pool_Infr = [[ Cer_ID[l], Cer_sr_pdl_Com[l]] for l in range(len(Cer_sr_pdl_Com))]

        cer_crc_Matrix = deepcopy(cer_Matrix)
        for sub_cer_Infr in cer_sr_pool_Infr:
            cer_crc_Matrix[ sub_cer_Infr[0] ] = sub_cer_Infr[1].split(',')
            
        cer_de_T, Mat_bits = Seq2Digital_Decode.Coder_CRC32(self, cer_crc_Matrix)
        if cer_de_T:
            return [True, cer_crc_Matrix, Mat_bits]
        else:
            return [False, [], '']


    def Coder_CRC32(self, CRC_Cpdna):
        '''#TODO:  Verify one matrix with CRC32'''
        CRC_Bit_str = ''
        count = -1
        for block_cpdna in CRC_Cpdna:
            count += 1
            #TODO: letter →  bits
            rs_block_bits = [ self.CpDNA2Bit_set[','.join(block_cpdna[l:l+self.CpDNA2bit[0]])].zfill(self.CpDNA2bit[1])  for l in range(0, self.Block_cpdna_size, self.CpDNA2bit[0]) ]
            block_bits = ''.join(rs_block_bits)[:-self.padding_bits] if self.padding_bits != 0 else ''.join(rs_block_bits)      
            Pld_bits = [ block_bits[l:l+self.Symbol2bit[-1]] for l in range(0, self.Pld_bit_size, self.Symbol2bit[-1])]
            CRC_Bit_str += ''.join(Pld_bits)
        
        CRC_Bit_payload = CRC_Bit_str[:-32]
        CRC_Bit_utf =  CRC_Bit_payload.encode('utf-8')
        CRC_digital = crc32(CRC_Bit_utf)
        CRC_bit = '{:08b}'.format(CRC_digital).zfill(32)
        
        if CRC_bit == CRC_Bit_str[-32:]:
            return True, CRC_Bit_payload
        else:
            return False, ''


    def Letter2GFint(self,  block_cpdna: list):
        '''#TODO: Convert the letters into GFint in decoding'''
        rs_block_bits = [ self.CpDNA2Bit_set[','.join(block_cpdna[l:l+self.CpDNA2bit[0]])].zfill(self.CpDNA2bit[1])  for l in range(0, self.Block_cpdna_size, self.CpDNA2bit[0])]
        block_bits = ''.join(rs_block_bits)[:-self.padding_bits] if self.padding_bits != 0 else ''.join(rs_block_bits)
        Block_GFint = [ int(block_bits[l:l+self.Symbol2bit[-1]], 2) for l in range(0, self.Block_bit_size, self.Symbol2bit[-1])]
        return Block_GFint


    def GFint2Letter(self,  block_GFint: list) -> list:
        '''#TODO: Convert the GFint into letters #GFint2Bit and padding bits in decoding'''
        Block_bits = ''.join([ bin(l).split('b')[-1].zfill(self.Symbol2bit[1]) for l in block_GFint ]) + '0'*self.padding_bits
        '''#Bits2CpDNA'''
        Block_CpDNA = []
        for l in range(0, len(Block_bits), self.CpDNA2bit[1]):
            CpDNA_str = Block_bits[l:l+self.CpDNA2bit[1]].lstrip('0')
            CpDNA_val = CpDNA_str if CpDNA_str != '' else '0'
            Block_CpDNA.extend(self.Bit2CpDNA_set[ CpDNA_val ])
        return Block_CpDNA        


    @staticmethod
    def Basic_infr(k: int, alp: int):
        '''#TODO: The basic information used for encode and decode.
        Cpdna_set = {
            10: {0: '1:4:4:1', 1: '1:8:0:1', 2: '0:8:2:0', 3: '1:1:2:6', 4: '2:4:4:0', 5: '5:1:3:1', 6: '5:0:5:0', 7: '6:2:1:1', 8: '7:0:2:1', 9: '3:0:1:6', 10: '2:5:0:3', 11: '0:4:4:2', 12: '1:1:7:1', 13: '3:5:1:1', 14: '5:3:2:0', 15: '0:0:4:6', 16: '6:1:1:2', 17: '2:2:0:6', 18: '0:3:5:2', 19: '0:0:10:0', 20: '8:0:1:1', 21: '0:1:0:9', 22: '1:1:1:7', 23: '1:4:5:0', 24: '2:1:1:6', 25: '0:9:1:0', 26: '6:3:1:0', 27: '9:1:0:0', 28: '2:4:3:1', 29: '0:2:6:2', 30: '2:0:5:3', 31: '0:5:5:0', 32: '1:5:3:1', 33: '5:1:0:4', 34: '1:5:1:3', 35: '5:3:0:2', 36: '0:0:1:9', 37: '1:0:5:4', 38: '0:3:6:1', 39: '1:1:8:0', 40: '4:4:1:1', 41: '0:3:1:6', 42: '4:0:0:6', 43: '0:2:3:5', 44: '4:5:1:0', 45: '2:6:2:0', 46: '1:6:0:3', 47: '2:3:5:0', 48: '0:1:4:5', 49: '0:10:0:0', 50: '5:0:4:1', 51: '5:0:2:3', 52: '2:2:5:1', 53: '5:0:0:5', 54: '4:1:0:5', 55: '0:1:3:6', 56: '1:4:3:2', 57: '2:1:6:1', 58: '3:5:0:2', 59: '3:1:1:5', 60: '0:6:3:1', 61: '4:2:4:0', 62: '2:1:5:2', 63: '1:0:4:5', 64: '2:5:2:1', 65: '2:3:4:1', 66: '1:6:3:0', 67: '8:0:2:0', 68: '2:4:1:3', 69: '4:1:4:1', 70: '0:4:1:5', 71: '2:0:0:8', 72: '4:0:6:0', 73: '6:0:1:3', 74: '5:0:1:4', 75: '0:0:9:1', 76: '4:0:1:5', 77: '0:9:0:1', 78: '6:1:3:0', 79: '0:5:4:1', 80: '1:0:6:3', 81: '3:1:4:2', 82: '0:2:0:8', 83: '1:3:1:5', 84: '1:1:3:5', 85: '2:5:3:0', 86: '3:1:6:0', 87: '3:0:2:5', 88: '6:0:2:2', 89: '0:7:3:0', 90: '0:4:2:4', 91: '0:6:4:0', 92: '0:3:2:5', 93: '2:4:0:4', 94: '7:0:1:2', 95: '1:4:1:4', 96: '1:3:0:6', 97: '7:2:0:1', 98: '0:4:0:6', 99: '0:7:2:1', 100: '5:4:1:0', 101: '1:1:5:3', 102: '0:8:1:1', 103: '7:0:3:0', 104: '7:1:2:0', 105: '5:3:1:1', 106: '1:4:2:3', 107: '4:0:4:2', 108: '2:6:0:2', 109: '10:0:0:0', 110: '0:2:1:7', 111: '1:4:0:5', 112: '0:0:2:8', 113: '3:7:0:0', 114: '2:2:1:5', 115: '0:7:0:3', 116: '0:2:2:6', 117: '1:0:3:6', 118: '1:0:0:9', 119: '0:1:6:3', 120: '5:5:0:0', 121: '7:0:0:3', 122: '4:1:3:2', 123: '3:1:2:4', 124: '4:0:5:1', 125: '2:5:1:2', 126: '2:1:0:7', 127: '0:3:0:7', 128: '7:1:1:1', 129: '2:7:0:1', 130: '0:2:7:1', 131: '3:6:0:1', 132: '5:1:4:0', 133: '1:2:5:2', 134: '8:0:0:2', 135: '0:4:6:0', 136: '2:0:8:0', 137: '3:2:1:4', 138: '8:1:1:0', 139: '6:0:4:0', 140: '5:0:3:2', 141: '4:5:0:1', 142: '6:0:3:1', 143: '0:1:1:8', 144: '3:2:5:0', 145: '0:1:8:1', 146: '6:1:0:3', 147: '6:3:0:1', 148: '2:1:4:3', 149: '0:6:1:3', 150: '2:0:7:1', 151: '6:1:2:1', 152: '2:8:0:0', 153: '5:1:1:3', 154: '2:3:1:4', 155: '4:1:2:3', 156: '1:2:6:1', 157: '3:2:0:5', 158: '9:0:0:1', 159: '1:6:2:1', 160: '2:6:1:1', 161: '1:9:0:0', 162: '3:0:6:1', 163: '9:0:1:0', 164: '5:2:1:2', 165: '1:0:8:1', 166: '3:2:4:1', 167: '0:5:1:4', 168: '0:0:7:3', 169: '2:1:7:0', 170: '1:5:2:2', 171: '0:5:3:2', 172: '6:2:2:0', 173: '0:8:0:2', 174: '0:6:2:2', 175: '0:5:0:5', 176: '3:0:7:0', 177: '3:6:1:0', 178: '4:6:0:0', 179: '0:3:7:0', 180: '0:6:0:4', 181: '2:0:3:5', 182: '0:5:2:3', 183: '1:3:4:2', 184: '2:0:4:4', 185: '2:0:2:6', 186: '2:0:1:7', 187: '4:2:0:4', 188: '7:2:1:0', 189: '1:2:7:0', 190: '1:0:9:0', 191: '1:2:3:4', 192: '0:0:5:5', 193: '1:3:2:4', 194: '1:5:0:4', 195: '1:2:0:7', 196: '1:2:4:3', 197: '6:2:0:2', 198: '2:3:0:5', 199: '3:5:2:0', 200: '8:1:0:1', 201: '7:3:0:0', 202: '1:1:4:4', 203: '0:0:3:7', 204: '1:8:1:0', 205: '1:1:0:8', 206: '0:1:9:0', 207: '1:3:6:0', 208: '1:0:1:8', 209: '3:0:5:2', 210: '1:3:5:1', 211: '8:2:0:0', 212: '3:4:2:1', 213: '0:2:5:3', 214: '1:6:1:2', 215: '2:1:2:5', 216: '7:1:0:2', 217: '0:2:8:0', 218: '0:1:5:4', 219: '4:4:2:0', 220: '5:2:0:3', 221: '1:7:0:2', 222: '1:0:2:7', 223: '3:1:0:6', 224: '0:7:1:2', 225: '0:0:8:2', 226: '3:0:0:7', 227: '1:2:2:5', 228: '0:4:5:1', 229: '1:5:4:0', 230: '5:2:2:1', 231: '3:4:1:2', 232: '3:1:5:1', 233: '1:1:6:2', 234: '0:0:0:10', 235: '6:4:0:0', 236: '2:1:3:4', 237: '2:0:6:2', 238: '1:7:1:1', 239: '0:2:4:4', 240: '5:1:2:2', 241: '0:0:6:4', 242: '4:1:1:4', 243: '2:2:6:0', 244: '1:7:2:0', 245: '4:4:0:2', 246: '4:1:5:0', 247: '2:7:1:0', 248: '0:1:2:7', 249: '1:0:7:2', 250: '1:2:1:6', 251: '6:0:0:4', 252: '4:0:2:4', 253: '5:4:0:1', 254: '0:1:7:2', 255: '5:2:3:0'},
            8: {0: '0:4:4:0', 1: '0:7:1:0', 2: '2:1:5:0', 3: '6:0:1:1', 4: '3:1:3:1', 5: '5:1:1:1', 6: '3:0:1:4', 7: '3:4:0:1', 8: '0:5:2:1', 9: '0:1:5:2', 10: '5:2:0:1', 11: '1:1:6:0', 12: '6:0:0:2', 13: '0:0:2:6', 14: '6:2:0:0', 15: '0:3:1:4', 16: '4:1:0:3', 17: '0:3:5:0', 18: '0:0:3:5', 19: '1:0:5:2', 20: '0:3:4:1', 21: '6:0:2:0', 22: '0:0:7:1', 23: '2:0:0:6', 24: '0:4:0:4', 25: '4:2:1:1', 26: '1:0:7:0', 27: '1:1:0:6', 28: '0:1:6:1', 29: '3:4:1:0', 30: '1:3:4:0', 31: '7:0:0:1', 32: '1:1:5:1', 33: '3:0:0:5', 34: '5:0:3:0', 35: '5:0:2:1', 36: '0:0:0:8', 37: '1:2:5:0', 38: '0:1:4:3', 39: '0:2:0:6', 40: '3:1:1:3', 41: '4:3:1:0', 42: '0:0:6:2', 43: '4:0:4:0', 44: '4:0:1:3', 45: '0:5:1:2', 46: '1:2:1:4', 47: '1:6:1:0', 48: '2:4:1:1', 49: '1:3:3:1', 50: '0:5:3:0', 51: '6:1:1:0', 52: '1:1:3:3', 53: '1:0:2:5', 54: '0:6:1:1', 55: '1:4:1:2', 56: '0:1:3:4', 57: '0:1:2:5', 58: '1:4:0:3', 59: '2:6:0:0', 60: '1:0:1:6', 61: '0:6:2:0', 62: '5:3:0:0', 63: '4:1:1:2', 64: '1:0:3:4', 65: '4:1:2:1', 66: '1:5:2:0', 67: '5:1:0:2', 68: '3:3:1:1', 69: '0:4:1:3', 70: '2:1:4:1', 71: '4:0:0:4', 72: '0:1:1:6', 73: '2:0:1:5', 74: '3:5:0:0', 75: '3:1:0:4', 76: '0:1:0:7', 77: '1:1:4:2', 78: '8:0:0:0', 79: '0:0:4:4', 80: '1:3:0:4', 81: '0:6:0:2', 82: '1:6:0:1', 83: '1:2:4:1', 84: '1:5:1:1', 85: '1:0:4:3', 86: '4:4:0:0', 87: '7:1:0:0', 88: '0:0:5:3', 89: '1:4:3:0', 90: '4:0:3:1', 91: '0:0:1:7', 92: '0:3:0:5', 93: '1:3:1:3', 94: '4:1:3:0', 95: '2:1:1:4', 96: '0:2:6:0', 97: '5:0:0:3', 98: '1:2:0:5', 99: '4:3:0:1', 100: '1:5:0:2', 101: '5:1:2:0', 102: '3:0:4:1', 103: '1:7:0:0', 104: '5:2:1:0', 105: '2:0:6:0', 106: '1:4:2:1', 107: '3:0:5:0', 108: '0:4:3:1', 109: '0:0:8:0', 110: '7:0:1:0', 111: '6:1:0:1', 112: '3:1:4:0', 113: '0:5:0:3', 114: '0:1:7:0', 115: '2:5:1:0', 116: '1:1:2:4', 117: '5:0:1:2', 118: '0:7:0:1', 119: '1:0:0:7', 120: '0:2:1:5', 121: '2:1:0:5', 122: '1:1:1:5', 123: '0:8:0:0', 124: '1:0:6:1', 125: '2:0:5:1', 126: '0:2:5:1', 127: '2:5:0:1'},
            6: {0: '2:4:0:0', 1: '1:1:0:4', 2: '6:0:0:0', 3: '4:0:2:0', 4: '1:5:0:0', 5: '1:1:1:3', 6: '5:0:0:1', 7: '0:3:0:3', 8: '3:3:0:0', 9: '0:0:3:3', 10: '1:0:3:2', 11: '0:2:4:0', 12: '0:0:6:0', 13: '0:2:0:4', 14: '0:0:1:5', 15: '0:0:5:1', 16: '0:0:2:4', 17: '0:6:0:0', 18: '0:4:2:0', 19: '0:1:0:5', 20: '1:1:3:1', 21: '1:4:0:1', 22: '0:1:1:4', 23: '4:1:0:1', 24: '1:3:2:0', 25: '1:0:5:0', 26: '5:1:0:0', 27: '3:0:0:3', 28: '0:2:3:1', 29: '1:2:0:3', 30: '3:0:3:0', 31: '2:0:0:4', 32: '1:0:4:1', 33: '0:2:1:3', 34: '0:0:0:6', 35: '1:4:1:0', 36: '0:5:0:1', 37: '4:0:0:2', 38: '0:4:0:2', 39: '1:3:0:2', 40: '4:2:0:0', 41: '0:1:3:2', 42: '2:0:3:1', 43: '4:1:1:0', 44: '0:1:5:0', 45: '1:3:1:1', 46: '0:4:1:1', 47: '0:3:2:1', 48: '1:0:1:4', 49: '0:0:4:2', 50: '0:1:2:3', 51: '1:1:4:0', 52: '1:0:2:3', 53: '4:0:1:1', 54: '0:3:3:0', 55: '2:0:4:0', 56: '0:3:1:2', 57: '1:2:3:0', 58: '2:0:1:3', 59: '0:5:1:0', 60: '5:0:1:0', 61: '1:0:0:5', 62: '3:1:1:1', 63: '0:1:4:1'},
        }
        '''
        Cpdna_set = {
            10: {0: '10:0:0:0', 1: '0:10:0:0', 2: '0:0:10:0', 3: '0:0:0:10', 4: '9:1:0:0', 5: '9:0:1:0', 6: '9:0:0:1', 7: '1:9:0:0', 8: '1:0:9:0', 9: '1:0:0:9', 10: '0:9:1:0', 11: '0:9:0:1', 12: '0:1:9:0', 13: '0:1:0:9', 14: '0:0:9:1', 15: '0:0:1:9', 16: '8:2:0:0', 17: '8:0:2:0', 18: '8:0:0:2', 19: '2:8:0:0', 20: '2:0:8:0', 21: '2:0:0:8', 22: '0:8:2:0', 23: '0:8:0:2', 24: '0:2:8:0', 25: '0:2:0:8', 26: '0:0:8:2', 27: '0:0:2:8', 28: '7:3:0:0', 29: '7:0:3:0', 30: '7:0:0:3', 31: '3:7:0:0', 32: '3:0:7:0', 33: '3:0:0:7', 34: '0:7:3:0', 35: '0:7:0:3', 36: '0:3:7:0', 37: '0:3:0:7', 38: '0:0:7:3', 39: '0:0:3:7', 40: '6:4:0:0', 41: '6:0:4:0', 42: '6:0:0:4', 43: '4:6:0:0', 44: '4:0:6:0', 45: '4:0:0:6', 46: '0:6:4:0', 47: '0:6:0:4', 48: '0:4:6:0', 49: '0:4:0:6', 50: '0:0:6:4', 51: '0:0:4:6', 52: '5:5:0:0', 53: '5:0:5:0', 54: '5:0:0:5', 55: '0:5:5:0', 56: '0:5:0:5', 57: '0:0:5:5', 58: '8:1:1:0', 59: '8:1:0:1', 60: '8:0:1:1', 61: '1:8:1:0', 62: '1:8:0:1', 63: '1:1:8:0', 64: '1:1:0:8', 65: '1:0:8:1', 66: '1:0:1:8', 67: '0:8:1:1', 68: '0:1:8:1', 69: '0:1:1:8', 70: '7:2:1:0', 71: '7:2:0:1', 72: '7:1:2:0', 73: '7:1:0:2', 74: '7:0:2:1', 75: '7:0:1:2', 76: '2:7:1:0', 77: '2:7:0:1', 78: '2:1:7:0', 79: '2:1:0:7', 80: '2:0:7:1', 81: '2:0:1:7', 82: '1:7:2:0', 83: '1:7:0:2', 84: '1:2:7:0', 85: '1:2:0:7', 86: '1:0:7:2', 87: '1:0:2:7', 88: '0:7:2:1', 89: '0:7:1:2', 90: '0:2:7:1', 91: '0:2:1:7', 92: '0:1:7:2', 93: '0:1:2:7', 94: '6:3:1:0', 95: '6:3:0:1', 96: '6:1:3:0', 97: '6:1:0:3', 98: '6:0:3:1', 99: '6:0:1:3', 100: '3:6:1:0', 101: '3:6:0:1', 102: '3:1:6:0', 103: '3:1:0:6', 104: '3:0:6:1', 105: '3:0:1:6', 106: '1:6:3:0', 107: '1:6:0:3', 108: '1:3:6:0', 109: '1:3:0:6', 110: '1:0:6:3', 111: '1:0:3:6', 112: '0:6:3:1', 113: '0:6:1:3', 114: '0:3:6:1', 115: '0:3:1:6', 116: '0:1:6:3', 117: '0:1:3:6', 118: '5:4:1:0', 119: '5:4:0:1', 120: '5:1:4:0', 121: '5:1:0:4', 122: '5:0:4:1', 123: '5:0:1:4', 124: '4:5:1:0', 125: '4:5:0:1', 126: '4:1:5:0', 127: '4:1:0:5', 128: '4:0:5:1', 129: '4:0:1:5', 130: '1:5:4:0', 131: '1:5:0:4', 132: '1:4:5:0', 133: '1:4:0:5', 134: '1:0:5:4', 135: '1:0:4:5', 136: '0:5:4:1', 137: '0:5:1:4', 138: '0:4:5:1', 139: '0:4:1:5', 140: '0:1:5:4', 141: '0:1:4:5', 142: '6:2:2:0', 143: '6:2:0:2', 144: '6:0:2:2', 145: '2:6:2:0', 146: '2:6:0:2', 147: '2:2:6:0', 148: '2:2:0:6', 149: '2:0:6:2', 150: '2:0:2:6', 151: '0:6:2:2', 152: '0:2:6:2', 153: '0:2:2:6', 154: '5:3:2:0', 155: '5:3:0:2', 156: '5:2:3:0', 157: '5:2:0:3', 158: '5:0:3:2', 159: '5:0:2:3', 160: '3:5:2:0', 161: '3:5:0:2', 162: '3:2:5:0', 163: '3:2:0:5', 164: '3:0:5:2', 165: '3:0:2:5', 166: '2:5:3:0', 167: '2:5:0:3', 168: '2:3:5:0', 169: '2:3:0:5', 170: '2:0:5:3', 171: '2:0:3:5', 172: '0:5:3:2', 173: '0:5:2:3', 174: '0:3:5:2', 175: '0:3:2:5', 176: '0:2:5:3', 177: '0:2:3:5', 178: '4:4:2:0', 179: '4:4:0:2', 180: '4:2:4:0', 181: '4:2:0:4', 182: '4:0:4:2', 183: '4:0:2:4', 184: '2:4:4:0', 185: '2:4:0:4', 186: '2:0:4:4', 187: '0:4:4:2', 188: '0:4:2:4', 189: '0:2:4:4', 190: '7:1:1:1', 191: '1:7:1:1', 192: '1:1:7:1', 193: '1:1:1:7', 194: '6:2:1:1', 195: '6:1:2:1', 196: '6:1:1:2', 197: '2:6:1:1', 198: '2:1:6:1', 199: '2:1:1:6', 200: '1:6:2:1', 201: '1:6:1:2', 202: '1:2:6:1', 203: '1:2:1:6', 204: '1:1:6:2', 205: '1:1:2:6', 206: '5:3:1:1', 207: '5:1:3:1', 208: '5:1:1:3', 209: '3:5:1:1', 210: '3:1:5:1', 211: '3:1:1:5', 212: '1:5:3:1', 213: '1:5:1:3', 214: '1:3:5:1', 215: '1:3:1:5', 216: '1:1:5:3', 217: '1:1:3:5', 218: '4:4:1:1', 219: '4:1:4:1', 220: '4:1:1:4', 221: '1:4:4:1', 222: '1:4:1:4', 223: '1:1:4:4', 224: '5:2:2:1', 225: '5:2:1:2', 226: '5:1:2:2', 227: '2:5:2:1', 228: '2:5:1:2', 229: '2:2:5:1', 230: '2:2:1:5', 231: '2:1:5:2', 232: '2:1:2:5', 233: '1:5:2:2', 234: '1:2:5:2', 235: '1:2:2:5', 236: '4:1:3:2', 237: '4:1:2:3', 238: '3:4:2:1', 239: '3:4:1:2', 240: '3:2:4:1', 241: '3:2:1:4', 242: '3:1:4:2', 243: '3:1:2:4', 244: '2:4:3:1', 245: '2:4:1:3', 246: '2:3:4:1', 247: '2:3:1:4', 248: '2:1:4:3', 249: '2:1:3:4', 250: '1:4:3:2', 251: '1:4:2:3', 252: '1:3:4:2', 253: '1:3:2:4', 254: '1:2:4:3', 255: '1:2:3:4', 256: '4:3:3:0', 257: '4:3:0:3', 258: '4:0:3:3', 259: '3:4:3:0', 260: '3:4:0:3', 261: '3:3:4:0', 262: '3:3:0:4', 263: '3:0:4:3', 264: '3:0:3:4', 265: '0:4:3:3', 266: '0:3:4:3', 267: '0:3:3:4', 268: '4:3:2:1', 269: '4:3:1:2', 270: '4:2:3:1', 271: '4:2:1:3', 272: '3:3:3:1', 273: '3:3:1:3', 274: '3:1:3:3', 275: '1:3:3:3', 276: '4:2:2:2', 277: '2:4:2:2', 278: '2:2:4:2', 279: '2:2:2:4', 280: '3:3:2:2', 281: '3:2:3:2', 282: '3:2:2:3', 283: '2:3:3:2', 284: '2:3:2:3', 285: '2:2:3:3'},
            8:  {0: '8:0:0:0', 1: '0:8:0:0', 2: '0:0:8:0', 3: '0:0:0:8', 4: '7:1:0:0', 5: '7:0:1:0', 6: '7:0:0:1', 7: '1:7:0:0', 8: '1:0:7:0', 9: '1:0:0:7', 10: '0:7:1:0', 11: '0:7:0:1', 12: '0:1:7:0', 13: '0:1:0:7', 14: '0:0:7:1', 15: '0:0:1:7', 16: '6:2:0:0', 17: '6:0:2:0', 18: '6:0:0:2', 19: '2:6:0:0', 20: '2:0:6:0', 21: '2:0:0:6', 22: '0:6:2:0', 23: '0:6:0:2', 24: '0:2:6:0', 25: '0:2:0:6', 26: '0:0:6:2', 27: '0:0:2:6', 28: '5:3:0:0', 29: '5:0:3:0', 30: '5:0:0:3', 31: '3:5:0:0', 32: '3:0:5:0', 33: '3:0:0:5', 34: '0:5:3:0', 35: '0:5:0:3', 36: '0:3:5:0', 37: '0:3:0:5', 38: '0:0:5:3', 39: '0:0:3:5', 40: '4:4:0:0', 41: '4:0:4:0', 42: '4:0:0:4', 43: '0:4:4:0', 44: '0:4:0:4', 45: '0:0:4:4', 46: '6:1:1:0', 47: '6:1:0:1', 48: '6:0:1:1', 49: '1:6:1:0', 50: '1:6:0:1', 51: '1:1:6:0', 52: '1:1:0:6', 53: '1:0:6:1', 54: '1:0:1:6', 55: '0:6:1:1', 56: '0:1:6:1', 57: '0:1:1:6', 58: '5:2:1:0', 59: '5:2:0:1', 60: '5:1:2:0', 61: '5:1:0:2', 62: '5:0:2:1', 63: '5:0:1:2', 64: '2:5:1:0', 65: '2:5:0:1', 66: '2:1:5:0', 67: '2:1:0:5', 68: '2:0:5:1', 69: '2:0:1:5', 70: '1:5:2:0', 71: '1:5:0:2', 72: '1:2:5:0', 73: '1:2:0:5', 74: '1:0:5:2', 75: '1:0:2:5', 76: '0:5:2:1', 77: '0:5:1:2', 78: '0:2:5:1', 79: '0:2:1:5', 80: '0:1:5:2', 81: '0:1:2:5', 82: '4:3:1:0', 83: '4:3:0:1', 84: '4:1:3:0', 85: '4:1:0:3', 86: '4:0:3:1', 87: '4:0:1:3', 88: '3:4:1:0', 89: '3:4:0:1', 90: '3:1:4:0', 91: '3:1:0:4', 92: '3:0:4:1', 93: '3:0:1:4', 94: '1:4:3:0', 95: '1:4:0:3', 96: '1:3:4:0', 97: '1:3:0:4', 98: '1:0:4:3', 99: '1:0:3:4', 100: '0:4:3:1', 101: '0:4:1:3', 102: '0:3:4:1', 103: '0:3:1:4', 104: '0:1:4:3', 105: '0:1:3:4', 106: '5:1:1:1', 107: '1:5:1:1', 108: '1:1:5:1', 109: '1:1:1:5', 110: '4:2:1:1', 111: '4:1:2:1', 112: '4:1:1:2', 113: '2:4:1:1', 114: '2:1:4:1', 115: '2:1:1:4', 116: '1:4:2:1', 117: '1:4:1:2', 118: '1:2:4:1', 119: '1:2:1:4', 120: '1:1:4:2', 121: '1:1:2:4', 122: '3:3:1:1', 123: '3:1:3:1', 124: '3:1:1:3', 125: '1:3:3:1', 126: '1:3:1:3', 127: '1:1:3:3', 128: '4:2:2:0', 129: '4:2:0:2', 130: '4:0:2:2', 131: '2:4:2:0', 132: '2:4:0:2', 133: '2:2:4:0', 134: '2:2:0:4', 135: '2:0:4:2', 136: '2:0:2:4', 137: '0:4:2:2', 138: '0:2:4:2', 139: '0:2:2:4', 140: '3:3:2:0', 141: '3:3:0:2', 142: '3:2:3:0', 143: '3:2:0:3', 144: '3:0:3:2', 145: '3:0:2:3', 146: '2:3:3:0', 147: '2:3:0:3', 148: '2:0:3:3', 149: '0:3:3:2', 150: '0:3:2:3', 151: '0:2:3:3', 152: '3:2:2:1', 153: '3:2:1:2', 154: '3:1:2:2', 155: '2:3:2:1', 156: '2:3:1:2', 157: '2:2:3:1', 158: '2:2:1:3', 159: '2:1:3:2', 160: '2:1:2:3', 161: '1:3:2:2', 162: '1:2:3:2', 163: '1:2:2:3', 164: '2:2:2:2'},
            6:  {0: '6:0:0:0', 1: '0:6:0:0', 2: '0:0:6:0', 3: '0:0:0:6', 4: '5:1:0:0', 5: '5:0:1:0', 6: '5:0:0:1', 7: '1:5:0:0', 8: '1:0:5:0', 9: '1:0:0:5', 10: '0:5:1:0', 11: '0:5:0:1', 12: '0:1:5:0', 13: '0:1:0:5', 14: '0:0:5:1', 15: '0:0:1:5', 16: '4:2:0:0', 17: '4:0:2:0', 18: '4:0:0:2', 19: '2:4:0:0', 20: '2:0:4:0', 21: '2:0:0:4', 22: '0:4:2:0', 23: '0:4:0:2', 24: '0:2:4:0', 25: '0:2:0:4', 26: '0:0:4:2', 27: '0:0:2:4', 28: '3:3:0:0', 29: '3:0:3:0', 30: '3:0:0:3', 31: '0:3:3:0', 32: '0:3:0:3', 33: '0:0:3:3', 34: '4:1:1:0', 35: '4:1:0:1', 36: '4:0:1:1', 37: '1:4:1:0', 38: '1:4:0:1', 39: '1:1:4:0', 40: '1:1:0:4', 41: '1:0:4:1', 42: '1:0:1:4', 43: '0:4:1:1', 44: '0:1:4:1', 45: '0:1:1:4', 46: '2:0:3:1', 47: '2:0:1:3', 48: '1:3:2:0', 49: '1:3:0:2', 50: '1:2:3:0', 51: '1:2:0:3', 52: '1:0:3:2', 53: '1:0:2:3', 54: '0:3:2:1', 55: '0:3:1:2', 56: '0:2:3:1', 57: '0:2:1:3', 58: '0:1:3:2', 59: '0:1:2:3', 60: '3:1:1:1', 61: '1:3:1:1', 62: '1:1:3:1', 63: '1:1:1:3', 64: '3:2:1:0', 65: '3:2:0:1', 66: '3:1:2:0', 67: '3:1:0:2', 68: '3:0:2:1', 69: '3:0:1:2', 70: '2:3:1:0', 71: '2:3:0:1', 72: '2:1:3:0', 73: '2:1:0:3', 74: '2:2:2:0', 75: '2:2:0:2', 76: '2:0:2:2', 77: '0:2:2:2', 78: '2:2:1:1', 79: '2:1:2:1', 80: '2:1:1:2', 81: '1:2:2:1', 82: '1:2:1:2', 83: '1:1:2:2'}
        }
        
        Cpdna_k_set = { l: Cpdna_set[k][l] for l in range(alp)}
        
        '''The transform relationship within symbol,bit,CpDNA in differernt resolution k'''
        Symbol2bit = { 64: [1, 6], 84: [1, 8], 128: [1, 7], 258: [1, 8] }
        GFint = [2, Symbol2bit[alp][1]] #GFint = { 64: [2, 6], 84: [2, 8], 128: [2, 7], 256: [2, 8] }
        CpDNA2bit = { 64: [1, 6], 84:[3, 19], 128: [1, 7], 258: [1, 8] }        
        
        return Cpdna_k_set, Symbol2bit[alp], GFint, CpDNA2bit[alp]

    
    @staticmethod
    def Trans_CpDNA2Bit(Cp_set: list, CpDNA2bit: list):
        #TODO: The mapping rule between letters and bits
        CpDNA_combin = []
        for l in itertools.combinations_with_replacement(Cp_set, CpDNA2bit):
            ll_list = [ll for ll in itertools.permutations(l)]
            CpDNA_combin.extend(list(set(ll_list)))
        CpDNA_combin.sort()
        
        CpDNA_count = pow(len(Cp_set), CpDNA2bit)        
        Bit2CpDNA, fomer_bit = {}, -1
        CpDNA2Bit = {}
        for l in range(CpDNA_count):
            if fomer_bit == -1:
                bit_val = '0'
            else:
                bit_val = Seq2Digital_Decode.Binary_add(fomer_bit, '1')
            fomer_bit = bit_val
            Bit2CpDNA[bit_val] = CpDNA_combin[l]
            CpDNA2Bit[','.join(CpDNA_combin[l])] = bit_val
        return Bit2CpDNA, CpDNA2Bit

    '''    
    @staticmethod
    def Trans_CpDNA2Bit(Cp_set, CpDNA2bit):
        Bit2CpDNA, CpDNA2Bit = {}, {}
        fomer_bit = -1
        
        #Symbol is equal to three Cpdna#
        for key in Cp_set:
            if fomer_bit == -1:
                bit_val = '0'
            else:
                bit_val = Seq2Digital_Decode.Binary_add(fomer_bit, '1')
            fomer_bit = bit_val
            Bit2CpDNA[bit_val] = [key]
            CpDNA2Bit[ key ] = bit_val
        return Bit2CpDNA, CpDNA2Bit
    '''

    @staticmethod
    def Binary_add(a: str, b:str):
        '''#TODO: The basic information used for encode and decode.'''
        carry, result = 0, []
        max_len = max(len(a), len(b))
        a, b = a.zfill(max_len), b.zfill(max_len)
        
        for i in range(max_len-1, -1, -1):
            if a[i] == b[i]:
                val = '1' if carry == 1 else '0'
                carry = 1 if a[i] == '1' else 0
            else:
                val = '0' if carry == 1 else '1'
            result.append(val)            
                
        if carry == 1:
            result.append('1')
            
        return ''.join(result[::-1])


    @staticmethod
    def Basic_infr_SR(k):
        '''#TODO: basic information of soft decision'''
        Max_Sr_Symbol = { 6: 5, 8: 6, 10: 5 }
        Max_Location = {
            6: {1: 45, 2: 15, 3: 15, 4: 15, 5: 10}, 
            8: {1: 45, 2: 15, 3: 20, 4: 15, 5: 15, 6: 10}, 
            10: {1: 45, 2: 15, 3: 15, 4: 15, 5: 10, 6: 10} 
        }
        Max_Comloc = {
            6: {1: 45, 2: 105, 3: 455, 4: 300, 5: 200},
            8: {1: 45, 2: 105, 3: 1140, 4: 1360, 5: 3005, 6: 200}, 
            10: {1: 45, 2: 150, 3: 200, 4: 1360, 5: 210}
        }
        return Max_Sr_Symbol[k], Max_Location[k], Max_Comloc[k]    

    @staticmethod
    def Combinations_Cnt(candi_L, conbin_count):
        """List all combinations: choose k elements from list L"""
        n_L = len(candi_L)
        com_rel = []
        for l in range(n_L-conbin_count+1):
            if conbin_count > 1:
                newL = candi_L[l+1:]
                Comb = Seq2Digital_Decode.Combinations_Cnt(newL, conbin_count - 1)
                for item in Comb:
                    item.insert(0, candi_L[l])
                    com_rel.append(item)
            else:
                com_rel.append([candi_L[l]])
        return com_rel
    
    @staticmethod
    def Comb_SoftRule(sub_rule):
        '''#TODO:Combination #Initial all variances#'''
        max_y_idx = len(sub_rule)  
        row_max_idx = 1 
        arr_len, lst_row, lst_rst = [], [], []
        arr_idx = [0] * max_y_idx  
    
        '''#TODO: Transfrom Two dimensions array(sub_rule) into One dimension array(lst_row)#'''
        for item in sub_rule:
            _n = len(item)  
            arr_len.append(_n)  
            lst_row += item 
            row_max_idx *= _n  
    
        for row_idx in range(row_max_idx):
            for y_idx in range(max_y_idx):
                _pdt = 1
                for n in arr_len[y_idx+1:]:
                    _pdt *= n
                
                _offset = 0
                for n in arr_len[:y_idx]:
                    _offset += n
                arr_idx[y_idx] = (row_idx // _pdt) % arr_len[y_idx] + _offset
    
            _lst_tmp = []
            for idx in arr_idx:
                _lst_tmp.append(lst_row[idx])
            lst_rst.append(_lst_tmp)
        return lst_rst

    @staticmethod
    def Random_PI(input_bit):
        '''#TODO: Random bits with Pi'''
        root = os.path.abspath(".")
        Pi_path = root + '/Pi2Random.txt'
        with open(Pi_path, 'r') as f_pi:
            for line in f_pi:
                pi_bit = line.strip()
        pi_bit += len(input_bit)//len(pi_bit)*pi_bit
        
        rel_bit = []
        for r in range(len(input_bit)): 
            '''0^0 = 0, 0^1 = 1, 1^0 = 1, 1^1 = 0'''
            if input_bit[r] == '0' and pi_bit[r] == '0':
                rel_bit.append(0)
            elif input_bit[r] == '0' and pi_bit[r] == '1':
                rel_bit.append(1)
            elif input_bit[r] == '1' and pi_bit[r] == '0':
                rel_bit.append(1)
            elif input_bit[r] == '1' and pi_bit[r] == '1':
                rel_bit.append(0)
            else:
                print('Error in random with Pi!')
                print('input_bit:\t{}\t{}'.format(type(input_bit[r]), input_bit[r]))
                exit(1)
        return rel_bit

    def Decode_Soft_decision(self, file_in: str, file_out_bit: str, file_out_letter: str):
        '''#TODO: Initialize the class for soft decision#'''
        f_nr = open(file_in, 'r')
        f_letter = open(file_out_letter, 'w')
        NrCol_Infr, dp_list, decode_bit = [], [], []
        w = -1
        for line in f_nr:
            if re.search(r'^>', line):
                block_id = int(re.findall(r'\d+', line)[0])
            elif re.search(r'^CpDNA', line):
                line = line.strip().split('CpDNA:')[-1].strip()
                nr_block_cpdna = line.split(',')
            elif re.search(r'^Depth', line):
                line = line.strip().split('Depth:')[-1].strip()
                dp_list = list(map(int, line.split(',')))
                dp_list_form = [ round(l, 6) for l in dp_list]
            elif re.search(r'^Dist1', line):
                '''#TODO: Euclidean distance which were simulated in Simulate_preprocessing_pool.py'''
                nr_block_dist = line.strip().split('Dist1:')[-1].strip().split(',')
                nr_block_dist = list(map(float, nr_block_dist))
                NrCol_Infr.append([ (int(block_id)-1)%self.matrix_size, nr_block_cpdna, nr_block_dist])
                
                if block_id % 10 == 0:
                    '''To decode one CRC_Matrix in soft decision and record the decoding progress'''
                    print('#begin, {} block decoding...'.format( block_id // self.matrix_size))
                    time_start = time.time()
                    try:
                        Mat_TF, Mat_Deal_way, Mat_Decoded_letter, Mat_Decoded_bits = Seq2Digital_Decode.Coder_Matrix(self, NrCol_Infr, dp_list_form, need_log=False)
                    except:
                        Mat_TF, Mat_Deal_way, Mat_Decoded_letter, Mat_Decoded_bits = False, 'Failure', []*self.matrix_size, '0'*self.Matrix_bit_size
                    
                    time_end = time.time()                
                    if Mat_TF:
                        print('end, {} in {} sec'.format(Mat_Deal_way, round(time_end - time_start, 4)))
                    else:
                        print('end, {} in {} sec'.format(Mat_Deal_way, round(time_end - time_start, 4)))
                    
                    for l in Mat_Decoded_letter:
                        w += 1
                        f_letter.write('>contig.{}\n'.format(w))
                        f_letter.write('{}\n'.format(','.join(l)))
                                        
                    NrCol_Infr = []
                    decode_bit += Mat_Decoded_bits
            else:
                continue
        f_nr.close()
            
        '''#TODO: Derandomization with Pi'''
        decode_bit = Seq2Digital_Decode.Random_PI(decode_bit)
        ll = 0
        for l in decode_bit[::-1]:
            if l == 0:
                ll += 1
            else:
                break
        decode_bit = decode_bit[:-ll]
        decode_bit_count = len(decode_bit)

        '''#TODO: Remove the 64 bits representing the bit length of digital file'''
        for l in range(64, 0, -1):
            length_bit_str = ''.join(list(map(str, decode_bit[-l:]))) + '0'*(64-l)

            if bin(decode_bit_count-l).split('b')[1].zfill(64) == length_bit_str:
                decode_bit_np = np.array(decode_bit[:-l])
                byte_array = packbits(decode_bit_np.reshape((decode_bit_count-l) // 8, 8), axis=1).reshape(-1)
                byte_array.tofile(file=file_out_bit)
                break
        return 0


def read_args():
    """
    #TODO: Read arguments from the command line.
    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value in 6(64), 6(84), 8(128), 10(256) et al.")
    parser.add_argument("-c", "--alphabet_size", required=True, type=int,
                        help="the size of alphabet count in k(=6,8,10 et al.)")


    parser.add_argument("-i", "--inferred_path", required=True, type=str,
                        help="the inferred sequences path")
    parser.add_argument("-o", "--saved_digital_path", required=True, type=str,
                        help="the digital path with Derrick-cp.")
    parser.add_argument("-p", "--saved_letter_path", required=True, type=str,
                       help="the decoding seqeuences path with Derrick-cp.")
    

    parser.add_argument("-rsK", "--rsK", required=False, type=int,
                        help="RS(rsK, rsN)")
    parser.add_argument("-rsN", "--rsN", required=False, type=int,
                        help="RS(rsK, rsN)")
    return parser.parse_args()


if __name__ == "__main__":
    '''#TODO: global variance#'''
    params = read_args()
    print("The parameters are:")
    print("Resolution(k) = {}".format(params.resolution)) #-k
    print("Alphabet size = {}".format(params.alphabet_size)) #-c
    
    print("Inferred sequences path\t= ", params.inferred_path)  #-i
    print("Decoded digital path\t= ", params.saved_digital_path)   #-o
    print("Decoded letter path\t= ", params.saved_letter_path)   #-p
    print()
    
    Decoder_Matirx = Seq2Digital_Decode(params.resolution, params.alphabet_size, 41, 45)
    Decoder_Matirx.Decode_Soft_decision(params.inferred_path, params.saved_digital_path, params.saved_letter_path)
