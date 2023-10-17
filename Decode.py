# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:36:29 2023

@author: Xu Yaping
"""
import re, os, time
from numpy import packbits, unpackbits
import numpy as np
from zlib import crc32
import reedsolo as rs
from SoftRule import Soft_Rule, Soft_Coll
from copy import deepcopy
from argparse import ArgumentParser

class Decoder_Matrix():
    
    def __init__(self, k, rsK: 41, rsN: 45):
        """
        Initialize the coder of CRC_Matirx.
        """
        self.k, self.rsK, self.rsN = k, rsK, rsN
        self.rsD = self.rsN - self.rsK
        self.rsT = self.rsD//2

        self.Cpdna_set, self.GFint, self.Symbol2Cpdna, self.Symbol2bit = Decoder_Matrix.Basic_infr(self)
        self.Max_Sr_Symbol, self.Max_Location, self.Max_Comloc = Decoder_Matrix.Basic_infr_SR(self)
        
        self.GF2Cpdna, self.Cpdna2GF = Decoder_Matrix.Trans_GF2Int(self)
        self.Block_dna_len = self.rsN * self.Symbol2Cpdna
        self.Pld_dna_len = self.rsK * self.Symbol2Cpdna
        
        self.Block_Depth = []
        self.Matrix_len, self.read_size, self.decode_ID = 10, 150, 0
        
        '''Py3'''
        self.prim = rs.find_prime_polys(generator = self.GFint[0], c_exp=self.GFint[1], fast_primes=True, single=True)
        self.table_rel = rs.init_tables(generator = self.GFint[0], c_exp=self.GFint[1], prim=self.prim)
        self.generator = rs.rs_generator_poly_all(self.rsN)
        '''
        #Encode
        mes_ecc = rs.rs_encode_msg(mes_en, self.rsD, gen=self.generator[self.rsD])
        #Decode
        rmes, recc, errata_pos = rs.rs_correct_msg(mes_err, nsym) #纠错后的序列rmes， 及其相应的纠错码recc， 以及其纠正的位置errata_pos
        rmes, recc, errata_pos = rs.rs_correct_msg(mes_err, nsym, erase_pos=[3, 6]) #纠错后的序列rmes， 及其相应的纠错码recc， 以及其纠正的位置errata_pos
        #Py2: 
        #self.coder = rs.RSCoder(GFint=self.GFint, k=self.rsK, n=self.rsN)            
        '''

        
    '''The basic information used for encode and decode'''
    def Basic_infr(self):
        '''#The composite dna set filtered by their accuracy and used to the digital data storage (DDS), and the sizes of Cpdna set are convenient to converse between binary information and composite DNA letters#'''
        Cpdna_set = {
            10: {0: '1:4:4:1', 1: '1:8:0:1', 2: '0:8:2:0', 3: '1:1:2:6', 4: '2:4:4:0', 5: '5:1:3:1', 6: '5:0:5:0', 7: '6:2:1:1', 8: '7:0:2:1', 9: '3:0:1:6', 10: '2:5:0:3', 11: '0:4:4:2', 12: '1:1:7:1', 13: '3:5:1:1', 14: '5:3:2:0', 15: '0:0:4:6', 16: '6:1:1:2', 17: '2:2:0:6', 18: '0:3:5:2', 19: '0:0:10:0', 20: '8:0:1:1', 21: '0:1:0:9', 22: '1:1:1:7', 23: '1:4:5:0', 24: '2:1:1:6', 25: '0:9:1:0', 26: '6:3:1:0', 27: '9:1:0:0', 28: '2:4:3:1', 29: '0:2:6:2', 30: '2:0:5:3', 31: '0:5:5:0', 32: '1:5:3:1', 33: '5:1:0:4', 34: '1:5:1:3', 35: '5:3:0:2', 36: '0:0:1:9', 37: '1:0:5:4', 38: '0:3:6:1', 39: '1:1:8:0', 40: '4:4:1:1', 41: '0:3:1:6', 42: '4:0:0:6', 43: '0:2:3:5', 44: '4:5:1:0', 45: '2:6:2:0', 46: '1:6:0:3', 47: '2:3:5:0', 48: '0:1:4:5', 49: '0:10:0:0', 50: '5:0:4:1', 51: '5:0:2:3', 52: '2:2:5:1', 53: '5:0:0:5', 54: '4:1:0:5', 55: '0:1:3:6', 56: '1:4:3:2', 57: '2:1:6:1', 58: '3:5:0:2', 59: '3:1:1:5', 60: '0:6:3:1', 61: '4:2:4:0', 62: '2:1:5:2', 63: '1:0:4:5', 64: '2:5:2:1', 65: '2:3:4:1', 66: '1:6:3:0', 67: '8:0:2:0', 68: '2:4:1:3', 69: '4:1:4:1', 70: '0:4:1:5', 71: '2:0:0:8', 72: '4:0:6:0', 73: '6:0:1:3', 74: '5:0:1:4', 75: '0:0:9:1', 76: '4:0:1:5', 77: '0:9:0:1', 78: '6:1:3:0', 79: '0:5:4:1', 80: '1:0:6:3', 81: '3:1:4:2', 82: '0:2:0:8', 83: '1:3:1:5', 84: '1:1:3:5', 85: '2:5:3:0', 86: '3:1:6:0', 87: '3:0:2:5', 88: '6:0:2:2', 89: '0:7:3:0', 90: '0:4:2:4', 91: '0:6:4:0', 92: '0:3:2:5', 93: '2:4:0:4', 94: '7:0:1:2', 95: '1:4:1:4', 96: '1:3:0:6', 97: '7:2:0:1', 98: '0:4:0:6', 99: '0:7:2:1', 100: '5:4:1:0', 101: '1:1:5:3', 102: '0:8:1:1', 103: '7:0:3:0', 104: '7:1:2:0', 105: '5:3:1:1', 106: '1:4:2:3', 107: '4:0:4:2', 108: '2:6:0:2', 109: '10:0:0:0', 110: '0:2:1:7', 111: '1:4:0:5', 112: '0:0:2:8', 113: '3:7:0:0', 114: '2:2:1:5', 115: '0:7:0:3', 116: '0:2:2:6', 117: '1:0:3:6', 118: '1:0:0:9', 119: '0:1:6:3', 120: '5:5:0:0', 121: '7:0:0:3', 122: '4:1:3:2', 123: '3:1:2:4', 124: '4:0:5:1', 125: '2:5:1:2', 126: '2:1:0:7', 127: '0:3:0:7', 128: '7:1:1:1', 129: '2:7:0:1', 130: '0:2:7:1', 131: '3:6:0:1', 132: '5:1:4:0', 133: '1:2:5:2', 134: '8:0:0:2', 135: '0:4:6:0', 136: '2:0:8:0', 137: '3:2:1:4', 138: '8:1:1:0', 139: '6:0:4:0', 140: '5:0:3:2', 141: '4:5:0:1', 142: '6:0:3:1', 143: '0:1:1:8', 144: '3:2:5:0', 145: '0:1:8:1', 146: '6:1:0:3', 147: '6:3:0:1', 148: '2:1:4:3', 149: '0:6:1:3', 150: '2:0:7:1', 151: '6:1:2:1', 152: '2:8:0:0', 153: '5:1:1:3', 154: '2:3:1:4', 155: '4:1:2:3', 156: '1:2:6:1', 157: '3:2:0:5', 158: '9:0:0:1', 159: '1:6:2:1', 160: '2:6:1:1', 161: '1:9:0:0', 162: '3:0:6:1', 163: '9:0:1:0', 164: '5:2:1:2', 165: '1:0:8:1', 166: '3:2:4:1', 167: '0:5:1:4', 168: '0:0:7:3', 169: '2:1:7:0', 170: '1:5:2:2', 171: '0:5:3:2', 172: '6:2:2:0', 173: '0:8:0:2', 174: '0:6:2:2', 175: '0:5:0:5', 176: '3:0:7:0', 177: '3:6:1:0', 178: '4:6:0:0', 179: '0:3:7:0', 180: '0:6:0:4', 181: '2:0:3:5', 182: '0:5:2:3', 183: '1:3:4:2', 184: '2:0:4:4', 185: '2:0:2:6', 186: '2:0:1:7', 187: '4:2:0:4', 188: '7:2:1:0', 189: '1:2:7:0', 190: '1:0:9:0', 191: '1:2:3:4', 192: '0:0:5:5', 193: '1:3:2:4', 194: '1:5:0:4', 195: '1:2:0:7', 196: '1:2:4:3', 197: '6:2:0:2', 198: '2:3:0:5', 199: '3:5:2:0', 200: '8:1:0:1', 201: '7:3:0:0', 202: '1:1:4:4', 203: '0:0:3:7', 204: '1:8:1:0', 205: '1:1:0:8', 206: '0:1:9:0', 207: '1:3:6:0', 208: '1:0:1:8', 209: '3:0:5:2', 210: '1:3:5:1', 211: '8:2:0:0', 212: '3:4:2:1', 213: '0:2:5:3', 214: '1:6:1:2', 215: '2:1:2:5', 216: '7:1:0:2', 217: '0:2:8:0', 218: '0:1:5:4', 219: '4:4:2:0', 220: '5:2:0:3', 221: '1:7:0:2', 222: '1:0:2:7', 223: '3:1:0:6', 224: '0:7:1:2', 225: '0:0:8:2', 226: '3:0:0:7', 227: '1:2:2:5', 228: '0:4:5:1', 229: '1:5:4:0', 230: '5:2:2:1', 231: '3:4:1:2', 232: '3:1:5:1', 233: '1:1:6:2', 234: '0:0:0:10', 235: '6:4:0:0', 236: '2:1:3:4', 237: '2:0:6:2', 238: '1:7:1:1', 239: '0:2:4:4', 240: '5:1:2:2', 241: '0:0:6:4', 242: '4:1:1:4', 243: '2:2:6:0', 244: '1:7:2:0', 245: '4:4:0:2', 246: '4:1:5:0', 247: '2:7:1:0', 248: '0:1:2:7', 249: '1:0:7:2', 250: '1:2:1:6', 251: '6:0:0:4', 252: '4:0:2:4', 253: '5:4:0:1', 254: '0:1:7:2', 255: '5:2:3:0'},
            8: {0: '0:4:4:0', 1: '0:7:1:0', 2: '2:1:5:0', 3: '6:0:1:1', 4: '3:1:3:1', 5: '5:1:1:1', 6: '3:0:1:4', 7: '3:4:0:1', 8: '0:5:2:1', 9: '0:1:5:2', 10: '5:2:0:1', 11: '1:1:6:0', 12: '6:0:0:2', 13: '0:0:2:6', 14: '6:2:0:0', 15: '0:3:1:4', 16: '4:1:0:3', 17: '0:3:5:0', 18: '0:0:3:5', 19: '1:0:5:2', 20: '0:3:4:1', 21: '6:0:2:0', 22: '0:0:7:1', 23: '2:0:0:6', 24: '0:4:0:4', 25: '4:2:1:1', 26: '1:0:7:0', 27: '1:1:0:6', 28: '0:1:6:1', 29: '3:4:1:0', 30: '1:3:4:0', 31: '7:0:0:1', 32: '1:1:5:1', 33: '3:0:0:5', 34: '5:0:3:0', 35: '5:0:2:1', 36: '0:0:0:8', 37: '1:2:5:0', 38: '0:1:4:3', 39: '0:2:0:6', 40: '3:1:1:3', 41: '4:3:1:0', 42: '0:0:6:2', 43: '4:0:4:0', 44: '4:0:1:3', 45: '0:5:1:2', 46: '1:2:1:4', 47: '1:6:1:0', 48: '2:4:1:1', 49: '1:3:3:1', 50: '0:5:3:0', 51: '6:1:1:0', 52: '1:1:3:3', 53: '1:0:2:5', 54: '0:6:1:1', 55: '1:4:1:2', 56: '0:1:3:4', 57: '0:1:2:5', 58: '1:4:0:3', 59: '2:6:0:0', 60: '1:0:1:6', 61: '0:6:2:0', 62: '5:3:0:0', 63: '4:1:1:2', 64: '1:0:3:4', 65: '4:1:2:1', 66: '1:5:2:0', 67: '5:1:0:2', 68: '3:3:1:1', 69: '0:4:1:3', 70: '2:1:4:1', 71: '4:0:0:4', 72: '0:1:1:6', 73: '2:0:1:5', 74: '3:5:0:0', 75: '3:1:0:4', 76: '0:1:0:7', 77: '1:1:4:2', 78: '8:0:0:0', 79: '0:0:4:4', 80: '1:3:0:4', 81: '0:6:0:2', 82: '1:6:0:1', 83: '1:2:4:1', 84: '1:5:1:1', 85: '1:0:4:3', 86: '4:4:0:0', 87: '7:1:0:0', 88: '0:0:5:3', 89: '1:4:3:0', 90: '4:0:3:1', 91: '0:0:1:7', 92: '0:3:0:5', 93: '1:3:1:3', 94: '4:1:3:0', 95: '2:1:1:4', 96: '0:2:6:0', 97: '5:0:0:3', 98: '1:2:0:5', 99: '4:3:0:1', 100: '1:5:0:2', 101: '5:1:2:0', 102: '3:0:4:1', 103: '1:7:0:0', 104: '5:2:1:0', 105: '2:0:6:0', 106: '1:4:2:1', 107: '3:0:5:0', 108: '0:4:3:1', 109: '0:0:8:0', 110: '7:0:1:0', 111: '6:1:0:1', 112: '3:1:4:0', 113: '0:5:0:3', 114: '0:1:7:0', 115: '2:5:1:0', 116: '1:1:2:4', 117: '5:0:1:2', 118: '0:7:0:1', 119: '1:0:0:7', 120: '0:2:1:5', 121: '2:1:0:5', 122: '1:1:1:5', 123: '0:8:0:0', 124: '1:0:6:1', 125: '2:0:5:1', 126: '0:2:5:1', 127: '2:5:0:1'},
            6: {0: '2:4:0:0', 1: '1:1:0:4', 2: '6:0:0:0', 3: '4:0:2:0', 4: '1:5:0:0', 5: '1:1:1:3', 6: '5:0:0:1', 7: '0:3:0:3', 8: '3:3:0:0', 9: '0:0:3:3', 10: '1:0:3:2', 11: '0:2:4:0', 12: '0:0:6:0', 13: '0:2:0:4', 14: '0:0:1:5', 15: '0:0:5:1', 16: '0:0:2:4', 17: '0:6:0:0', 18: '0:4:2:0', 19: '0:1:0:5', 20: '1:1:3:1', 21: '1:4:0:1', 22: '0:1:1:4', 23: '4:1:0:1', 24: '1:3:2:0', 25: '1:0:5:0', 26: '5:1:0:0', 27: '3:0:0:3', 28: '0:2:3:1', 29: '1:2:0:3', 30: '3:0:3:0', 31: '2:0:0:4', 32: '1:0:4:1', 33: '0:2:1:3', 34: '0:0:0:6', 35: '1:4:1:0', 36: '0:5:0:1', 37: '4:0:0:2', 38: '0:4:0:2', 39: '1:3:0:2', 40: '4:2:0:0', 41: '0:1:3:2', 42: '2:0:3:1', 43: '4:1:1:0', 44: '0:1:5:0', 45: '1:3:1:1', 46: '0:4:1:1', 47: '0:3:2:1', 48: '1:0:1:4', 49: '0:0:4:2', 50: '0:1:2:3', 51: '1:1:4:0', 52: '1:0:2:3', 53: '4:0:1:1', 54: '0:3:3:0', 55: '2:0:4:0', 56: '0:3:1:2', 57: '1:2:3:0', 58: '2:0:1:3', 59: '0:5:1:0', 60: '5:0:1:0', 61: '1:0:0:5', 62: '3:1:1:1', 63: '0:1:4:1'},
        }
        
        '''the GFint for differernt resolution k !!!'''
        '''!!! need to change!'''
        GFint = { 6: [2, 6], 8: [2, 7], 10: [2, 8] }
        Symbol2bit = { 6: 6, 8: 7, 10: 8 }
        Symbol2Cpdna = { 6: 1, 8: 1, 10: 1 }
        
        return Cpdna_set[self.k], GFint[self.k], Symbol2Cpdna[self.k], Symbol2bit[self.k]

    def Basic_infr_SR(self):
        '''basic information of soft decision'''
        Max_Sr_Symbol = { 6: 5, 8: 6, 10: 5 }
        Max_Location = {
            6: {1: 45, 2: 15, 3: 15, 4: 15, 5: 10}, 
            8: {1: 45, 2: 15, 3: 20, 4: 15, 5: 15, 6: 10}, 
            10: {1: 45, 2: 15, 3: 15, 4: 15, 5: 10, 6: 10} 
        }
        Max_Comloc = {
            6: {1: 45, 2: 150, 3: 500, 4: 300, 5: 200},
            8: {1: 45, 2: 150, 3: 1200, 4: 1500, 5: 3000, 6: 200}, 
            10: {1: 45, 2: 150, 3: 200, 4: 1500, 5: 200}
        }
        return Max_Sr_Symbol[self.k], Max_Location[self.k], Max_Comloc[self.k]

    def Trans_GF2Int(self):
        GF2Cpdna = {}
        GF_int = -1
        if self.Symbol2Cpdna == 3:
            for d_1 in self.Cpdna_set:
                for d_2 in self.Cpdna_set:
                    for d_3 in self.Cpdna_set:
                        GF_int += 1
                        GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(self.Symbol2bit) ,','.join([self.Cpdna_set[d_1],self.Cpdna_set[d_2],self.Cpdna_set[d_3]])]
        elif self.Symbol2Cpdna == 2:
            for d_1 in self.Cpdna_set:
                for d_2 in self.Cpdna_set:
                    GF_int += 1
                    GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(self.Symbol2bit) ,','.join([self.Cpdna_set[d_1],self.Cpdna_set[d_2]])]
        elif self.Symbol2Cpdna == 1:
            for key in self.Cpdna_set:
                GF_int += 1
                GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(self.Symbol2bit) ,','.join([self.Cpdna_set[key]])]
        else:
            print('ERROR IN Function -- GF2Cpdna in Trans_GF2Int.')
            exit(1)
        Cpdna2GF = {vv[1] : v for v,vv in GF2Cpdna.items()} 
        return GF2Cpdna, Cpdna2GF

    def Trans_Oligo2GF(self, Matrix):
        Matrix_GF = []
        for block in Matrix:
            block_GF = [self.Cpdna2GF[l] for l in block]
            Matrix_GF.append(block_GF)
        return Matrix_GF
    
    
    def Coder_Matrix(self, Matirx_Norm, Depth_list, need_log=False):
        '''Initialize the depth of the Matrix'''
        self.decode_ID = 0
        self.Block_Depth = Depth_list
        CRC_Matrix_TF = 'Failure(soft)'
        
        '''To decode with the hard decision'''
        Matrix_Hard_rel = Decoder_Matrix.Coder_Hard(self, Matirx_Norm)
        Matrix_Hard_deal = [ l[1] for l in Matrix_Hard_rel ]
        Matirx_Hard = [ l[2][1] for l in Matrix_Hard_rel ]
        Matrix_Decode = [l for l in Matirx_Hard]

        if need_log:    print('#Decoder result in hard decision:\t{}\t{}'.format(len(Matrix_Hard_deal), Matrix_Hard_deal) )
        if 'UnDe' not in Matrix_Hard_deal:
            '''['DeRt', 'DeIn', 'DeEq2Rt', DeEq2Coll', 'UnDe']'''        
            crc_de_T = Decoder_Matrix.Coder_CRC32(self, Matirx_Hard)
            
            if crc_de_T:
                CRC_Matrix_TF = 'Success(hard)'
                Matrix_Decode = [l for l in Matirx_Hard]
                CRC_Block_TF = True
            else:
                '''#Only decoder Collision: decoder conflict, then filter out all RS_Block with De_Eq to decoded by soft decision#'''
                if need_log:    print('1: Existing the decoder collision in hard decision!' )
                if 'DeEq2Coll' in Matrix_Hard_deal:
                    Sr_Nr_Infr = [ [ l[0], l[3], l[2][0], ]  for l in Matrix_Hard_rel if l[1] == 'DeEq2Coll']            
                    Sr_record = { l[0]: { 0: {','.join( l[2][-1]): 2} }  for l in Matrix_Hard_rel if l[1] == 'DeEq2Coll'}
                    
                    if need_log:    print('1.1:\t{} and ID:\t{}'.format('DeEq2Coll', list(Sr_record.keys())) )
                    Sr_TF, Sr_decoded_ID, Sr_Matrix = Decoder_Matrix.Soft_OnlyColl(self, Sr_record, Sr_Nr_Infr, Matirx_Hard, need_log, need_log)
                    if Sr_TF:
                        CRC_Matrix_TF = 'Success(soft)'
                        Matrix_Decode = [l for l in Sr_Matrix]
                        for l in Sr_decoded_ID:
                            Matrix_Hard_rel[l][1] = 'DeSR'
                        CRC_Block_TF = True
                    else:
                        '''decode with #DeEq2Coll and #DeEq2Rt'''
                        CRC_Block_TF = False
                        
                else:
                    '''decode with #DeEq2Rt'''
                    Sr_Nr_Infr = [ [ l[0], l[3], l[2][0], ]  for l in Matrix_Hard_rel if l[1] == 'DeEq2Rt'] #Sr_Nr_Infr = [[block_id(0), Dist(list, 1), nr_block(list, 2)],  ..., ...]                
                    Sr_record = { l[0]: { 0: {','.join( l[2][-1]): 2} }  for l in Matrix_Hard_rel if l[1] == 'DeEq2Rt'} #Eq_Nr_record = {block_ID(rs_eq): {0: {rs_pdl(string): 2}}}

                    if need_log:    print('1.2:\t{} and ID:\t{}'.format('DeEq2Rt', list(Sr_record.keys()) ))
                    Sr_TF, Sr_decoded_ID, Sr_Matrix = Decoder_Matrix.Soft_OnlyColl(self, Sr_record, Sr_Nr_Infr, Matirx_Hard, need_log)
                    Matrix_Decode = [l for l in Sr_Matrix]
                    if Sr_TF:
                        CRC_Matrix_TF = 'Success(soft)'
                        CRC_Block_TF = True
                    else:
                        CRC_Block_TF = False
                        
        else:
            if need_log:    print('2: Existing the decoder fuilure and collision in hard decision!' )
            UnDe_Nr_Infr, Matirx_Hard, Sr_record = [], [], {}
            '''Initialize the variance for decodeing in soft decision'''
            for l in Matrix_Hard_rel:
                Matirx_Hard.append(l[2][1]) #[nr_pdl, sr_pdl]
                if l[1] in ['UnDe', 'DeEq2Coll']:
                    UnDe_Nr_Infr.append([l[0], l[3], l[2][0]])
                    if l[1] == 'UnDe':
                        Sr_record[l[0]] = {}
                    else:
                        Sr_record[l[0]] = { 0: {','.join( l[2][-1]): 2} }
                        
            if need_log:    print('2.1:\t{}\t{} and ID:\t{}'.format('UnDe', 'DeEq2Coll', list(Sr_record.keys()) ))
            Sr_TF, Sr_decoded_ID, Sr_stop_loc, Sr_record, Sr_Matrix = Decoder_Matrix.Soft_UnDe_Coll(self, Sr_record, UnDe_Nr_Infr, Matirx_Hard, need_log)
            if Sr_TF:
                Matrix_Decode = [l for l in Sr_Matrix]
                CRC_Matrix_TF = 'Success(soft)'
                CRC_Block_TF = True
            else:
                UnDe_Nr_Infr, Matirx_Hard = [], []
                for l in Matrix_Hard_rel:
                    Matirx_Hard.append(l[2][1]) #[nr_pdl, sr_pdl]
                    if l[1] in ['UnDe', 'DeEq2Coll', 'DeEq2Rt']:
                        UnDe_Nr_Infr.append([l[0], l[3], l[2][0]])
                        if l[1] == 'DeEq2Rt':
                            Sr_record[l[0]] = { 0: {','.join( l[2][-1]): 2} }
                            Sr_stop_loc[l[0]] = {}
                
                if need_log:    print('2.2:\t{}\t{}\t{} and ID:\t{}'.format('UnDe', 'DeEq2Coll', 'DeEq2Rt', list(Sr_record.keys()) ))
                Sr_TF, Sr_stop_loc, Sr_record, Sr_Matrix = Decoder_Matrix.Soft_MoreBlock_MoreErr(self, 1, self.Max_Sr_Symbol, \
                                                                                                                Sr_stop_loc, Sr_record, UnDe_Nr_Infr, Matirx_Hard)
                if Sr_TF:
                    Matrix_Decode = [l for l in Sr_Matrix]
                    CRC_Matrix_TF = 'Success(soft)'
                    CRC_Block_TF = True
                else:
                    CRC_Block_TF = False
                    
        Matrix_decode_bit = []
        for l in Matrix_Decode:
            l_int = [ self.Cpdna2GF[','.join(l[ll : ll + self.Symbol2Cpdna])] for ll in range(0, self.Pld_dna_len, self.Symbol2Cpdna) ]
            for ll in l_int:
                l_bit = bin(ll).split('b')[1].zfill(self.Symbol2bit)
            
                Matrix_decode_bit.extend(list(map(int, l_bit)))
        return CRC_Block_TF, CRC_Matrix_TF, Matrix_decode_bit[:-32]
                
    '''Decode in hard decision'''
    def Coder_Hard(self,  matirx_hard):
        '''To decode each RS Block just with Reed-Solomon(Hard decision)'''
        hard_rel = []
        for hard_block in matirx_hard:
            '''#Deal_Way_Rel: 'DeRt', 'DeIn', 'DeEq2Rt', DeEq2Coll', 'UnDe'#'''
            self.decode_ID = hard_block[0]
            block_rel = Decoder_Matrix.Coder_Hard_Block(self, hard_block[1])
            '''#[ID, deal_way(str), [nr_pdl(list), sr_pdl(list)], dist(list)]'''
            hard_rel.append( [self.decode_ID, block_rel[0], [hard_block[1], block_rel[1]], hard_block[2]] )
        return hard_rel

    '''Decode each rs block with hard and soft decision'''
    def Coder_Hard_Block(self, rs_block):
        block_nr_GF = [ self.Cpdna2GF[','.join(rs_block[l : l + self.Symbol2Cpdna ])] for l in range(0, self.Block_dna_len, self.Symbol2Cpdna) ]
        try:
            rmes, recc, errata_pos = rs.rs_correct_msg(block_nr_GF, self.rsD)
        except:
            rmes, recc, errata_pos = [], [], []

        if rmes != []:
            '''#To certain the RS whether or not collision#'''
            '''#Decoded successfully, and then to judge whether or not decoder collision#''' 
            block_de_GF_byte = rmes + recc
            block_de_GF = [ block_de_GF_byte[l] for l in range(self.rsN)]
            if len(errata_pos) == self.rsT:
                diff_nr_de_GF = [ [block_nr_GF[l], block_de_GF[l]] for l in range(self.rsN) if block_de_GF[l] != block_nr_GF[l] ]

                '''#To certain decoder conflict in further'''
                decoded_Cpdna = [ [self.GF2Cpdna[l[0]][1], self.GF2Cpdna[l[1]][1]] for l in diff_nr_de_GF]
                SR_Oligo_IN = True
                for diff_Cpdna in decoded_Cpdna:
                    cpdan_sr_set = Soft_Coll( self.k, diff_Cpdna[0] )
                    if diff_Cpdna[1] not in cpdan_sr_set:
                        SR_Oligo_IN = False
                        break
                
                
                de_block_rel = ','.join([ self.GF2Cpdna[l][1] for l in block_de_GF])
                if SR_Oligo_IN: 
                    '''#True: all decoded cpdna were conform to rules using in soft decision'''
                    return ['DeEq2Rt', de_block_rel.split(',')]
                else:
                    '''#False: There are decoded cpdna which were conform to rules using in soft decision'''
                    return ['DeEq2Coll', de_block_rel.split(',')]
            else:
                '''len(errata_pos) < self.rsT:'''
                de_block_rel = ','.join([ self.GF2Cpdna[l][1] for l in block_de_GF])
                if len(errata_pos) == 0:
                    return ['DeRt', de_block_rel.split(',')]
                else:
                    return ['DeIn', de_block_rel.split(',')]
        else:
            return ['UnDe', []]


    def Soft_OnlyColl(self, Coll_record, Coll_Infr, Coll_Matrix, need_log=False):
        if len(Coll_Infr) == 1:
            if need_log:    print('---RS_Block(decoder collision) = 1---')
            Coll_TF, Coll_record, Coll_Matrix = Decoder_Matrix.Soft_Only1Block(self, 1, Coll_record, Coll_Infr[0], Coll_Matrix)
            if Coll_TF:
                return True, [Coll_Infr[0][0]], Coll_Matrix
            else:
                return False, [], Coll_Matrix
        else:
            '''Existing more than one rs_block with decoding collision'''
            if need_log:    print('---RS_Block(decoder collision) > 1---')
            Coll_TF, Coll_decoded_ID, Coll_record_Infr, Coll_Matrix = Decoder_Matrix.Soft_MoreBlock_ErrOne(self, Coll_record, Coll_Infr, Coll_Matrix, need_log)
            
            if Coll_TF:
                return True, Coll_decoded_ID, Coll_Matrix
            else:
                if need_log:    print('At least one rs block containing more than (self.rsT+1) error symbols.' )
                '''Updata these baisc information for soft decision'''
                Coll_record = Coll_record_Infr[0]
                Coll_Matrix_1 = deepcopy(Coll_Matrix)
                Coll_record_1, Coll_Stop_loc = {}, {}
                Coll_Infr_1 = []
                for l in Coll_Infr:
                    if len(Coll_record[l[0]][1]) != 1:
                        '''the sr_block with more than self.rsT+1 error symbols'''
                        Coll_Infr_1.append(l)
                        Coll_record_1[l[0]] = Coll_record[l[0]]
                        Coll_Stop_loc[l[0]] = {ll: 'End' for ll in Coll_record_1[l[0]] if ll != 0}
                    else:
                        '''the sr_block with self.rsT+1 error symbols have been decoded by_Soft_MoreBlock_ErrOne'''
                        Coll_Matrix_1[l[0]] = list(Coll_record[l[0]][1].keys())[0].split(',')

                Coll_TF, Coll_Stop_loc, Coll_record_1, Coll_Matrix_1 = Decoder_Matrix.Soft_MoreBlock_MoreErr(self, 2, self.Max_Sr_Symbol, \
                                                                                    Coll_Stop_loc, Coll_record_1, Coll_Infr_1, Coll_Matrix_1, need_log)
                if Coll_TF:            
                    Coll_decoded_ID = [ l for l in range(len(Coll_Matrix)) if Coll_Matrix[l] != Coll_Matrix_1[l] ]
                    return True, Coll_decoded_ID, Coll_Matrix_1
                else:
                    return False, [], Coll_Matrix

        return False, [], Coll_Matrix


    def Soft_UnDe_Coll(self, UnCo_record, UnCo_Nr_Infr, UnCo_Matrix, need_log):
        if len(UnCo_Nr_Infr) == 1:
            if need_log:    print('---CRC_Matrix(decoder fuilure) = 1---' )
            UnCo_Sr_TF, UnCo_record, UnCo_Sr_Block = Decoder_Matrix.Soft_Only1Block(self, 1, UnCo_record, UnCo_Nr_Infr[0], UnCo_Matrix)
            if UnCo_Sr_TF:
                return True, [UnCo_Nr_Infr[0][0]], {}, {}, UnCo_Sr_Block
            else:
                UnCo_Stop_loc = {}
                UnCo_Stop_loc[UnCo_Nr_Infr[0][0]] = { l: 'End' for l in UnCo_record[UnCo_Nr_Infr[0][0]]}
                return False, [], UnCo_Stop_loc, UnCo_record, UnCo_Matrix
        else:
            if need_log:    print('---CRC_Matrix(decoder fuilure) > 1---')
            '''To fileter out the rs_block exsiting the self.rsT+1 error symbols, in order to narrow the error rs_block'''
            UnCo_Sr_TF, UnCo_decoded_ID, UnCo_record_list, UnCo_Matrix = Decoder_Matrix.Soft_MoreBlock_ErrOne(self, UnCo_record, UnCo_Nr_Infr, UnCo_Matrix)
            
            if UnCo_Sr_TF:
                return True, UnCo_decoded_ID, {}, {}, UnCo_Matrix
            else: 
                '''Updata these baisc information for soft decision'''
                UnCo_record = UnCo_record_list[0]
                UnCo_Matrix_1 = deepcopy(UnCo_Matrix)
                UnCo_record_1 = {}
                UnCo_Stop_loc = {}
                UnCo_Nr_Infr_1 = []
                
                Finish_1_Infr = []
                for l in UnCo_Nr_Infr:
                    if len(UnCo_record[l[0]][1]) != 1:
                        '''the sr_block with more than (self.rsT+1) error symbols'''
                        UnCo_Nr_Infr_1.append(l)
                        UnCo_record_1[l[0]] = UnCo_record[l[0]]
                        UnCo_Stop_loc[l[0]] = {ll: 'End' for ll in UnCo_record_1[l[0]]}
                    else:
                        '''the sr_block with (self.rsT+1) error symbols have been decoded by def_Soft_MoreBlock_ErrOne'''
                        Finish_1_Infr.append(l)
                        UnCo_Matrix_1[l[0]] = list(UnCo_record[l[0]][1].keys())[0].split(',')
                '''#In rare cases#'''
                if UnCo_Nr_Infr_1 == []:
                    UnCo_Nr_Infr_1 = UnCo_Nr_Infr
                    UnCo_record_1 = UnCo_record
                    for l in UnCo_record_1:
                        UnCo_Stop_loc[l] = { ll: 'End' for ll in UnCo_record_1[l] if ll != 0}

                UnCo_Sr_TF, UnCo_Stop_loc, UnCo_record_1, UnCo_Matrix_Fin = Decoder_Matrix.Soft_MoreBlock_MoreErr(self, 2, self.Max_Sr_Symbol, \
                                                                                                                UnCo_Stop_loc, UnCo_record_1, UnCo_Nr_Infr_1, UnCo_Matrix_1, need_log)
                
                if UnCo_Sr_TF:
                    UnCo_decoded_ID = [l for l in range(len(UnCo_Matrix)) if UnCo_Matrix[l] != UnCo_Matrix_Fin[l]]
                    return True, UnCo_decoded_ID, {}, {}, UnCo_Matrix_Fin
                else:
                    if Finish_1_Infr != []:
                        if need_log:    print('Decode the filtered block(=1) again' )
                        UnCo_Nr_Infr_1.extend(Finish_1_Infr)
                        for fin_1 in Finish_1_Infr:
                            UnCo_Nr_Infr_1.append(fin_1)
                            UnCo_Stop_loc[fin_1[0]] = { 1: 'End'}
                            UnCo_record_1[fin_1[0]] = UnCo_record[fin_1[0]]
                            
                        UnCo_Sr_TF, UnCo_Stop_loc, UnCo_record_1, UnCo_Matrix_Fin = Decoder_Matrix.Soft_MoreBlock_MoreErr(self, 2, self.Max_Sr_Symbol, \
                                                                                                                        UnCo_Stop_loc, UnCo_record_1, UnCo_Nr_Infr_1, UnCo_Matrix, need_log)
                        if UnCo_Sr_TF:
                            UnCo_decoded_ID = [l for l in range(len(UnCo_Matrix)) if UnCo_Matrix[l] != UnCo_Matrix_Fin[l]]
                            return True, UnCo_decoded_ID, {}, {}, UnCo_Matrix_Fin
                    else:
                        return False, [], UnCo_Stop_loc, UnCo_record_1, UnCo_Matrix
        return False, [], UnCo_Stop_loc, UnCo_record_1, UnCo_Matrix


    def Soft_Only1Block(self, One_start_loc, One_record, One_Infr, One_Matrix, need_log=False):
        self.decode_ID = One_Infr[0]
        for One_rec_cnt in range(One_start_loc, self.Max_Sr_Symbol+1):
            if need_log:    print('Decode the Block(ID={}) and correct {} error symbol.'.format(One_Infr[0], One_rec_cnt) )
            One_record[One_Infr[0]][One_rec_cnt] = {}
            One_stop_index = 0
            while One_stop_index != 'End':
                One_stop_index, One_rel_set = Decoder_Matrix.Soft_RecCerCnt(self, One_rec_cnt, One_stop_index, One_Infr[1], One_Infr[2])
                if One_rel_set != []:
                    for sub_rel in One_rel_set:
                        '''to update record'''
                        sub_rel_str = ','.join(sub_rel)
                        if sub_rel_str not in One_record[One_Infr[0]][One_rec_cnt]:
                            One_record[One_Infr[0]][One_rec_cnt][sub_rel_str] = 1
                        else:
                            One_record[One_Infr[0]][One_rec_cnt][sub_rel_str] += 1
                        
                        One_Block_pdl_1 = deepcopy(One_Matrix)
                        One_Block_pdl_1[ One_Infr[0] ] = sub_rel
                        '''#Check with CRC32#'''
                        One_De_crc_T = Decoder_Matrix.Coder_CRC32(self, One_Block_pdl_1)
                        if One_De_crc_T:
                            return True, One_record, One_Block_pdl_1
        return False, One_record, []


    def Soft_MoreBlock_ErrOne(self, Mbl_O1_record, Mbl_O1_Nr_Infr, Mbl_O1_Matrix, need_log=False):
        '''For Only_Coll:   Mbl_O1_Nr_Infr[0] = [Pdl_Index(0), order_list(1), nr_pdl_list(2)]   '''
        if need_log:   print('#Decode {} block(ID={}) and correct {} error symbol'.format(len(Mbl_O1_Nr_Infr), ','.join(list(map(str, list(Mbl_O1_record.keys())))), 1) )
        Mbl_O1_ID = list(Mbl_O1_record.keys())
        Mbl_O1_record_1 = deepcopy(Mbl_O1_record)
        for sub_Nr_Infr in Mbl_O1_Nr_Infr:
            self.decode_ID = sub_Nr_Infr[0]
            MrPdl_O1_TF, MrPdl_O1_de_pdl_set = Decoder_Matrix.Soft_1Block_ErrOne(self, sub_Nr_Infr)
            
            if MrPdl_O1_TF:
                Mbl_O1_record[sub_Nr_Infr[0]][1] = { ','.join(m): self.rsT+1 for m in MrPdl_O1_de_pdl_set}
                Mbl_O1_record_1[sub_Nr_Infr[0]][1] = { ','.join(m): self.rsT+1 for m in MrPdl_O1_de_pdl_set}
            else:
                Mbl_O1_record[sub_Nr_Infr[0]][1] = {}
                Mbl_O1_record_1[sub_Nr_Infr[0]][1] = { ','.join(m): self.rsT for m in MrPdl_O1_de_pdl_set}
            
            if MrPdl_O1_de_pdl_set != []:
                '''To ensure whether or not each sr_pdl have sr_set --- RS_Eq pdl have its raw rs_pdl'''
                Mbl_01_sr_set = []
                for M_id in Mbl_O1_ID:
                    if M_id == sub_Nr_Infr[0]:
                        Mbl_srblock_set = list(Mbl_O1_record_1[M_id][1].keys())
                    else:
                        Mbl_srblock_set = []
                        for M_rec_cnt in Mbl_O1_record_1[M_id]:\
                            #Mbl_O1_record_1 = {ID: {rec_cnt: {'block': cnt, 'block': cnt, ..., ...}, rec_cnt: {}, ...}, ID: {}, ..., ...}
                            Mbl_srblock_set.extend(list(Mbl_O1_record_1[M_id][M_rec_cnt].keys()))
    
                    Mbl_srblock_set_list = [ l.split(',') for l in Mbl_srblock_set]
                    Mbl_01_sr_set.append(Mbl_srblock_set_list)
    
                MrPdl_O1_SR_Combin = Decoder_Matrix.Comb_SoftRule(Mbl_01_sr_set)
                for M_Sr_Combin in MrPdl_O1_SR_Combin:
                    Mbl_O1_RS_Block_1 = deepcopy(Mbl_O1_Matrix)
                    for l in range(len(Mbl_O1_ID)):
                        Mbl_O1_RS_Block_1[Mbl_O1_ID[l]] = M_Sr_Combin[l]
                    
                    '''CRC32'''
                    Mbl_01_Sr_T = Decoder_Matrix.Coder_CRC32(self, Mbl_O1_RS_Block_1)
                    if Mbl_01_Sr_T:
                        Mbl_01_decoded_id = []
                        for l in Mbl_O1_ID:
                            if Mbl_O1_RS_Block_1[l] != Mbl_O1_Matrix[l]:
                                Mbl_01_decoded_id.append(l)
                        '''Effect for all UnDe and Coll_Pdl just have one Err_Code'''
                        return True, Mbl_01_decoded_id, Mbl_O1_record_1, Mbl_O1_RS_Block_1
                    
        '''All RS_Eq have been decoded on OverOneCode'''
        if need_log:    print('All block in OverOne all combinations of which are uneffect!' )
        return False, [], [Mbl_O1_record, Mbl_O1_record_1], Mbl_O1_Matrix

    def Soft_RecOne(self, rec_cnt, rec_start_loc, rec_loc, rec_block ):
        rec_comindex_set = Decoder_Matrix.Combin_Order(self, rec_cnt, rec_loc)
        '''#To separete the all candidate into little interval as "range_loc" ,then decode with reed-solomon if crc_block true then return#'''
        pool_rel = []
        for sub_loc in range(rec_start_loc, len(rec_comindex_set)):
            sub_sr = Decoder_Matrix.Soft_CerLocation(self, rec_block, rec_comindex_set[sub_loc])
            pool_rel.extend(sub_sr)

        return 'End', pool_rel

    '''#对单个序列纠正1个错误'''
    def Soft_1Block_ErrOne(self, OnePdl_Infr):
        '''OnePdl_Infr = [Pdl_Index(0), order_list(1), nr_pdl_list(2)]'''
        rec_comindex_set = Decoder_Matrix.Combin_Order(self, 1, OnePdl_Infr[1])
        OneB_Sr_set = []
        for sub_loc in range(0, len(rec_comindex_set)):
            sub_sr = Decoder_Matrix.Soft_CerLocation(self, OnePdl_Infr[2], rec_comindex_set[sub_loc])
            OneB_Sr_set.extend(sub_sr)

        #OneB_stop_index, OneB_Sr_set = Decoder_Matrix.Soft_RecCerCnt(self, 1, 0, OnePdl_Infr[1], OnePdl_Infr[2])
        OneB_Sr_uniq = []
        OneB_Sr_uniq_Over1 = []
        if OneB_Sr_set != []:
            '''To get uniq_sr_pdl and its count'''
            for sub_sr in OneB_Sr_set:
                if sub_sr not in OneB_Sr_uniq:
                    OneB_Sr_uniq.append(sub_sr)
                    if OneB_Sr_set.count(sub_sr) == self.rsT+1:
                        OneB_Sr_uniq_Over1.append(sub_sr)
            if len(OneB_Sr_uniq_Over1) == 1:
                return True, OneB_Sr_uniq_Over1
            else:
                return False, OneB_Sr_uniq
        return False, OneB_Sr_uniq

    def Soft_MoreBlock_MoreErr(self, start_rec_cnt, end_rec_cnt, Mbl_M_stop_loc, Mbl_M_record, Mbl_M_Nr_Infr, Mbl_Matrix, need_log=False):
        if need_log:   print('#Decode {} block(ID={}) and correct {} ~ {} error symbol'.format(len(Mbl_M_Nr_Infr), \
                                                                        ','.join(list(map(str, list(Mbl_M_record.keys())))), start_rec_cnt, end_rec_cnt) )
        '''#ID#'''
        M_Sr_ID_list = list(Mbl_M_stop_loc.keys())
        M_Sr_ID_list.sort()

        for sub_rec_cnt in range(start_rec_cnt, end_rec_cnt+1):
            '''#Initail the variance of Mbl_M_record and Mbl_M_stop_loc which record the candidate decoded result by soft decision and the newest location in soft decision#'''
            for l in Mbl_M_Nr_Infr:
                if sub_rec_cnt not in Mbl_M_record[l[0]]:
                    Mbl_M_record[l[0]][sub_rec_cnt] = {}
                    Mbl_M_stop_loc[l[0]][sub_rec_cnt] = 0
            
            '''#M_Rec_Cnt_Finish_T: judge whether or not one rs_block on the cetained Over_Err_Code_Cnt have finished possible of each index and combinations#'''
            M_Rec_Cnt_Finish_T = [ True if Mbl_M_stop_loc[l][sub_rec_cnt] == 'End' else False for l in Mbl_M_stop_loc ]
            while False in M_Rec_Cnt_Finish_T:
                del M_Rec_Cnt_Finish_T

                for sub_Nr_Infr in Mbl_M_Nr_Infr: #sub_Nr_Infr = [block_ID(int)(0), block_location(list)(1), block_nr_cpdna(list)(2)]
                    self.decode_ID = sub_Nr_Infr[0]
                    sub_M_stop_loc = Mbl_M_stop_loc[sub_Nr_Infr[0]][sub_rec_cnt]
                    
                    if sub_M_stop_loc != 'End':
                        if need_log:   print('#Decode the block(ID={}) and correct {} error symbol'.format(sub_Nr_Infr[0], sub_rec_cnt))

                        sub_M_stop_loc, sub_M_Sr_block_set = Decoder_Matrix.Soft_RecCerCnt(self, sub_rec_cnt, sub_M_stop_loc, sub_Nr_Infr[1], sub_Nr_Infr[2])
                        '''#Update the newest location#'''
                        Mbl_M_stop_loc[sub_Nr_Infr[0]][sub_rec_cnt] = sub_M_stop_loc
                        if sub_M_Sr_block_set != []:
                            '''#Statistics the number of occurrence of each decoded rs_block'''
                            for sub_block in sub_M_Sr_block_set:
                                sub_block_str = ','.join(sub_block)
                                if sub_block_str in Mbl_M_record[sub_Nr_Infr[0]][sub_rec_cnt]:
                                    Mbl_M_record[sub_Nr_Infr[0]][sub_rec_cnt][sub_block_str] += 1
                                else:
                                    Mbl_M_record[sub_Nr_Infr[0]][sub_rec_cnt][sub_block_str] = 1

                            '''#Shape candidate_Pdl_Pool for each rs_block with decoding failure#'''
                            #M_Sr_ID_reccnt = [[[ID_1, 0], [ID_1, 1]], [[ID_2, 1]], [[ID_3, 1], [ID_3, 2]], ..., ...]
                            M_Sr_ID_reccnt = []
                            sub_M_Sr_ID_Empty_T = False
                            for sub_Sr_ID in M_Sr_ID_list:
                                if sub_Sr_ID == sub_Nr_Infr[0]:
                                    '''The current decoded rs_block'''
                                    sub_M_Sr_ID_reccnt = [[sub_Sr_ID, sub_rec_cnt]]
                                    M_Sr_ID_reccnt.append(sub_M_Sr_ID_reccnt)
                                else:
                                    '''#The others decoded rs_block'''
                                    sub_M_Sr_ID_reccnt = []
                                    sub_Rec_Cnt_list = list(Mbl_M_record[sub_Sr_ID].keys())
                                    sub_Rec_Cnt_list.sort()
                                    for pass_rec_cnt in sub_Rec_Cnt_list:
                                        if Mbl_M_record[sub_Sr_ID][pass_rec_cnt] != {}:
                                            sub_M_Sr_ID_reccnt.append( [sub_Sr_ID, pass_rec_cnt] )
                                            
                                    '''#If there is one candidte_pool is empty, then we cannot to decode the crc_block and need to decode in futher#'''
                                    if sub_M_Sr_ID_reccnt == []:
                                        sub_M_Sr_ID_Empty_T = True
                                        break
                                    else:
                                        M_Sr_ID_reccnt.append(sub_M_Sr_ID_reccnt)
    
                            '''#Each decoded rs_block have at least one candidate block, and then to shape block_combinations#'''
                            if not sub_M_Sr_ID_Empty_T: 
                                M_Sr_ID_RecCnt_Combin = Decoder_Matrix.Comb_SoftRule(M_Sr_ID_reccnt)
                                '''#Sort the candidata decoded rs_block by the sum of correction counts in positive sequence#'''
                                for t in range(len(M_Sr_ID_RecCnt_Combin)):
                                    M_OneCom_Sum = 0
                                    for tt in M_Sr_ID_RecCnt_Combin[t]:
                                        M_OneCom_Sum += tt[-1]
                                    M_Sr_ID_RecCnt_Combin[t].append(M_OneCom_Sum)
                                M_Sr_ID_RecCnt_Combin.sort(key=lambda l: l[-1])
                                
                                for t in range(len(M_Sr_ID_RecCnt_Combin)):
                                    '''#Retain valid information---ID and Correction number  --- [[ID_1, 1], [ID_2, 0], [ID_3, 2]]#'''
                                    M_Sr_ID_RecCnt_Combin[t] =  M_Sr_ID_RecCnt_Combin[t][:-1]
                                    
                                    '''Extract the corresponding decoding information form Mbl_M_record'''
                                    M_Sr_Block_Set = []
                                    for tt in M_Sr_ID_RecCnt_Combin[t]:
                                        M = [ [ l, Mbl_M_record[ tt[0] ][ tt[1] ][l] ]  for l in Mbl_M_record[ tt[0] ][ tt[1] ]]
                                        M_Sr_Block_Set.append(M)
                                    M_Sr_Block_Combin = Decoder_Matrix.Comb_SoftRule(M_Sr_Block_Set)
                                    
                                    '''#Sort the combinations of the decoded rs_block by the sum of occurrence counts in negative sequence#'''
                                    M_Pure_block_Com = []
                                    for sub_combin in M_Sr_Block_Combin:
                                        sum_occur_cnt = 0
                                        Sub_M_Pure_block_Com = []
                                        for tt in sub_combin:
                                            sum_occur_cnt += tt[1]
                                            Sub_M_Pure_block_Com.append(tt[0])
                                        M_Pure_block_Com.append([Sub_M_Pure_block_Com,  sum_occur_cnt])
                                    M_Pure_block_Com.sort(key=lambda l: l[-1], reverse=True)
                                                                    
                                    
                                    '''#Check with CRC32 and shape the multiprocess#'''
                                    M_Sr_Com_ID = [ l[0] for l in M_Sr_ID_RecCnt_Combin[t] ]
                                    for tt in range(0, len(M_Pure_block_Com)):
                                        '''#Retain valid information --- M_Pool_Pdl_Comb_Set = [[sr_block_1, sr_block_2, sr_block_3 [], ...]'''
                                        Sub_M_Pool_Cer_Rel = Decoder_Matrix.Cer_CRC_Block(self, M_Pure_block_Com[tt][0], M_Sr_Com_ID, Mbl_Matrix)
                                        '''one of pool_pdl_com_rel = [True, SR_Block] or [False, []]'''
                                        if Sub_M_Pool_Cer_Rel[0]:
                                            return True, {}, {}, Sub_M_Pool_Cer_Rel[1]

                '''M_Rec_Cnt_Finish_T:  Judge whether or not ending decoding in the certained correction Symbols'''
                M_Rec_Cnt_Finish_T = [True if Mbl_M_stop_loc[l][sub_rec_cnt] == 'End' else False for l in Mbl_M_stop_loc ]
            
            '''#Finish the certained correction counts, and then sub_rec_cnt++#'''
            '''#According to the current decoding situation, then adjust the order of these blocks#'''
            M_Sr_Empty = []
            M_Sr_UnEmpty = []
            
            for l in Mbl_M_Nr_Infr:
                Empty_TF = True 
                M_Block_Pass_Cnt = list(Mbl_M_stop_loc[l[0]].keys())
                M_Block_Pass_Cnt.sort()
                for ll in M_Block_Pass_Cnt:
                    if Mbl_M_record[l[0]][ll] != []:
                        '''UnEmpty'''
                        Empty_TF = False
                        break
                if Empty_TF:    M_Sr_Empty.append(l)  #Empty
                else:   M_Sr_UnEmpty.append(l) #UnEmpty
            
            '''#Empty in front, not empty in back#'''
            Mbl_M_Nr_Infr = M_Sr_Empty
            Mbl_M_Nr_Infr.extend(M_Sr_UnEmpty)
            del M_Sr_Empty, M_Sr_UnEmpty
        return False, Mbl_M_stop_loc, Mbl_M_record, []


    def Soft_RecCerCnt(self, rec_cnt, rec_start_loc, rec_loc, rec_block ):
        rec_comindex_set = Decoder_Matrix.Combin_Order(self, rec_cnt, rec_loc)
        '''#To separete the all candidate into little interval as "range_loc" ,then decode with reed-solomon if crc_block true then return#'''
        for sub_loc in range(rec_start_loc, len(rec_comindex_set)):
            sub_sr = Decoder_Matrix.Soft_CerLocation(self, rec_block, rec_comindex_set[sub_loc])
            if sub_sr != []:
                if sub_loc+1 < len(rec_comindex_set):   return sub_loc+1, sub_sr
                else:   return 'End', sub_sr
        return 'End', []


    def Soft_CerLocation(self, loc_sr_block, loc_combin):
        loc_sr_refer = [  Soft_Rule(self.k, self.Block_Depth[l], loc_sr_block[l] )  for l in loc_combin ] 
        loc_sr_combin = Decoder_Matrix.Comb_SoftRule(loc_sr_refer)
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
            dpindex_soft_rel = Decoder_Matrix.Coder_Hard_Block(self, loc_sr_block_1)
            if dpindex_soft_rel[0] not in  ['UnDe', 'DeEq2Coll']:
                '''#Make sure the correction item works#'''
                softindex_effect_T = True
                for rec_index in range(len(sub_loc_combin)):
                    if dpindex_soft_rel[1][ loc_combin[rec_index] ] != sub_loc_combin[rec_index]:
                        softindex_effect_T = False
                        break
                if softindex_effect_T:
                    dpindex_rel.append(dpindex_soft_rel[1][:self.rsK])
                    
        return dpindex_rel
    

    def Combin_Order(self, com_cnt, com_loc):
        limit_loc = self.Max_Location[com_cnt]
        limit_com = self.Max_Comloc[com_cnt]
        '''Reversed order'''
        loc_order = sorted(range(len(com_loc)), key=lambda l : com_loc[l], reverse=True)
        loc_order_part = loc_order[: limit_loc]
        
        combin_loc_set = Decoder_Matrix.Combinations_Cnt(loc_order_part, com_cnt)
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
        cer_sr_pool_Infr = [[ Cer_ID[l], Cer_sr_pdl_Com[l]] for l in range(len(Cer_sr_pdl_Com))]

        cer_crc_Matrix = deepcopy(cer_Matrix)
        for sub_cer_Infr in cer_sr_pool_Infr:
            cer_crc_Matrix[ sub_cer_Infr[0] ] = sub_cer_Infr[1].split(',')
            
        cer_de_T = Decoder_Matrix.Coder_CRC32(self, cer_crc_Matrix)
        if cer_de_T:
            return [True, cer_crc_Matrix]
        else:
            return [False, []]

    def Coder_CRC32(self, CRC_Cpdna):
        CRC_GF = [ self.Cpdna2GF[','.join(l[ll:ll+self.Symbol2Cpdna])] for l in CRC_Cpdna for ll in range(0, self.Pld_dna_len, self.Symbol2Cpdna)]
        CRC_Bit = [ bin( l ).split('b')[1].zfill(self.Symbol2bit) for l in CRC_GF]
        CRC_Bit_str = ''.join(CRC_Bit)
        
        CRC_Bit_payload = CRC_Bit_str[:-32]
        CRC_Bit_utf =  CRC_Bit_payload.encode('utf-8')
        CRC_digital = crc32(CRC_Bit_utf)
        CRC_bit = '{:08b}'.format(CRC_digital).zfill(32)
        
        if CRC_bit == CRC_Bit_payload[-32:]:
            return True
        else:
            return True

    @staticmethod
    def Combinations_Cnt(candi_L, conbin_count):
        """List all combinations: choose k elements from list L"""
        n_L = len(candi_L)
        com_rel = []
        for l in range(n_L-conbin_count+1):
            if conbin_count > 1:
                newL = candi_L[l+1:]
                Comb = Decoder_Matrix.Combinations_Cnt(newL, conbin_count - 1)
                for item in Comb:
                    item.insert(0, candi_L[l])
                    com_rel.append(item)
            else:
                com_rel.append([candi_L[l]])
        return com_rel
    
    @staticmethod
    def Comb_SoftRule(sub_rule):
        '''#Initial all variances#'''
        max_y_idx = len(sub_rule)  
        row_max_idx = 1 
        arr_len, lst_row, lst_rst = [], [], []
        arr_idx = [0] * max_y_idx  
    
        '''#Transfrom Two dimensions array(sub_rule) into One dimension array(lst_row)#'''
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
        root = os.path.abspath(".")
        Pi_path = root + '/Pi2Random.txt'
        with open(Pi_path, 'r') as f_pi:
            for line in f_pi:
                pi_bit = line.strip()
        pi_bit += len(input_bit)//len(pi_bit)*pi_bit
        
        rel_bit = []
        for r in range(len(input_bit)): 
            '''0^0 = 0, 0^1 = 1, 1^0 = 1, 1^1 = 0'''
            if input_bit[r] == 0 and pi_bit[r] == '0':
                rel_bit.append(0)
            elif input_bit[r] == 0 and pi_bit[r] == '1':
                rel_bit.append(1)
            elif input_bit[r] == 1 and pi_bit[r] == '0':
                rel_bit.append(1)
            elif input_bit[r] == 1 and pi_bit[r] == '1':
                rel_bit.append(0)
            else:
                continue
        return rel_bit

    def Decode_Softrule(self, file_in, file_out):
        '''#initialize the class for soft decision#'''
        f_nr = open(file_in, 'r')
        NrCol_Infr, dp_list, decode_bit = [], [], []
        for line in f_nr:
            '''>contig.1'''
            if re.search(r'^>', line):
                nr_block_id = re.findall(r'\d+', line)[0]
                nr_block_id = int(nr_block_id)
            elif re.search(r'^CpDNA', line):
                line = line.strip().split('CpDNA:')[-1].strip()
                nr_block_cpdna = line.split(',')
            elif re.search(r'Depth', line):
                line = line.strip().split('Depth:')[-1].strip()
                dp_list = list(map(int, line.split(',')))
                dp_list_form = [ round(l, 6) for l in dp_list]
            elif re.search(r'EDist', line):
                nr_block_dist = line.strip().split('EDist:')[-1].strip().split(',')
                nr_block_dist = list(map(float, nr_block_dist))

                NrCol_Infr.append([ (int(nr_block_id)-1)%self.Matrix_len, nr_block_cpdna, nr_block_dist])
                if nr_block_id % self.Matrix_len == 0:
                    '''To decode one CRC_Matrix in soft decision and record the decoding progress'''
                    print('#begin, {} block decoding...'.format( nr_block_id // self.Matrix_len))
                    time_start = time.time()
                    #try:
                    Matrix_TF, Matrix_Deal_way, Matrix_decoded = Decoder_Matrix.Coder_Matrix(self, NrCol_Infr, dp_list_form)
                    '''
                    except:
                        Matrix_TF, Matrix_Deal_way, Matrix_decoded = False, 'Failure', '' #change'''
                    time_end = time.time()                
                    if Matrix_TF:
                        print('end, {} in {} sec'.format(Matrix_Deal_way, round(time_end - time_start, 4)))
                    else:
                        exit(1)
                        print('end, {} in {} sec'.format(Matrix_Deal_way, round(time_end - time_start, 4)))
                                        
                    
                    NrCol_Infr = []
                    decode_bit += Matrix_decoded
                else:
                    continue
        f_nr.close()

        '''remove the random with pi'''
        decode_bit = Decoder_Matrix.Random_PI(decode_bit)
        l_0 = 0
        for l in decode_bit[::-1]:
            if l == 0:
                l_0 += 1
            else:
                break
        decode_bit = decode_bit[:-l_0] #decode_bit = decode_bit.rstrip('0')
        decode_bit_count = len(decode_bit)
        
        for l in range(64, 0, -1):
            length_bit_try = decode_bit[-l:] + [0]*(64-l)
            length_bit_str = ''.join(list(map(str, length_bit_try)))
            if bin(decode_bit_count-l).split('b')[1].zfill(64) == length_bit_str:
                decode_bit_np = np.array(decode_bit[:-l])
                del decode_bit
                byte_array = packbits(decode_bit_np.reshape((decode_bit_count-l) // 8, 8), axis=1).reshape(-1)
                byte_array.tofile(file=file_out)
                break
        return 0

def read_args():
    """
    Read arguments from the command line.

    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value in 6(64), 8(128), 10(256).")
    parser.add_argument("-i", "--inferred_path", required=True, type=str,
                        help="the inferred sequences path")
    parser.add_argument("-s", "--saved_path", required=True, type=str,
                        help="the decoding seqeuences path with Derrick-cp.")
    return parser.parse_args()


if __name__ == "__main__":
    '''#the global variance#'''
    params = read_args()

    print("The parameters are:")
    print("K\t= ", params.resolution) #-k
    print("Inferred sequences path\t= ", params.inferred_path)  #-i
    print("Decoding sequences path\t= ", params.saved_path)   #-s

    Decoder_Matirx = Decoder_Matrix(params.resolution, 41, 45)
    Decoder_Matirx.Decode_Softrule(params.inferred_path, params.saved_path)
