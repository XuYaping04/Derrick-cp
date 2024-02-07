#-*- coding : utf-8-*-
"""
Created on Wed Jun 22 14:27:09 2022

@author: Xu Yaping
"""

import os, time, itertools
import reedsolo as rs
from zlib import crc32
from argparse import ArgumentParser
from numpy import fromfile, unpackbits, expand_dims, uint8

class Digital2Seq_Encode():
    def __init__(self, k: int, alp_size: int, rsK: int = 41, rsN: int = 45):
        '''# TODO 0-0: Initialize the coder of Matirx consisting of ten consecutive blocks and CRC32.'''
        self.k, self.alp_size, self.matrix_size, self.read_size , self.crc= k, alp_size, 10, 150, 32
        self.rsK, self.rsN = rsK, rsN
        self.rsD = self.rsN - self.rsK
        self.rsT = self.rsD//2

        '''# TODO 0-1:  Two conversions --- Bits and GFint, Bits and letters'''
        self.Cpdna_set, self.Symbol2bit, self.GFint, self.CpDNA2bit = Digital2Seq_Encode.Basic_infr(self.k, self.alp_size)
        self.Bit2CpDNA_set = Digital2Seq_Encode.Trans_CpDNA2Bit(list(self.Cpdna_set.values()), self.CpDNA2bit[0])
        
        '''# TODO 0-2:  Simualte the padding number of bits, to comlish the mapping between bits and letters.'''
        padding_val = divmod(self.rsN * self.Symbol2bit[1], self.CpDNA2bit[1])
        self.padding_bits = 0
        if padding_val[1] != 0:
            self.Block_cpdna_size = (padding_val[0] + 1)*self.CpDNA2bit[0]
            self.padding_bits = self.CpDNA2bit[1] - padding_val[1]
        del padding_val
        
        #Enconding: The size of bits in one whole matrix
        self.Pld_bit_size = self.rsK * self.Symbol2bit[1]
        self.Matrix_bit_size = self.Pld_bit_size * self.matrix_size - self.crc
        assert self.read_size % self.matrix_size == 0
        self.MultiMatrix_bit_size = self.Matrix_bit_size * (self.read_size//self.matrix_size)
        
        '''
        # TODO 0-3:  The basic informatio of Reed-solomon code
        mes_ecc = rs.rs_encode_msg(mes_en, self.rsD, gen=self.generator[self.rsD])
        '''
        self.prim = rs.find_prime_polys(generator = self.GFint[0], c_exp=self.GFint[1], fast_primes=True, single=True)
        self.table_rel = rs.init_tables(generator = self.GFint[0], c_exp=self.GFint[1], prim=self.prim)
        self.generator = rs.rs_generator_poly_all(self.rsN)
        

    def Digital2Seq(self, file_in: str, file_encode: str, file_encode_col: str, need_logs: bool = True):
        '''
        # TODO 1-0: Read the binary information and padding '0' to form the complete multi-matrix. In addition, random with Pi.
        '''
        bits = Digital2Seq_Encode.Digital2Bits(file_in)
        bits += bin(len(bits)).split('b')[1].zfill(64)
        div_infr = divmod(len(bits), self.MultiMatrix_bit_size)
        bits += '0'*(self.MultiMatrix_bit_size - div_infr[1])
        bit_random = self.Random_PI(bits)
        
        row_no, Col_no, CRC_no = 0, 0, 0
        f_en = open(file_encode, 'w')
        f_en_col = open(file_encode_col, 'w')
        for i in range(0, self.MultiMatrix_bit_size*(div_infr[0]+1), self.MultiMatrix_bit_size):
            MultiMatrix = bit_random[i: i+self.MultiMatrix_bit_size]
            
            MultiMatrix_CpDNA = []
            for j in range(0, self.MultiMatrix_bit_size, self.Matrix_bit_size):
                CRC_no += 1
                Matirx_Bit = MultiMatrix[j : j+self.Matrix_bit_size]
                
                '''# TODO 1-1: Encode the Matrix with reed-solomon encoder and CRC32.'''
                CRCMatrix_CpDNA = self.Encode_CRCMatirx(Matirx_Bit)
                MultiMatrix_CpDNA.extend(CRCMatrix_CpDNA)
                
            '''# TODO 1-2: Convert the blocks in column into the read with letters in row.'''
            for l in range(len(MultiMatrix_CpDNA[0])):
                row_no += 1
                read_cpdna = [MultiMatrix_CpDNA[ll][l] for ll in range(len(MultiMatrix_CpDNA))]
                f_en.write('>contig{}\n'.format(row_no))
                f_en.write('{}\n'.format(','.join(read_cpdna)))
                

            for l in MultiMatrix_CpDNA:
                Col_no += 1
                f_en_col.write('>contig{}\n'.format(Col_no))
                f_en_col.write('{}\n'.format(','.join(l)))
        f_en.close()
        f_en_col.close()
        return CRC_no


    def Encode_CRCMatirx(self, Matirx_Bit, need_logs=True):
        '''
        #Py2:
            CRC_digital:  1907163293
            CRC_bit:  01110001101011010000000010011101
        #Py3:
            block2crc = block_crc.encode('utf-8')
            CRC_digital = crc32(block2crc)
            CRC_bit = '{:08b}'.format(CRC_digital).zfill(32)
        '''
        '''# TODO 2-0: Encode the matrix with reed-solomon encoder and CRC32, which consists of ten consecutive blocks '''
        Matirx_Bit_Bin = Matirx_Bit.encode('utf-8')
        CRC_digital = crc32(Matirx_Bit_Bin)
        CRC_bit = '{:08b}'.format(CRC_digital).zfill(self.crc)
        Matirx_Bit += CRC_bit
        
        Matrix_CpDNA = []
        for l in range(0, self.Matrix_bit_size+self.crc, self.Pld_bit_size):
            pld_Bit = Matirx_Bit[l : l+self.Pld_bit_size]
            
            '''# TODO 2-1: Bit2GFint, meaning mapping the bits into GFint.'''
            pld_GFint = [ int(pld_Bit[l:l+self.Symbol2bit[1]],2) for l in range(0, self.Pld_bit_size, self.Symbol2bit[1]) ]
            Block_GFint = rs.rs_encode_msg(pld_GFint, self.rsD, gen=self.generator[self.rsD]) #Py2: Block_GFint = self.coder.encode(pld_GFint)
            
            '''# TODO 2-2: GFint2Bit, meaning mapping the GFint into bits;  and padding bits'''
            Block_bits = ''.join([ bin(l).split('b')[-1].zfill(self.Symbol2bit[1]) for l in Block_GFint ]) + '0'*self.padding_bits
            
            '''# TODO 2-3: Bits2CpDNA, meaning mapping the Bits2CpDNA into letters'''
            Block_CpDNA = []
            for l in range(0, len(Block_bits), self.CpDNA2bit[1]):
                CpDNA_str = Block_bits[l:l+self.CpDNA2bit[1]].lstrip('0')
                CpDNA_val = CpDNA_str if CpDNA_str != '' else '0'
                Block_CpDNA.extend(self.Bit2CpDNA_set[ CpDNA_val ])
            Matrix_CpDNA.append(Block_CpDNA)
            
        return Matrix_CpDNA

    @staticmethod
    def Binary_add(a: str, b:str):
        '''# TODO: Binary addition.'''
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

    '''
    @staticmethod
    def Trans_CpDNA2Bit(Cp_set: list, conbin_count: int):    
        Bit2CpDNA, fomer_bit = {}, -1
        if conbin_count == 3:
            #Symbol is equal to three Cpdna#
            for l1 in Cp_set:
                for l2 in Cp_set:
                    for l3 in Cp_set:
                        if fomer_bit == -1: 
                            bit_val = '0'
                        else:                        
                            bit_val = Digital2Seq_Encode.Binary_add(fomer_bit, '1')
                        fomer_bit = bit_val
                        Bit2CpDNA[bit_val] = [l1, l2, l3]
        elif conbin_count == 2:
            #Symbol is equal to three Cpdna#
            for l1 in Cp_set:
                for l2 in Cp_set:
                    if fomer_bit == -1: 
                        bit_val = '0'
                    else:                        
                        bit_val = Digital2Seq_Encode.Binary_add(fomer_bit, '1')
                    fomer_bit = bit_val
                    Bit2CpDNA[bit_val] = [l1, l2]
        elif conbin_count == 1:
            #Symbol is equal to three Cpdna#
            for l1 in Cp_set:
                if fomer_bit == -1:
                   bit_val = '0'
                else:
                   bit_val = Digital2Seq_Encode.Binary_add(fomer_bit, '1') 
                fomer_bit = bit_val
                Bit2CpDNA[bit_val] = [l1]
        else:
            print('ERROR in forming GF2Cpdna.')
            exit(1)
        return Bit2CpDNA
    
    
    @staticmethod
    def Combination_Count(candi_L: list, conbin_count: int):    
        """List all combinations: choose k elements from list L"""
        n_L = len(candi_L)
        com_rel = []
        for l in range(n_L-conbin_count+1):
            if conbin_count > 1:
                newL = candi_L[l+1:]
                Comb = Digital2Seq_Encode.Combination_Count(newL, conbin_count - 1)
                for item in Comb:
                    item.insert(0, candi_L[l])
                    com_rel.append(item)
            else:
                com_rel.append([candi_L[l]])
        return com_rel
    '''
        
    @staticmethod
    def Trans_CpDNA2Bit(Cp_set: list, CpDNA2bit: list):
        '''#TODO: The mapping rule between letters and bits
            #CpDNA2bit[0] CpDNA is equal to CpDNA2bit[1] bits, e.g.:19bits is equal to 3 letters in Sup.Fig.5
            #CpDNA_combin = Digital2Seq_Encode.Combination_Count(Cp_set, CpDNA2bit[0])
        '''
        CpDNA_combin = []
        for l in itertools.combinations_with_replacement(Cp_set, CpDNA2bit):
            ll_list = [ll for ll in itertools.permutations(l)]
            CpDNA_combin.extend(list(set(ll_list)))
        CpDNA_combin.sort()
        
        CpDNA_count = pow(len(Cp_set), CpDNA2bit)        
        Bit2CpDNA, fomer_bit = {}, -1
        #CpDNA2Bit = {}
        for l in range(CpDNA_count):
            if fomer_bit == -1:
                bit_val = '0'
            else:
                bit_val = Digital2Seq_Encode.Binary_add(fomer_bit, '1')
            fomer_bit = bit_val
            Bit2CpDNA[bit_val] = CpDNA_combin[l]
            #CpDNA2Bit[','.join(CpDNA_combin[l])] = bit_val
        return Bit2CpDNA


    @staticmethod
    def Basic_infr(k: int, alp: int):
        '''#TODO: The basic information used for encode and decode.
            #The composite dna set filtered by their accuracy and used to the digital data storage (DDS), and the sizes of Cpdna set are convenient to converse between binary information and composite DNA letters#
            depth = 50 in k = 6,8,10
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
    def Digital2Bits(file_bit_path: str):
        '''#TODO: Read binary information from the digital file'''
        bit2stream = unpackbits(expand_dims(fromfile(file=file_bit_path, dtype=uint8), 1), axis=1).reshape(-1)
        bit2stream = bit2stream.tolist()
        return ''.join(list(map(str,bit2stream)))

    @staticmethod
    def Random_PI(input_bit: str):
        '''#TODO: Random bits with Pi'''
        root = os.path.abspath(".")
        Pi_path = root + '/Pi2Random.txt'
        with open(Pi_path, 'r') as f_pi:
            for line in f_pi:
                pi_bit = line.strip()
        pi_bit += len(input_bit)//len(pi_bit)*pi_bit
        
        rel_bit = ''
        for r in range(len(input_bit)): 
            '''0^0 = 0, 0^1 = 1, 1^0 = 1, 1^1 = 0'''
            if input_bit[r] == '0' and pi_bit[r] == '0':
                rel_bit += '0'
            elif input_bit[r] == '0' and pi_bit[r] == '1':
                rel_bit += '1'
            elif input_bit[r] == '1' and pi_bit[r] == '0':
                rel_bit += '1'
            elif input_bit[r] == '1' and pi_bit[r] == '1':
                rel_bit += '0'
            else:
                continue
        return rel_bit    
    

def read_args():
    """
    Read arguments from the command line.

    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value in 6(64), 6(84), 8(128), 10(256) et al.")
    parser.add_argument("-c", "--alphabet_size", required=True, type=int,
                        help="the size of alphabet count in k(=6,8,10 et al.)")
    parser.add_argument("-i", "--digital_path", required=True, type=str,
                        help="the inferred sequences path")
    parser.add_argument("-o", "--encoded_path", required=True, type=str,
                        help="the encoding seqeuences path with Derrick-cp.")
    parser.add_argument("-p", "--encoded_col_path", required=True, type=str,
                        help="the encoding seqeuences path with Derrick-cp.")

    #Optional
    parser.add_argument("-rsK", "--rsK", required=False, type=int,
                        help="RS(rsK, rsN)")
    parser.add_argument("-rsN", "--rsN", required=False, type=int,
                        help="RS(rsK, rsN)")
    return parser.parse_args()

#------------------------------------------------------------------------------
if __name__ == "__main__":
    '''#The global variance#'''
    params = read_args()
    print("The parameters are:")
    print("Resolution(k) = {}".format(params.resolution)) #-k
    print("Alphabet size = {}".format(params.alphabet_size)) #-c
    print("Digital file path = ", params.digital_path)  #-i
    print("Encoding sequences path = ", params.encoded_path) #-o
    print("Encoding sequences path = ", params.encoded_col_path) #-p
    print()
    
    time_start = time.time()
    print('Begin, encoding...')
    Encoder = Digital2Seq_Encode(params.resolution, params.alphabet_size)
    CRC_matrix_no_cnt = Encoder.Digital2Seq( params.digital_path, params.encoded_path, params.encoded_col_path, need_logs=True )
    time_end = time.time()
    time_encode = time_end - time_start
    print('End, {} blocks {} sec.'.format(CRC_matrix_no_cnt, round(time_encode, 3)))
        
'''
/public/agis/ruanjue_group/xuyaping/3-DNACoding/3-Test/1-SourceToOligo/0-En.Shell/0-Encode.NoCRC.FullSet.py 
python3 -k resolution -c alphabet_size -rsK rsK -rsN rsN -i digital_path -o encoded_path
file_in = '/public/agis/ruanjue_group/xuyaping/3-DNACoding/3-Test/1-SourceToOligo/3-CharToBp//2-Bit.out'
file_col = '/public/agis/ruanjue_group/xuyaping/3-DNACoding/3-Test/1-SourceToOligo/4-BinToCpDNA/6-K6/2-RS45.41.8Bits.Full/3-Bit.col.sim.out'
'''
