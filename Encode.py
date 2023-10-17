# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:27:09 2022

@author: Xu Yaping
"""

from zlib import crc32
import reedsolo as rs
import os, time
from argparse import ArgumentParser
from numpy import fromfile, unpackbits, expand_dims, uint8


class Encode_Matrix():
    def __init__(self, k, rsK = 41, rsN = 45):
        """
        Initialize the coder of CRC_Matirx.
        """
        self.k = k
        self.rsK, self.rsN = rsK, rsN
        self.rsD = self.rsN - self.rsK
        self.rsT = self.rsD//2
        self.Matrix_len, self.read_size = 10, 150

        self.Cpdna_set, self.GFint, self.Symbol2Cpdna, self.Symbol2bit = Encode_Matrix.Basic_infr(self.k)
        self.GF2Cpdna, self.Cpdna2GF = Encode_Matrix.Trans_GF2Int(self.Cpdna_set, self.Symbol2Cpdna, self.Symbol2bit)

        self.Block_dna_size, self.Pld_dna_size, self.Pld_bit_size = self.rsN * self.Symbol2Cpdna, self.rsK * self.Symbol2Cpdna, self.rsK * self.Symbol2bit
        self.Matrix_bit_size = self.Pld_bit_size * self.Matrix_len - 32
        self.MultiMatrix_bit_size = self.Matrix_bit_size * (self.read_size//self.Matrix_len)
        
        '''Py3'''
        self.prim = rs.find_prime_polys(generator = self.GFint[0], c_exp=self.GFint[1], fast_primes=True, single=True)
        self.table_rel = rs.init_tables(generator = self.GFint[0], c_exp=self.GFint[1], prim=self.prim)
        self.generator = rs.rs_generator_poly_all(self.rsN)
        '''mes_ecc = rs.rs_encode_msg(mes_en, self.rsD, gen=self.generator[self.rsD])'''

        '''Py2
        self.coder = rs.RSCoder(GFint=self.GFint, k=self.rsK, n=self.rsN)
        '''
        
    def Encode_Matirx(self, Matirx_Bit, need_logs=True):
        '''
        #Py2:
            CRC_digital:  1907163293
            CRC_bit:  01110001101011010000000010011101
        #Py3:
            block2crc = block_crc.encode('utf-8')
            CRC_digital = crc32(block2crc)
            CRC_bit = '{:08b}'.format(CRC_digital).zfill(32)
        '''
        '''Firstly, encode the Matirx_Oligo with CRC32, then the crc matrix was supplement into ( self.Pld_bit_size * self.Matrix_len )'''
        Matirx_Bit_Bin = Matirx_Bit.encode('utf-8')
        CRC_digital = crc32(Matirx_Bit_Bin)
        CRC_bit = '{:08b}'.format(CRC_digital).zfill(32)
        
        Matirx_Bit += CRC_bit
        Matrix_CpDNA = []
        for l in range(0, self.Matrix_bit_size, self.Pld_bit_size):
            pld_Bit = Matirx_Bit[l : l+self.Pld_bit_size]
            pld_GFint = [ int(pld_Bit[l:l+self.Symbol2bit],2) for l in range(0, self.Pld_bit_size, self.Symbol2bit) ]
            '''Py2
            Block_GFint = self.coder.encode(pld_GFint)
            '''
            Block_GFint = rs.rs_encode_msg(pld_GFint, self.rsD, gen=self.generator[self.rsD])
            Block_CpDNA = []
            for l in Block_GFint:
                Block_CpDNA.extend(self.GF2Cpdna[l][1].split(','))
            Matrix_CpDNA.append(Block_CpDNA)
        return Matrix_CpDNA

    def Bit_to_dna(self, file_in, file_encode, need_logs=True):
        bits = Encode_Matrix.Infr_to_bits(file_in)
        bits += bin(len(bits)).split('b')[1].zfill(64)
        
        div_infr = divmod(len(bits), self.MultiMatrix_bit_size)
        self.supplement = self.MultiMatrix_bit_size - div_infr[1]
        bits += '0'*self.supplement
        
        bit_random = self.Random_PI(bits)
    
        to_row, CRC_Block = 0, 0
        f_en = open(file_encode, 'w')
        for i in range(0, self.MultiMatrix_bit_size*(div_infr[0]+1), self.MultiMatrix_bit_size):
            MultiMatrix = bit_random[i: i+self.MultiMatrix_bit_size]
            '''For One CRC Matirx'''
            Encode_MultiMatrix_CpDNA = []
            for j in range(0, self.MultiMatrix_bit_size, self.Matrix_bit_size):
                CRC_Block += 1
                Matirx_Bit = MultiMatrix[j : j+self.Matrix_bit_size]
                '''encode the Matrix with reed-solomon coder and CRC32'''
                Encode_Matrix_CpDNA = self.Encode_Matirx(Matirx_Bit)
                Encode_MultiMatrix_CpDNA.extend(Encode_Matrix_CpDNA)
                
            '''To transforma the column into row'''
            for l in range(self.Block_dna_size):
                read_cpdna = [Encode_MultiMatrix_CpDNA[ll][l] for ll in range(self.read_size)]
                to_row += 1
                f_en.write('>contig{}\n'.format(to_row))
                f_en.write('{}\n'.format(','.join(read_cpdna)))
            #f_en.write('{}'.format('') + '\n')
        f_en.close()
        return CRC_Block
    
    
    '''The basic information used for encode and decode'''
    @staticmethod
    def Basic_infr(alp):
        '''#The composite dna set filtered by their accuracy and used to the digital data storage (DDS), and the sizes of Cpdna set are convenient to converse between binary information and composite DNA letters#'''
        Cpdna_set = {
            10: {0: '1:4:4:1', 1: '1:8:0:1', 2: '0:8:2:0', 3: '1:1:2:6', 4: '2:4:4:0', 5: '5:1:3:1', 6: '5:0:5:0', 7: '6:2:1:1', 8: '7:0:2:1', 9: '3:0:1:6', 10: '2:5:0:3', 11: '0:4:4:2', 12: '1:1:7:1', 13: '3:5:1:1', 14: '5:3:2:0', 15: '0:0:4:6', 16: '6:1:1:2', 17: '2:2:0:6', 18: '0:3:5:2', 19: '0:0:10:0', 20: '8:0:1:1', 21: '0:1:0:9', 22: '1:1:1:7', 23: '1:4:5:0', 24: '2:1:1:6', 25: '0:9:1:0', 26: '6:3:1:0', 27: '9:1:0:0', 28: '2:4:3:1', 29: '0:2:6:2', 30: '2:0:5:3', 31: '0:5:5:0', 32: '1:5:3:1', 33: '5:1:0:4', 34: '1:5:1:3', 35: '5:3:0:2', 36: '0:0:1:9', 37: '1:0:5:4', 38: '0:3:6:1', 39: '1:1:8:0', 40: '4:4:1:1', 41: '0:3:1:6', 42: '4:0:0:6', 43: '0:2:3:5', 44: '4:5:1:0', 45: '2:6:2:0', 46: '1:6:0:3', 47: '2:3:5:0', 48: '0:1:4:5', 49: '0:10:0:0', 50: '5:0:4:1', 51: '5:0:2:3', 52: '2:2:5:1', 53: '5:0:0:5', 54: '4:1:0:5', 55: '0:1:3:6', 56: '1:4:3:2', 57: '2:1:6:1', 58: '3:5:0:2', 59: '3:1:1:5', 60: '0:6:3:1', 61: '4:2:4:0', 62: '2:1:5:2', 63: '1:0:4:5', 64: '2:5:2:1', 65: '2:3:4:1', 66: '1:6:3:0', 67: '8:0:2:0', 68: '2:4:1:3', 69: '4:1:4:1', 70: '0:4:1:5', 71: '2:0:0:8', 72: '4:0:6:0', 73: '6:0:1:3', 74: '5:0:1:4', 75: '0:0:9:1', 76: '4:0:1:5', 77: '0:9:0:1', 78: '6:1:3:0', 79: '0:5:4:1', 80: '1:0:6:3', 81: '3:1:4:2', 82: '0:2:0:8', 83: '1:3:1:5', 84: '1:1:3:5', 85: '2:5:3:0', 86: '3:1:6:0', 87: '3:0:2:5', 88: '6:0:2:2', 89: '0:7:3:0', 90: '0:4:2:4', 91: '0:6:4:0', 92: '0:3:2:5', 93: '2:4:0:4', 94: '7:0:1:2', 95: '1:4:1:4', 96: '1:3:0:6', 97: '7:2:0:1', 98: '0:4:0:6', 99: '0:7:2:1', 100: '5:4:1:0', 101: '1:1:5:3', 102: '0:8:1:1', 103: '7:0:3:0', 104: '7:1:2:0', 105: '5:3:1:1', 106: '1:4:2:3', 107: '4:0:4:2', 108: '2:6:0:2', 109: '10:0:0:0', 110: '0:2:1:7', 111: '1:4:0:5', 112: '0:0:2:8', 113: '3:7:0:0', 114: '2:2:1:5', 115: '0:7:0:3', 116: '0:2:2:6', 117: '1:0:3:6', 118: '1:0:0:9', 119: '0:1:6:3', 120: '5:5:0:0', 121: '7:0:0:3', 122: '4:1:3:2', 123: '3:1:2:4', 124: '4:0:5:1', 125: '2:5:1:2', 126: '2:1:0:7', 127: '0:3:0:7', 128: '7:1:1:1', 129: '2:7:0:1', 130: '0:2:7:1', 131: '3:6:0:1', 132: '5:1:4:0', 133: '1:2:5:2', 134: '8:0:0:2', 135: '0:4:6:0', 136: '2:0:8:0', 137: '3:2:1:4', 138: '8:1:1:0', 139: '6:0:4:0', 140: '5:0:3:2', 141: '4:5:0:1', 142: '6:0:3:1', 143: '0:1:1:8', 144: '3:2:5:0', 145: '0:1:8:1', 146: '6:1:0:3', 147: '6:3:0:1', 148: '2:1:4:3', 149: '0:6:1:3', 150: '2:0:7:1', 151: '6:1:2:1', 152: '2:8:0:0', 153: '5:1:1:3', 154: '2:3:1:4', 155: '4:1:2:3', 156: '1:2:6:1', 157: '3:2:0:5', 158: '9:0:0:1', 159: '1:6:2:1', 160: '2:6:1:1', 161: '1:9:0:0', 162: '3:0:6:1', 163: '9:0:1:0', 164: '5:2:1:2', 165: '1:0:8:1', 166: '3:2:4:1', 167: '0:5:1:4', 168: '0:0:7:3', 169: '2:1:7:0', 170: '1:5:2:2', 171: '0:5:3:2', 172: '6:2:2:0', 173: '0:8:0:2', 174: '0:6:2:2', 175: '0:5:0:5', 176: '3:0:7:0', 177: '3:6:1:0', 178: '4:6:0:0', 179: '0:3:7:0', 180: '0:6:0:4', 181: '2:0:3:5', 182: '0:5:2:3', 183: '1:3:4:2', 184: '2:0:4:4', 185: '2:0:2:6', 186: '2:0:1:7', 187: '4:2:0:4', 188: '7:2:1:0', 189: '1:2:7:0', 190: '1:0:9:0', 191: '1:2:3:4', 192: '0:0:5:5', 193: '1:3:2:4', 194: '1:5:0:4', 195: '1:2:0:7', 196: '1:2:4:3', 197: '6:2:0:2', 198: '2:3:0:5', 199: '3:5:2:0', 200: '8:1:0:1', 201: '7:3:0:0', 202: '1:1:4:4', 203: '0:0:3:7', 204: '1:8:1:0', 205: '1:1:0:8', 206: '0:1:9:0', 207: '1:3:6:0', 208: '1:0:1:8', 209: '3:0:5:2', 210: '1:3:5:1', 211: '8:2:0:0', 212: '3:4:2:1', 213: '0:2:5:3', 214: '1:6:1:2', 215: '2:1:2:5', 216: '7:1:0:2', 217: '0:2:8:0', 218: '0:1:5:4', 219: '4:4:2:0', 220: '5:2:0:3', 221: '1:7:0:2', 222: '1:0:2:7', 223: '3:1:0:6', 224: '0:7:1:2', 225: '0:0:8:2', 226: '3:0:0:7', 227: '1:2:2:5', 228: '0:4:5:1', 229: '1:5:4:0', 230: '5:2:2:1', 231: '3:4:1:2', 232: '3:1:5:1', 233: '1:1:6:2', 234: '0:0:0:10', 235: '6:4:0:0', 236: '2:1:3:4', 237: '2:0:6:2', 238: '1:7:1:1', 239: '0:2:4:4', 240: '5:1:2:2', 241: '0:0:6:4', 242: '4:1:1:4', 243: '2:2:6:0', 244: '1:7:2:0', 245: '4:4:0:2', 246: '4:1:5:0', 247: '2:7:1:0', 248: '0:1:2:7', 249: '1:0:7:2', 250: '1:2:1:6', 251: '6:0:0:4', 252: '4:0:2:4', 253: '5:4:0:1', 254: '0:1:7:2', 255: '5:2:3:0'},
            8: {0: '0:4:4:0', 1: '0:7:1:0', 2: '2:1:5:0', 3: '6:0:1:1', 4: '3:1:3:1', 5: '5:1:1:1', 6: '3:0:1:4', 7: '3:4:0:1', 8: '0:5:2:1', 9: '0:1:5:2', 10: '5:2:0:1', 11: '1:1:6:0', 12: '6:0:0:2', 13: '0:0:2:6', 14: '6:2:0:0', 15: '0:3:1:4', 16: '4:1:0:3', 17: '0:3:5:0', 18: '0:0:3:5', 19: '1:0:5:2', 20: '0:3:4:1', 21: '6:0:2:0', 22: '0:0:7:1', 23: '2:0:0:6', 24: '0:4:0:4', 25: '4:2:1:1', 26: '1:0:7:0', 27: '1:1:0:6', 28: '0:1:6:1', 29: '3:4:1:0', 30: '1:3:4:0', 31: '7:0:0:1', 32: '1:1:5:1', 33: '3:0:0:5', 34: '5:0:3:0', 35: '5:0:2:1', 36: '0:0:0:8', 37: '1:2:5:0', 38: '0:1:4:3', 39: '0:2:0:6', 40: '3:1:1:3', 41: '4:3:1:0', 42: '0:0:6:2', 43: '4:0:4:0', 44: '4:0:1:3', 45: '0:5:1:2', 46: '1:2:1:4', 47: '1:6:1:0', 48: '2:4:1:1', 49: '1:3:3:1', 50: '0:5:3:0', 51: '6:1:1:0', 52: '1:1:3:3', 53: '1:0:2:5', 54: '0:6:1:1', 55: '1:4:1:2', 56: '0:1:3:4', 57: '0:1:2:5', 58: '1:4:0:3', 59: '2:6:0:0', 60: '1:0:1:6', 61: '0:6:2:0', 62: '5:3:0:0', 63: '4:1:1:2', 64: '1:0:3:4', 65: '4:1:2:1', 66: '1:5:2:0', 67: '5:1:0:2', 68: '3:3:1:1', 69: '0:4:1:3', 70: '2:1:4:1', 71: '4:0:0:4', 72: '0:1:1:6', 73: '2:0:1:5', 74: '3:5:0:0', 75: '3:1:0:4', 76: '0:1:0:7', 77: '1:1:4:2', 78: '8:0:0:0', 79: '0:0:4:4', 80: '1:3:0:4', 81: '0:6:0:2', 82: '1:6:0:1', 83: '1:2:4:1', 84: '1:5:1:1', 85: '1:0:4:3', 86: '4:4:0:0', 87: '7:1:0:0', 88: '0:0:5:3', 89: '1:4:3:0', 90: '4:0:3:1', 91: '0:0:1:7', 92: '0:3:0:5', 93: '1:3:1:3', 94: '4:1:3:0', 95: '2:1:1:4', 96: '0:2:6:0', 97: '5:0:0:3', 98: '1:2:0:5', 99: '4:3:0:1', 100: '1:5:0:2', 101: '5:1:2:0', 102: '3:0:4:1', 103: '1:7:0:0', 104: '5:2:1:0', 105: '2:0:6:0', 106: '1:4:2:1', 107: '3:0:5:0', 108: '0:4:3:1', 109: '0:0:8:0', 110: '7:0:1:0', 111: '6:1:0:1', 112: '3:1:4:0', 113: '0:5:0:3', 114: '0:1:7:0', 115: '2:5:1:0', 116: '1:1:2:4', 117: '5:0:1:2', 118: '0:7:0:1', 119: '1:0:0:7', 120: '0:2:1:5', 121: '2:1:0:5', 122: '1:1:1:5', 123: '0:8:0:0', 124: '1:0:6:1', 125: '2:0:5:1', 126: '0:2:5:1', 127: '2:5:0:1'},
            6: {0: '2:4:0:0', 1: '1:1:0:4', 2: '6:0:0:0', 3: '4:0:2:0', 4: '1:5:0:0', 5: '1:1:1:3', 6: '5:0:0:1', 7: '0:3:0:3', 8: '3:3:0:0', 9: '0:0:3:3', 10: '1:0:3:2', 11: '0:2:4:0', 12: '0:0:6:0', 13: '0:2:0:4', 14: '0:0:1:5', 15: '0:0:5:1', 16: '0:0:2:4', 17: '0:6:0:0', 18: '0:4:2:0', 19: '0:1:0:5', 20: '1:1:3:1', 21: '1:4:0:1', 22: '0:1:1:4', 23: '4:1:0:1', 24: '1:3:2:0', 25: '1:0:5:0', 26: '5:1:0:0', 27: '3:0:0:3', 28: '0:2:3:1', 29: '1:2:0:3', 30: '3:0:3:0', 31: '2:0:0:4', 32: '1:0:4:1', 33: '0:2:1:3', 34: '0:0:0:6', 35: '1:4:1:0', 36: '0:5:0:1', 37: '4:0:0:2', 38: '0:4:0:2', 39: '1:3:0:2', 40: '4:2:0:0', 41: '0:1:3:2', 42: '2:0:3:1', 43: '4:1:1:0', 44: '0:1:5:0', 45: '1:3:1:1', 46: '0:4:1:1', 47: '0:3:2:1', 48: '1:0:1:4', 49: '0:0:4:2', 50: '0:1:2:3', 51: '1:1:4:0', 52: '1:0:2:3', 53: '4:0:1:1', 54: '0:3:3:0', 55: '2:0:4:0', 56: '0:3:1:2', 57: '1:2:3:0', 58: '2:0:1:3', 59: '0:5:1:0', 60: '5:0:1:0', 61: '1:0:0:5', 62: '3:1:1:1', 63: '0:1:4:1'},
        }
        
        '''the GFint for differernt resolution k !!!'''
        GFint = { 6: [2, 6], 8: [2, 7], 10: [2, 8] }
        Symbol2bit = { 6: 6, 8: 7, 10: 8 }
        Symbol2Cpdna = { 6: 1, 8: 1, 10: 1 }
        return Cpdna_set[alp], GFint[alp], Symbol2Cpdna[alp], Symbol2bit[alp]

    @staticmethod
    def Trans_GF2Int(Cp_set, Sym2Cp, Sym2bit):
        GF2Cpdna, GF_int = {}, -1
        if Sym2Cp == 4:
            '''#Symbol is equal to three Cpdna#'''
            for l1 in Cp_set:
                for l2 in Cp_set:
                    for l3 in Cp_set:
                        for l4 in Cp_set:
                            GF_int += 1
                            GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(Sym2bit) ,','.join([Cp_set[l1],Cp_set[l2],Cp_set[l3]])]
        elif Sym2Cp == 3:
            '''#Symbol is equal to three Cpdna#'''
            for l1 in Cp_set:
                for l2 in Cp_set:
                    for l3 in Cp_set:
                        GF_int += 1
                        GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(Sym2bit) ,','.join([Cp_set[l1],Cp_set[l2],Cp_set[l3]])]
        elif Sym2Cp == 2:
            '''#Symbol is equal to three Cpdna#'''
            for l1 in Cp_set:
                for l2 in Cp_set:
                    GF_int += 1
                    GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(Sym2bit) ,','.join([Cp_set[l1],Cp_set[l2]])]
        elif Sym2Cp == 1:
            '''#Symbol is equal to three Cpdna#'''
            for l1 in Cp_set:
                GF_int += 1
                GF2Cpdna[GF_int] = [bin(GF_int).split('b')[1].zfill(Sym2bit) ,','.join([Cp_set[l1]])]
        else:
            print('ERROR in forming GF2Cpdna.')
            exit(1)
        Cpdna2GF = { ll[1] : l for l,ll in GF2Cpdna.items() }
        return GF2Cpdna, Cpdna2GF

    @staticmethod
    def Infr_to_bits(file_bit_path):
        '''binary file'''
        bit2stream = unpackbits(expand_dims(fromfile(file=file_bit_path, dtype=uint8), 1), axis=1).reshape(-1)
        bit2stream = bit2stream.tolist()
        return ''.join(list(map(str,bit2stream)))
    
    @staticmethod
    def Random_PI(input_bit):
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
                        help="the resolution value in 6(64), 8(128), 10(256).")
    parser.add_argument("-i", "--digital_path", required=True, type=str,
                        help="the inferred sequences path")
    parser.add_argument("-s", "--saved_path", required=True, type=str,
                        help="the encoding seqeuences path with Derrick-cp.")
    return parser.parse_args()


if __name__ == "__main__":
    params = read_args()
    print("The parameters are:")
    print("k = ", params.resolution) #-k
    print("Digital file path = ", params.digital_path)  #-i
    print("Encoding sequences path = ", params.saved_path)   #-s

    
    time_start = time.time()
    print('begin, encoding...')
    Encoder_Matirx = Encode_Matrix(params.resolution)
    CRC_Block_cnt = Encoder_Matirx.Bit_to_dna( params.digital_path, params.saved_path, need_logs=True )
    time_end = time.time()
    time_encode = time_end - time_start
    print('end, {} blocks {} sec.'.format(CRC_Block_cnt, round(time_encode, 3)))
