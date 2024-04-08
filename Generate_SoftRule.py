# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:55:39 2022

@author: XuYaping
"""

import os, math, re, shutil
import numpy as np
import copy as cp
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

def Mkdir(path):
    #TODO
    folder = os.path.exists(path)
    if not folder:  os.makedirs(path)	

def partition_into_four_integer(n):
    '''#TODO: partition the specified integer into four nonnegative integer'''
    def backtrack(start, target, path, rel):
        if len(path) == 4 and target == 0:
            rel.append(path[:])
            return
        if len(path) == 4 or target == 0:
            return
        for i in range(start, target + 1):
            path.append(i)
            backtrack(i, target - i, path, rel)
            path.pop()

    rel = []
    backtrack(0, n, [], rel)
    return rel

def permute_unique(nums):
    '''#TODO: full permutation'''
    if len(nums) == 0:
        return [[]]
    
    unique_permutations = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        remaining_nums = nums[:i] + nums[i+1:]
        sub_permutations = permute_unique(remaining_nums)
        for sub_permutation in sub_permutations:
            unique_permutations.append([nums[i]] + sub_permutation)
    
    return unique_permutations

def partition_permute(n):
    '''#TODO'''
    result = []
    result_list = partition_into_four_integer(n)
    for l in result_list:
        result.extend(permute_unique(sorted(l)))
    return result

def Factorial(sub_sample,sub_fmol):
    '''#TODO: C(sub_sample,sub_fmol) '''
    sum_former, sum_latter = 0, 0
    if sub_sample <= sub_fmol/2:
        sub_sample = sub_fmol - sub_sample
        
    for f in range(sub_sample+1,sub_fmol+1):
        sum_former += math.log10(f)
    for f in range(1,sub_fmol-sub_sample+1):
        sum_latter += math.log10(f)
    return sum_former - sum_latter

def Sample_Frequency(ratio_k, ratio_fmol, eft, uneft):
    '''#TODO: ratio_fmol -- sample_dp'''
    if uneft == []: #Flag
        comsum = [Factorial(eft[s],ratio_fmol[s]) for s in range(4)]
        comsum.append(deep*math.log10(0.99))
        comsum.sort()
        return sum(comsum)
    
    else:
        uneftsum = sum(uneft) 
        for s in range(4):
            #accordins current based ratiok , share uneffect to the effect site
            eft[s] += round((ratio_k[s]/k)*uneftsum) 
        
        #data pruning
        if sum(eft) != deep :
            maxvalue = max(eft)
            maxcount = eft.count(maxvalue) 
            if maxcount == 1 :  
                if sum(eft) == deep + 1:
                    eft[eft.index(max(eft))] += -1
                elif sum(eft) == deep - 1:
                    eft[eft.index(max(eft))] += 1
                else:
                    print('ERROR IN SUM!',ratio_k,ratio_fmol,eft,sum(eft))
            else: 
                # if existins lots of maximum in eftitem, then choosins the maximum ratio in ratiok to add difefficiency
                multi_maxsite = [[s,ratio_k[s]] for s in range(4) if eft[s] == maxvalue]  #[multimaxsite, subratio in ratiok]
                multi_maxsite_sort = sorted(multi_maxsite,key=(lambda x:x[-1]),reverse=True) # to sort multimaxsite by subratio in ratiok
                multi_maxratio = list(np.array(multi_maxsite_sort).T[1]).count(multi_maxsite_sort[0][1]) # to ensure the num of maximum of subratio in ratiok  , to Turn the multimaxsite
                if multi_maxratio == 1:
                    if sum(eft) == deep + 1:
                        eft[multi_maxsite_sort[0][0]] += -1
                    elif sum(eft) == deep - 1:
                        eft[multi_maxsite_sort[0][0]] += 1
                    else:
                        print('ERROR IN SUM!',ratio_k,ratio_fmol,eft,sum(eft))
                else: 
                    if sum(eft) == deep + 1:
                        eft[eft.index(max(eft))] += -1
                    elif sum(eft) == deep - 1:
                        eft[eft.index(max(eft))] += 1
                    else:
                        print('ERROR IN SUM!',ratio_k,ratio_fmol,eft,sum(eft))
                        
                        
        comsum = [Factorial(eft[s],ratio_fmol[s]) for s in range(4)]
        error_ratio = Factorial(uneftsum,deep) + uneftsum*math.log10(0.01) + (deep-uneftsum)*math.log10(0.99)
        comsum.append(error_ratio)
        
        uneft2value = 0
        for s in range(len(uneft)):
            uneft2value += Factorial(uneft[s],sum(uneft[s:])) + uneft[s]*math.log10(1/len(uneft))

        comsum.append(uneft2value)
        comsum.sort()
        return sum(comsum)


def Letter_Frequency(ratiok):
    '''#TODO: record the simulate results of single letter'''
    rootFre = '{}/1-FreqAll/{}.txt'.format(root, '.'.join(map(str,ratiok)))
    fw = open(rootFre,'w')

    Fmol = 300000000000
    ratioFmol = [int(r*Fmol/k) for r in ratiok]
    if sum(ratioFmol) != Fmol:
        ratioFmol[ratioFmol.index(max(ratioFmol))] += Fmol - sum(ratioFmol)
    
    for sub_deep in dp_set:
        uneftitem = []
        eftitem = []
        for rd,rk in zip(sub_deep,ratiok): 
            if rk != 0: 
                eftitem.append(rd)
            elif rd != 0 and rk == 0 :
                uneftitem.append(rd)  
                eftitem.append(0)
            else:
                eftitem.append(0)
                
        normal_ratio = Sample_Frequency(ratiok,ratioFmol,eftitem,uneftitem)
        fw.write( '{}\t{}\n'.format(':'.join(map(str,sub_deep)), normal_ratio))
    fw.close()
    return 0


def Infer2Letter(ratiok):
    '''#TODO: Inference from sample into letter'''
    rootFre = '{}/1-FreqAll/'.format(root)
    
    sample2CpDNA = cp.deepcopy(dp_dict)
    for s in range(len(ratiok)):
        normratiok = ':'.join(map(str,ratiok[s]))
        inputpath = '{}/{}.txt'.format(rootFre, '.'.join(map(str,ratiok[s])))
        fi = open(inputpath,'r')
        for line in fi:
            if re.search(r'^\d',line) :
                line = line.strip().split()
                line[1] = float(line[1])
                sample = line[0]
                #TODO: Update
                if line[1] > sample2CpDNA[sample][1]:
                    sample2CpDNA[sample] = [ [normratiok] , line[1] ]
                elif line[1] == sample2CpDNA[sample][1]: 
                    sample2CpDNA[sample][0].append(normratiok)
                else:
                    continue
            else:
                continue
        fi.close()

    '''#TODO: Cluster the sample by the infered letter'''
    letter_sample = cp.deepcopy(k_dict)   
    for key in sample2CpDNA: 
        for s in range(len(sample2CpDNA[key][0])):
            letter_sample[sample2CpDNA[key][0][s]].append([key,sample2CpDNA[key][1]])
    return sample2CpDNA, letter_sample


def ClusterWithLetter(sample2CpDNA, sub_ratio):
    '''#TODO: Cluster the sample by letter'''
    Frepath = '{}/1-FreqAll/{}.txt'.format( root, '.'.join(map(str,sub_ratio)))
    Normpath = '{}/2-Norm/{}.txt'.format(root, '.'.join(map(str,sub_ratio)))

    fi = open(Frepath,'r')
    fo = open(Normpath,'w')
    CpDNA2Cluster = cp.deepcopy(k_dict)
    
    for line in fi:
        if re.search(r'^\d',line):
            line = line.strip().split()
            sample = line[0]  #[line[0], line[1]] = [Sample,Fre]
            '''
             # sample2CpDNA = { Sample : [[CpDNA1,CpDNA2,...],Fre], ... , ... }
             # letter_sample = { CpDNA1 : [[Sample1,Fre],[Sample2,Fre],[Sample3,Fre],...] , CpDNA2 : [[...,...],[...,...]] ] ]
             '''
            for s in sample2CpDNA[sample][0]:
                CpDNA2Cluster[s].append( [line[0], line[1]] )
    fi.close()
    
    for key in CpDNA2Cluster:
        fo.write('<< {} >>\n'.format(key))
        for s in CpDNA2Cluster[key]:
            fo.write('{}\t{}\n'.format(s[0], s[1]))
        fo.write('\n')
    fo.close()
    return 0


def TransRatio_Letter(k_list): 
    '''#TODO: Sta the ratio of one letter transfer to others'''
    transto_path = '{}/1.3-TransRatio_Letter_Ratio.txt'.format(root)
    f_to = open(transto_path,'w')
    
    FilterCpDNA = []    
    trans_to = cp.deepcopy(k_dict)
    
    for sub_ratio in k_list:
        Normpath = '{}/2-Norm/{}.txt'.format(root, '.'.join(map(str,sub_ratio)))
        fi = open(Normpath,'r')
        
        norm_fre_rel = cp.deepcopy(k_dict)
        all_fre, relnorm = [], []
        for line in fi:
            if re.search(r'^<',line):
                norm_cpdna = line.split(' ')[1]
            elif re.search(r'^\d',line):
                sample_fre = float(line.split('\n')[0].split()[1])
                all_fre.append(sample_fre)
                norm_fre_rel[norm_cpdna].append(sample_fre)
            else:
                continue
        fi.close()
        
        all_fre.sort()
        undervalue = all_fre[-1] - 300
        
        sum1 = 0 
        for i in range(len(all_fre)):
            sum1 += pow(10,all_fre[i]-undervalue)

        for sub_key in norm_fre_rel:
            sum2 = 0 
            if norm_fre_rel[sub_key] != []:
                rr = norm_fre_rel[sub_key]
                norm_fre_rel[sub_key] = sorted(rr)
                for s in norm_fre_rel[sub_key]: # [Fre,Fre,Fre,...]
                    sum2 += pow(10,s-undervalue)

                relnorm.append([sub_key,float(sum2/sum1)])
                
            if ':'.join(map(str,sub_ratio))  == sub_key:
                FilterCpDNA.append([sub_key,float(sum2/sum1)])

        relsorted = sorted(relnorm,key=(lambda x:x[1]),reverse=True)
        trans_to[':'.join(map(str,sub_ratio))] = relsorted

        for s in relsorted : 
            f_to.write( '{}\t{}\n'.format(s[0], s[1]))
        f_to.write('\n')
    f_to.close()
    
    
    '''#TODO: Transfer fro other letters'''
    transfrom_path = '{}/1.4-TransFromLetter_Ratio.txt'.format(root)
    f_from = open(transfrom_path,'w')
    trans_from = cp.deepcopy(k_dict)
    for key in trans_to: 
        for s in trans_to[key]:
            trans_from[s[0]].append( [key,s[1]] )
        
    for key in trans_from:
        f_from.write('<< {} >>\n'.format(key))
        trans_from[key] = sorted(trans_from[key],key=(lambda x:x[1]),reverse=True)
        for s in range(len(trans_from[key])):
            f_from.write( '{}\t{}\n'.format(trans_from[key][s][0], trans_from[key][s][1]))
        f_from.write('\n')
    f_from.close()

    '''#TODO: Transfer fro other letters'''
    letter_count = math.factorial(k+3)//(math.factorial(3)*math.factorial(k))
    All2Filt = root + '/1.5-{}To{}.txt'.format(letter_count, pow(2, int(math.log(letter_count, 2))))
    f_order = open(All2Filt,'w')
    FilterCpDNA_sort = sorted(FilterCpDNA,key=(lambda x:x[1]),reverse=True)
    for s in FilterCpDNA_sort:
        f_order.write('{}\t{}\n'.format(s[0], s[1]))
    f_order.close()
    return trans_from, [l[0] for l in FilterCpDNA_sort]

def Generate_Softrule():
    Mkdir( root + '/1-FreqAll/')
    Mkdir( root + '/2-Norm/')

    '''#TODO: Simulate the Frequency with multiprocess'''
    p = Pool(10)
    p.map(Letter_Frequency,k_list)

    '''#TODO: Infer the sample into letter, and cluster sample by the inferenced letter'''
    sample2CpDNA, letter_sample = Infer2Letter(k_list)    
    
    Cluster = partial(ClusterWithLetter, sample2CpDNA)
    p.map(Cluster, k_list)
    
    '''#TODO: Summarizing'''
    trans_from, letter_order = TransRatio_Letter(k_list)
    
    '''#TODO: 
        Note: Generate the  transition library(SoftRule.py) by selecting from potential letters with the highest transition probabilities in the depth, to predict and correct letter errors.
        In the final application of Derrick-cp algorithm, the rules for multiple depths need to be summarized.
    '''
    soft_rule = cp.deepcopy(k_dict)
    for l in trans_from: #lette: [[], [], ...]
        soft_rule[l] = [ ll[0] for ll in trans_from[l] if ll[-1] >= ratio_min ]
    
    
    shutil.rmtree(root + '/1-FreqAll/')
    shutil.rmtree(root + '/2-Norm/')
    return soft_rule, letter_order


def read_arss():
    """
    #TODO: Read arsuments from the command line.
    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-k", "--resolution", required=True, type=int,
                        help="the resolution value")
    parser.add_argument("-d", "--deep", required=True, type=int,
                        help="the depth of each letter")

    parser.add_argument("-f", "--ratio_floor", required=True, type=float,
                        help="the ratio floor")
    parser.add_argument("-r", "--output_root", required=True, type=str,
                        help="the saved path")
    return parser.parse_args()

'''
#TODO: Note that: 
    The transition library in "SoftRule.py" is derived from the outcomes generated by "Generate_SoftRule.py", 
    with both functions generating alternative letter sets to predict the true letter. 
    However, for the sake of computational efficiency in soft-decision strategies, 
    we opt to employ the pre-built letter transition library embedded within "SoftRule.py", 
    rather than calling "Generate_SoftRule.py" to compute letter transition probabilities anew for various sequencing depths.
    
    The "Generate_SoftRule.py" is utilized for computing letter transition probabilities at distinct sequencing depths.
    In the final application of Derrick-cp algorithm, the transition library in "SoftRule.py" is directly called by decoder, instead of "Generate_SoftRule.py".
'''
if __name__ == "__main__":
    #TODO: Global variance
    params = read_arss()
    print("The parameters are:")
    print("Resolution(k) = {}".format(params.resolution))
    print("Depth = {}".format(params.deep))
    print("Ratio_limit = {}".format(params.ratio_floor))
    print("Output_path = {}\n\n".format(params.output_root))
    
    '''#TODO: global variance'''
    k = params.resolution
    deep = params.deep
    root = params.output_root
    ratio_min = params.ratio_floor

    k_list,k_dict = Sumk()
    dp_set,dp_dict = Sumdeep()

    soft_rule, letter_order = Generate_Softrule()
    
    print('#Sta:')
    print('Letter_order:\t{}'.format(letter_order))
    print('soft_rule:\t{}'.format(soft_rule))
