# Derrick-cp  
---This repo is still under construction---  
Derrick-cp is used for convert between digital information and DNA sequences (encoding and decoding) in large-scale composite DNA-storage systems. The encoding process involves converting digital files into composite DNA sequences, including randomization, adding CRC-32 and Reed-Solomon(RS) codes which form CRC matrices and RS block, and then translating them into DNA sequences; The decoding process utilizes soft decision decoding strategies to correct errors and to convert the composite DNA sequences back to the original files as high precision as possible.

## Composite DNA  
(1)the composite DNA encoding and decoding with hard decision in the paper:   
Data storage in DNA with fewer synthesis cycles using composite DNA letters  
https://doi.org/10.1038/s41587-019-0240-x  
(2)the composite DNA encoding and decoding with Derrick-cp in the paper:  XXX  

## Encoded data  
The encoded files are available at https://github.com/xu-yaping/Derrick-cp/tree/main/Data  
(1)info_to_code.tar.gz - Message from Erlich and Zielinski (DOI: 10.1126/science.aaj2038)  
(2)pt003.zip - Bi-lingual HTML version of the Bible from Mamre (https://www.mechon-mamre.org/p/pt/pt0101.htm)  

## Sequencing results  
All NGS raw data in available at https://www.ebi.ac.uk/ena/data/view/PRJEB32427  

## General usage
### Encode:   
convert the digital file into composite DNA sequences, encoding with RS(45,41) and CRC-32  
`Python3 Encoder_Digital2Letter.py [-h] -k RESOLUTION -c ALPHABET_SIZE -i DIGITAL_PATH -o ENCODED_PATH -p ENCODED_COL_PATH
                    [-rsK RSK] [-rsN RSN]`  

### Decode:   
convert composite DNA sequences into the digital file, decoding with RS(45,41) in soft decision strategy.  
`python3 Decoder_Letter2Digital.py [-h] -k RESOLUTION -c ALPHABET_SIZE -i INFERRED_PATH -o SAVED_DIGITAL_PATH -p
                           SAVED_LETTER_PATH [-rsK RSK] [-rsN RSN]`   
