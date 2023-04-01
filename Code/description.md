## Description
The **Code** directory contains the codes for training the RoBERTa encoder, as well as the GAN network.
```bash
├── sbert_training_roberta_sp.py
├── training-cDCGAN-sbert.py
├── training-cDCGAN-sent2vec.py
└── Desciption.md
```
## sbert_training_roberta_sp.py
This code makes use of the sentence-transformer library to train the RoBERTa-large-bne base model with a corpus of
descriptive text dataset CelebA in Spanish, in order to improve its performance. During training there must be a number
of entries that must be used for training and testing, therefore previously divide the corpus 
sbert_corpustrain_spanish.txt (published in this repository) into two files with an appropriate number of entries in 
each, i.e 249000 for training and 1000 for testing.

## training-cDCGAN-sbert.py
This code implements the training of a cDCGAN (based on the architecture proposed in the Text2faceGAN work) using the 
CelebA image dataset and its respective descriptive corpus in Spanish. To encode the sentences, the sentence-transformer 
library was used together with an own model trained as part of the present work called
RoBERTa_CelebA_Sp (robert-large-bne-celebAEs-UNI).

## training-cDCGAN-sent2vec.py
This code implements the training of a cDCGAN (based on the architecture proposed in the Text2faceGAN work) using the 
CelebA image dataset and its respective descriptive corpus in Spanish. To encode the sentences, the sent2vec 
library.

