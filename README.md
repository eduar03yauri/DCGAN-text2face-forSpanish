[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)](https://creativecommons.org/licenses/by-nc/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7463973.svg)](https://doi.org/10.5281/zenodo.7463973)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)
[![Documentation Sent2Vec+CelebA](https://img.shields.io/badge/Documentation-Sent2vec%2BCelebA-blue)](https://huggingface.co/oeg/Sent2vec_CelebA_Sp)
[![Documentation RoBERTa+CelebA](https://img.shields.io/badge/Documentation-RoBERTa%2BCelebA-blue)](https://huggingface.co/oeg/RoBERTa-CelebA-Sp)
[![Dataset CelebA - Sent2Vec](https://img.shields.io/badge/Dataset-CelebA%20--%20Sent2Vec-blue)](https://huggingface.co/datasets/oeg/CelebA_Sent2Vect_Sp)
[![Dataset CelebA - RoBERTa](https://img.shields.io/badge/Dataset-CelebA%20--%20RoBERTa-blue)](https://huggingface.co/datasets/oeg/CelebA_RoBERTa_Sp)

# Generative Adversarial Networks for Text-to-Face Synthesis & Generation: A Quantitative-Qualitative Analysis of Natural Language Processing Encoders for Spanish

This repository contains the code, models and corpus of the project _"Generative Adversarial Networks for Text-to-Face Synthesis & Generation: A Quantitative-Qualitative Analysis of Natural Language Processing Encoders for Spanish"_. 

This work develops a study to generate images of faces from a textual description in Spanish. A cDCGAN was used as a generator, and a comparison of the RoBERTa-large-bne, RoBERTa -large-bne-celebAEs-UNI (our model) and Sent2vec The last two models were trained using a Spanish descriptive corpus of the CelebA image dataset.

## Autors:
- [Eduardo Yauri Lozano](https://github.com/eduar03yauri)
- [Manuel Castillo-Cara](https://github.com/manwestc)

## Licensing information
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Licencia Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All code and resources of the present work in this repository are under the license <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Licencia Creative Commons Atribución-NoComercial 4.0 Internacional</a>.


## RoBERTa+CelebA

RoBERTa base BNE trained with data from the descriptive text corpus of the CelebA dataset
The model can be found at the following url:
[RoBERTa model in Drive](https://huggingface.co/oeg/RoBERTa-CelebA-Sp).

### Description
The new model called RoBERTa-base-bne-celebAEs-UNI has been generated as a result of training the base model [RoBERTa-large-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne) with a descriptive text corpus of the CelebA dataset in Spanish. For the training, a specific corpus with 249,999 entries was prepared. Each entry is made up of two sentences and their respective similarity value, value between 0 and 1, calculated using the [Spacy](https://spacy.io/) library on their English pairs. You can download said repository from this repository at the following link or from the [Huggingface repository](https://huggingface.co/oeg/RoBERTa-CelebA-Sp). The total training time using the Sentence-transformer library was 42 days using all the available GPUs of the server, and with exclusive dedication.

A comparison was made between the Spearman's correlation for 1000 test sentences between the base model and our trained model. 
As can be seen in the following table, our model obtains better results (correlation closer to 1).

| Models            | Spearman's correlation |
|    :---:          |     :---: |
| RoBERTa-base-bne  | 0.827176427 | 
| RoBERTa-celebA-Sp | 0.999913276 | 

### How to use
- Download the model (full directory) from Drive or [Huggingface repository](https://huggingface.co/oeg/RoBERTa-CelebA-Sp). 
- The downloaded model is in a directory named RoBERTa-base-bne-celebAEs-UNI.
- Move the downloaded directory to the same directory where the Python code that will use it is located.
- Install the Sentence-transformer library for python using the following command. To learn more about the management of the library, visit the following [link](https://www.sbert.net/).
```
pip install -U sentence-transformers
```
- Write the following code in the Python file to call the library and the model. Captions must be made up of lists of one or more sentences in Spanish.

```python
from sentence_transformers import SentenceTransformer, InputExample, models, losses, util, evaluation
model_sbert = SentenceTransformer('roberta-large-bne-celebAEs-UNI')
caption = ['La mujer tiene pomulos altos. Su cabello es de color negro. Tiene las cejas arqueadas y la boca ligeramente abierta. La joven y atractiva mujer sonriente tiene mucho maquillaje. Lleva aretes, collar y lapiz labial.']
vectors = model_sbert.encode(captions)
print(vector)
```
- As a result, the encoder will generate a numeric vector whose dimension is 1024.

```python
>>$ print(vector)
>>$ [0.2,0.5,0.45,........0.9]
>>$ len(vector)
>>$ 1024
```

## Sent2Vec+CelebA
Sent2vec trained with data from the descriptive text corpus of the CelebA dataset

The model can be found at the following url:
[Sent2vec Model in Drive](https://drive.google.com/drive/folders/188iDo2aBiWdTbZ1jicZV0k7gXC7_Tr_a?usp=sharing) or [HuggingFace](https://huggingface.co/oeg/Sent2vec_CelebA_Sp/)

### Description
Sent2vec can be used directly for English texts. For this purpose, all you have to do is download the library and enter the text to be coded, since most 
of these algorithms were trained using English as the original language. However, since this work is used with text in Spanish, it has been necessary 
to train it from zero in this new language. This training was carried out using the generated corpus ([in this respository](https://huggingface.co/datasets/oeg/CelebA_Sent2Vect_Sp)) 
with the following process:
- A corpus composed of a set of descriptive sentences of characteristics of each of the faces of the CelebA dataset in Spanish has been generated.
  A total of 192,209 sentences are available for training.
- Apply a pre-processing consisting of removing accents. _stopwords_ and connectors were retained as part of the sentence structure during training.
- Install the libraries _Sent2vec_ and _FastText_, and configure the parameters. The parameters have been fixed empirically after several
- tests, being: 4,800 dimensions of feature vectors, 5,000 epochs, 200 threads, 2 n-grams and a learning rate of 0.05.

In this context, the total training time lasted 7 hours working with all CPUs at maximum performance. 
As a result, it generates a _bin_ extension file which can be downloaded from this repository.

### How to use
- Download the model from Drive or [Huggingface repository](https://huggingface.co/oeg/Sent2vec_CelebA_Sp/). 
- The downloaded model is in a file named **sent2vec_celebAEs-UNI.bin**.
- Move the downloaded file to the same directory where the Python code that will use it is located.
- Install the libraries _Sent2vec_ and _FastText_. To learn more about the management of the libraries, visit the following [link](https://ilmoirfan.com/how-to-train-sent2vec-model/).
- Write the following code in the Python file to call the library and the model.

```python
import sent2vec
Model_path="sent2vec_celebAEs-UNI.bin"
s2vmodel = sent2vec.Sent2vecModel()
s2vmodel.load_model(Model_path)
caption = """El hombre luce una sombra a las 5 en punto. Su cabello es de color negro. Tiene una nariz grande con cejas tupidas. El hombre se ve atractivo"""
vector =  s2vmodel.embed_sentence(caption)
print(vector)
```
- As a result, the encoder will generate a numeric vector whose dimension is 4800.

```python
>>$ print(vector)
>>$ [[0.1,0.87,0.51,........0.7]]
>>$ len(vector[0])
>>$ 4800
```

## Institutions
<kbd><img src="https://www.uni.edu.pe/images/logos/logo_uni_2016.png" alt="Universidad Politécnica de Madrid" width="110"></kbd>
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-oeg.png" alt="Ontology Engineering Group" width="100"></kbd> 
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-upm.png" alt="Universidad Politécnica de Madrid" width="100"></kbd>
<kbd><img src="https://raw.githubusercontent.com/oeg-upm/TINTO/main/assets/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="90"></kbd> 
