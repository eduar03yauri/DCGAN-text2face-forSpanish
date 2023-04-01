# RoBERTa base BNE trained with data from the descriptive text corpus of the CelebA dataset
The model can be found at the following url:
[RoBERTa model in Drive](https://drive.google.com/drive/folders/1uIxn6CPPNRnThRRsAs2G9_i3gRfDfxof?usp=sharing).

## Description
The new model called RoBERTa-base-bne-celebAEs-UNI has been generated as a result of training the base model [RoBERTa-large-bne](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne) with a descriptive text corpus of the CelebA dataset in Spanish. For the training, a specific corpus with 249,000 entries was prepared. Each entry is made up of two sentences and their respective similarity value, value between 0 and 1, calculated using the [Spacy](https://spacy.io/) library on their English pairs. You can download said repository from this repository at the following link or from the [Huggingface repository](https://huggingface.co/oeg/RoBERTa-CelebA-Sp). The total training time using the Sentence-transformer library was 42 days using all the available GPUs of the server, and with exclusive dedication.

A comparison was made between the Spearman's correlation for 1000 test sentences between the base model and our trained model. 
As can be seen in the following table, our model obtains better results (correlation closer to 1).

| Models            | Spearman's correlation |
|    :---:          |     :---: |
| RoBERTa-base-bne  | 0.827176427 | 
| RoBERTa-celebA-Sp | 0.999913276 | 

## How to use
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
