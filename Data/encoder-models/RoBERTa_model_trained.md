# RoBERTa base BNE trained with data from the descriptive text corpus of the CelebA dataset
The model can be found at the following url:
[RoBERTa model in Drive](https://drive.google.com/drive/folders/1uIxn6CPPNRnThRRsAs2G9_i3gRfDfxof?usp=sharing).

## Description
The new model called *RoBERTa-base-bne-celebAEs-UNI* has been generated as a result of training the base model *RoBERTa-base-bne* with a descriptive text corpus of the CelebA dataset in Spanish. For the training, a specific corpus with 249,999 entries was prepared. Each entry is made up of two sentences and their respective similarity value, value between 0 and 1, calculated using the *Spacy* library on their English pairs. The total training time using the *Sentence-BERT* library was 42 days using all the available GPUs of the server, and with exclusive dedication. 
## How to use
- Download the model (full directory) from Drive or other oficial media. 
- The downloaded model is in a directory named RoBERTa-base-bne-celebAEs-UNI.
- Move the downloaded directory next to the Python code in Spanish that will use it.
- 
