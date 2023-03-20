# Sent2vec trained with data from the descriptive text corpus of the CelebA dataset

The model can be found at the following url:
[Sent2vec Model in Drive](https://drive.google.com/drive/folders/188iDo2aBiWdTbZ1jicZV0k7gXC7_Tr_a?usp=sharing)

## Description
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

## How to use
- Download the model from Drive or [Huggingface repository](https://huggingface.co/oeg/Sent2vec_CelebA_Sp/). 
- The downloaded model is in a file named **sent2vec_celebAEs-UNI.bin**.
- Move the downloaded file to the same directory where the Python code that will use it is located.
- Install the libraries _Sent2vec_ and _FastText_. To learn more about the management of the libraries, visit the following [link](https://ilmoirfan.com/how-to-train-sent2vec-model/).
- Write the following code in the Python file to call the library and the model.

```python
import sent2vec
Model_path="sent2vec_celebAEs-UNI.bin"
s2vmodel = sent2vec.Sent2vecModel()
s2vmode.load_model(Model_path)
caption = """El hombre luce una sombra a las 5 en punto. Su cabello es de color negro. Tiene una nariz grande con cejas tupidas. El hombre se ve atractivo"
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

