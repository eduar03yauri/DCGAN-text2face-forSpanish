###########################################################################################################################
#  Autor: Eduardo Yauri Lozano
#  Licencia: This code is licensed under the Creative Commons Atribución-NoComercial 4.0 Internacional licence.
#  Description:
#  This code makes use of the sentence-transformer library to train the RoBERTa-large-bne base model with a corpus of
#  descriptive text dataset CelebA in Spanish, in order to improve its performance. During training there must be a number
#  of entries that must be used for training and testing, therefore previously divide the corpus 
#  sbert_corpustrain_spanish.txt (published in this repository) into two files with an appropriate number of entries in 
#  each, i.e 249000 for training and 1000 for testing.
###########################################################################################################################
from sentence_transformers import SentenceTransformer, InputExample, models, losses, util,evaluation
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch
###open train corpus data
with open("caps_bert_train.txt",'r', encoding="utf-8") as c:
	train_samples = []
	for line in c.readlines():
		sentence1 = line.strip().split('|')[0]
		sentence2 = line.strip().split('|')[1]
		label_text = line.strip().split('|')[2]
		train_samples.append(InputExample(texts=[sentence1, sentences2],label=float(label_text)))
	c.close()
print(len(train_samples))
# open test corpus data
with open("caps_bert_test.txt",'r', encoding="utf-8") as cc:
	sentences1 = []
	sentences2 = []
	scores = []
	for line in cc.readlines():
		sentences1.append(line.strip().split('|')[0])
		sentences2.append(line.strip().split('|')[1])
		scores.append(float(line.strip().split('|')[2]))
	cc.close()
print(len(sentences1))
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores,write_csv=True)
batch_size = 1024
loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
#define base model
bert = models.Transformer('PlanTL-GOB-ES/roberta-large-bne')
pooler = models.Pooling(bert.get_word_embedding_dimension(),pooling_mode_mean_tokens=True)
model = SentenceTransformer(module=[bert, pooler])
print(model)
#use GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(f'moved to {device}')
loss = losses.CosineSimilarityLoss(model)
epochs = 200 # set epochs
print("epocas = "+str(epochs))
warmup_steps = int(len(loader) * epochs * 0.1)
model.fit(
	train_objectives=[(loader, loss)],
	epochs=epochs,
	warmup_steps=warmup_step,
    ##set name of new model
	output_path='roberta-large-bne_add_celeba',
	show_progress_bar = True,
	save_best_model = True,
	evaluator=evaluator,
	evaluation_steps=300
)
torch.cuda.empty_cache()