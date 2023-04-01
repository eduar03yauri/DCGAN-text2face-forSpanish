###########################################################################################################################
#  Autor: Eduardo Yauri Lozano
#  Licencia: This code is licensed under the Creative Commons Atribución-NoComercial 4.0 Internacional licence.
#  Description:
#  This code implements the training of a cDCGAN (based on the architecture proposed in the Text2faceGAN work) using the 
#  CelebA image dataset and its respective descriptive corpus in Spanish. To encode the sentences, the sentence-transformer 
#  library was used together with an own model trained as part of the present work called
#  RoBERTa-CelebA-Sp (robert-large-bne-celebAEs-UNI).
###########################################################################################################################
from sentence_transformers import SentenceTransformer, InputExample, models, losses, util, evaluation
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf
from keras.preprocessing.image import img_to_array,load_img
from PIL import Image
import math
import random
from collections import Counter
import nltk
import json
import h5py
import pickle
import re
import urllib.request
from mpl_toolkits.axes_grid1 import ImageGrid
import zipfile
from keras.models import Model,Sequential
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
#from keras.layers import Input,Dense,Reshape,concatenate,Flatten,Lambda,LeakyReLU, Dropout
from keras.layers.core import Activation
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization,UpSampling2D,Conv2D,MaxPooling2D,Conv2DTranspose,Input,Dense,Reshape,concatenate,Flatten,Lambda,LeakyReLU,Dropout,ReLU
#from keras.layers.convolutional import UpSampling2D,Conv2D,MaxPooling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from keras.initializers import TruncatedNormal,Zeros,RandomNormal,Constant
from keras import backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import time
from spectral_normalization import SpectralNormalization

from tqdm import tqdm

print("GPU:",tf.test.gpu_device_name(),"TF version:",tf.__version__)

#definiendo la libreria y cargando el modelo
model_sbert = SentenceTransformer('roberta-large-bne-celebAEs-UNI')

class DCGan(object):
	model_name='dc_ganES'
	def __init__(self):
		self.generator=None
		self.discriminator=None
		self.model=None
		self.img_width=64
		self.img_height=64
		self.img_channels=3
		self.text_input_dim=1024
		self.randoms_input_dim=100
		self.config=None
	def get_config_file_path(model_dir_path):
		return os.path.join(model_dir_path,DCGan.model_name+'-config.npy')
	def get_weight_file_path(model_dir_path,model_type):
		return os.path.join(model_dir_path,DCGan.model_name+'-'+model_type+'-weights.h5')
	def create_model(self):
		init_img_width=4
		init_img_height=4
		k_ini = RandomNormal(mean = 0, stddev=0.02)

		random_input=Input((self.random_input_dim,))
		text_input1=Input((self.text_input_dim,))
		text_layer1=Dense(256,kernel_initializer=RandomNormal(stddev=0.02))(text_input1)
		text_layer1=ReLU()(text_layer1)

		merged=concatenate([random_input,text_layer1])
		generator_layer = merged
		generator_layer=Activation('tanh')(merged)

		generator_layer=Dense(512*init_img_height*init_img_width,kernel_initializer=k_ini)(generator_layer)
		generator_layer=ReLU()(generator_layer)

		generator_layer=Reshape((init_img_height,init_img_width,512),input_shape=(512*init_img_height*init_img_width,))(generator_layer)

		generator_layer=Conv2DTranspose(256,kernel_size=4,strides=(2,2),padding='same',kernel_initializer=k_ini)(generator_layer)
		generator_layer= BatchNormalization(momentum=0.4)(generator_layer) 
		generator_layer=ReLU()(generator_layer)

		generator_layer=Conv2DTranspose(64,kernel_size=5,strides=(2,2),padding='same',kernel_initializer=k_ini)(generator_layer)
		generator_layer=ReLU()(generator_layer)

		generator_layer=Conv2DTranspose(64,kernel_size=4,strides=(2,2),padding='same',kernel_initializer=k_ini)(generator_layer)
		generator_layer= BatchNormalization(momentum=0.4)(generator_layer)
		generator_layer=ReLU()(generator_layer)

		generator_layer=Conv2DTranspose(self.img_channels,kernel_size=5,strides=(2,2),padding='same',kernel_initializer=k_ini)(generator_layer)
		generator_layer=Activation('tanh')(generator_layer)

		generator_layer=Lambda(lambda x:x/2.)(generator_layer)
		generator_output=Lambda(lambda x:x+0.5)(generator_layer)

		self.generator=Model([random_input,text_input1],generator_output, name='generator')
		g_optim=Adam(lr=0.0002,beta_1=0.5)

		self.generator.compile(loss='binary_crossentropy',optimizer=g_optim)
		self.generator.summary()

		#discriminator
		text_input2=Input((self.text_input_dim,))
		text_layer2=Dense(256,kernel_initializer=k_ini)(text_input2)

		img_input2=Input((self.img_height,self.img_width,self.img_channels))

		img_layer2=Conv2D(64,kernel_size=5,padding='same',strides=(2,2),kernel_initializer=k_ini)(img_input2)
        
		img_layer2 = BatchNormalization(momentum=0.4)(img_layer2)
		img_layer2=LeakyReLU(alpha=0.2)(img_layer2)

		img_layer2=Conv2D(128,kernel_size=5,padding='same',strides=(2,2),kernel_initializer=k_ini)(img_layer2)

		img_layer2 = BatchNormalization(momentum=0.4)(img_layer2)
		img_layer2=LeakyReLU(alpha=0.2)(img_layer2)

		img_layer2=Conv2D(256,kernel_size=5,padding='same',strides=(2,2),kernel_initializer=k_ini)(img_layer2)

		img_layer2 = BatchNormalization(momentum=0.4)(img_layer2)
		img_layer2=LeakyReLU(alpha=0.2)(img_layer2)

		text_layer2=Lambda(K.expand_dims,arguments={'axis':1})(text_layer2)
		text_layer2=Lambda(K.expand_dims,arguments={'axis':3})(text_layer2)
		text_layer2=Lambda(K.tile,arguments={'n':(1,8,8,1)})(text_layer2)

		img_layer2=concatenate([img_layer2,text_layer2],axis=3)
		
		img_layer2=Conv2D(512,kernel_size=4,padding='same',strides=(2,2),kernel_initializer=k_ini)(img_layer2)

		img_layer2 = BatchNormalization(momentum=0.4)(img_layer2)
		img_layer2=LeakyReLU(alpha=0.2)(img_layer2)

		img_layer2=Flatten()(img_layer2)
		
		img_layer2 = Dropout(0.2)(img_layer2)

		discriminator_layers=Dense(1,kernel_initializer=k_ini)(img_layer2)
		discriminator_output=Activation('sigmoid')(discriminator_layer)

		self.discriminator=Model([img_input2,text_input2],discriminator_output)
		d_optim=Adam(learning_rate=0.0001,beta_1=0.5)
		self.discriminator.compile(loss='binary_crossentropy',optimizer=d_optim)
		self.discriminator.summary()

		self.discriminator.trainable=False
		model_output=self.discriminator([self.generator.output,text_input1])

		self.model=Model([random_input,text_input1],model_output, name='discriminator')

		self.model.compile(loss='binary_crossentropy',optimizer=g_optim)
	def draw_model(self,path):
		plot_model(self.model,path,show_shapes=True)
		
	def load_batch(self,batch_idx,batch_size,image_label_pairs):
		image_label_pair_batch=image_label_pairs[batch_idx*batch_size:(batch_idx+1)*batch_size]
		image_files_batch=[]
		wrong_image_batch=np.zeros((batch_size,self.img_height,self.img_width,self.img_channels))
		real_image_batch=np.zeros((batch_size,self.img_height,self.img_width,self.img_channels))
		noise=np.zeros((batch_size,self.random_input_dim))
		sent2vec_batch=np.zeros((batch_size,self.text_input_dim))
		for i in range(batch_size):
			normalised_img=image_label_pair_batch[i][0]
			real_image_batch[i,:,:,:]=normalised_img

			idx=random.randint(0,len(image_label_pairs)-1)
			wrong_img=image_label_pairs[idx][0]
			wrong_image_batch[i,:,:,:]=wrong_img

			sent2vec_batch[i,:]=image_label_pair_batch[i][1]
			noise[i,:]=np.random.uniform(-1,1,self.random_input_dim)
			image_files_batch.append(image_label_pairs[i][2])
		return real_image_batch,wrong_image_batch,noise,sent2vec_batch,image_files_batch

	def fit(self,model_dir_path,image_label_pairs,epochs=None,batch_size=None,snapshot_dir_path=None):
		if epochs is None:
			epochs=100
		
		if batch_size is None:
			batch_size=128
		
		self.config=dict()
		self.config['img_width']=self.img_width
		self.config['img_height']=self.img_height
		self.config['random_input_dim']=self.random_input_dim
		self.config['text_input_dim']=self.text_input_dim
		self.config['img_channels']=self.img_channels

		config_file_path=DCGan.get_config_file_path(model_dir_path)

		np.save(config_file_path,self.config)

		n_batches=image_label_pairs.shape[0]//batch_size
		d_loss_list=[]
		g_loss_list=[]
		for epoch in range(epochs):
			epoch_d_loss=0
			epoch_g_loss=0
			start=time.time()
			for batch_idx in range(n_batches):
				real_images_batch,wrong_images_batch,noise,sent2vec_batch,image_files_batch=self.load_batch(batch_idx,batch_size,image_label_pairs)

				fake_images_batch=self.generator.predict([noise,sent2vec_batch],verbose=0)

				self.discriminator.trainable=True
				
				d_loss1=self.discriminator.train_on_batch([real_images_batch,sent2vec_batch],np.array([1]*batch_size))
				d_loss2=self.discriminator.train_on_batch([wrong_images_batch,sent2vec_batch],np.array([0]*batch_size))
				d_loss3=self.discriminator.train_on_batch([fake_images_batch,sent2vec_batch],np.array([0]*batch_size))
				
				self.discriminator.trainable=False

				d_loss=d_loss1+(d_loss2-d_loss3)
				
				g_loss=self.model.train_on_batch([noise,sent2vec_batch],np.array([1]*batch_size))

				epoch_d_loss+=d_loss
				epoch_g_loss+=g_loss

				if (batch_idx+1)%40==0 and snapshot_dir_path is not None:
					generated_images=self.generator.predict([noise,sent2vec_batch],verbose=0)
					self.save_snapshots(generated_images,snapshot_dir_path=snapshot_dir_path,epoch=epoch,batch_idx=batch_idx)
			d_loss_list.append(epoch_d_loss/n_batches)
			g_loss_list.append(epoch_g_loss/n_batches)
			print('Epoch: '+str(epoch+1)+'/'+str(epochs)+' epoch_duration: '+str(time.time()-start)+ ' d1: '+str(d_loss1/n_batches)+' d2: '+str(d_loss2/n_batches)+' d3: '+str(d_loss3/n_batches)+ ' discriminator_loss: '+str(epoch_d_loss/n_batches)+' generator_loss: '+str(epoch_g_loss/n_batches))
			if(epoch+1)%5==0 or (epoch+1)==epochs:
				self.generator.save_weights(DCGan.get_weight_file_path(model_dir_path,'generator'),True)
				self.discriminator.save_weights(DCGan.get_weight_file_path(model_dir_path,'discriminator'),True)
				with h5py.File('losses_list.h5','w') as out:
					out.create_dataset("discriminator",data=np.array(d_loss_list))
					out.create_dataset("generator",data=np.array(g_loss_list))
	def generate_image_from_text(self,caption,model_sbert):
    
		encoded_text = model_sbert.encode(caption)
		noise=np.random.uniform(-1,1,self.random_input_dim)
		noise=np.expand_dims(noise,axis=1)
		generated_image=self.generator.predict([noise,encoded_text],verbose=1)

		plt.imshow(generated_image[0])
		return generated_image

	def save_snapshots(self,generated_images,snapshot_dir_path,epoch,batch_idx):
		plot_batch(generated_images,DCGan.model_name,epoch,batch_idx,snapshot_dir_path)

def plot_batch(generated_images,model_name,epoch,batch_idx,snapshot_dir_path):
	fig=plt.figure(1)
	grid=ImageGrid(fig,111,nrows_ncols=(2,8),axes_pad=0.05)
	size=2*8
	for i in range(size):
		grid[i].axis('off')
		grid[i].imshow((generated_images[i]*255).astype(np.uint8))
	plt.savefig(os.path.join(snapshot_dir_path,model_name+'-'+str(epoch)+'-'+str(batch_idx)+'.png'))

def img_from_normalised_img(normalised_img):
	image=normalised_img.astype(float)*255
	image=image.astype('uint8')
	return image
def load_normalised_img_and_cap(img_path, caption_file):
	IMAGES_COUNT = 50000
	imgs = []
	names = []
	for pic_file in tqdm(os.listdir(img_path)[:IMAGES_COUNT]):
		pic = Image.open(img_path + pic_file).resize((img_width, img_height))
		pic.thumbnail((img_width, img_height), Image.ANTIALIAS)
		imgs.append(np.uint8(pic))
		names.append(pic_file)
	caps = {}
	#apeturando con utf8
	with open(caption_file,'r',encoding="utf-8") as c:
		for line in c.readlines():
			img_name = line.strip().split('||')[0]
			cap = " ".join(line.split('||')[1].split('|'))
			if cap == "\n" or cap == " \n":
				cap = "Esta es una persona con rostro y nada mas."
			caps[img_name] = cap
	captions=[]
	for i in range(len(names)):
		captions.append(caps.setdefault(names[i], "Esta es una persona con rostro y nada mas.."))
	vectors = captions
	if not os.path.exists('data/sbert_vectors_50k.pkl'):
		model_sbert = SentenceTransformer('roberta-large-bne-celebAEs-UNI')
		vectors = model_sbert.encode(captions)
		with open('data/sbert_vector_50k.pkl','wb') as f:
			pickle.dump(vectors,f)
	else:
		with open('data/sbert_vectors_50k.pkl','rb') as f:
			vectors=pickle.load(f)
	result=[]
	for i in range(len(imgs)):
		imgs[i]=imgs[i]/255 # normalize
		result.append([imgs[i],vectors[i],names[i]])
	return np.array(result)
def resize(img,input_shape):
	height,width=input_shape
	return cv2.resize(img,(width,height))
    
# la dimension de las imagenss sera de 64x64
img_width=64
img_height=64
img_channels=3

WIDTH = 80
HEIGHT = 96

seed=2020
np.random.seed(seed)
model_dir_path='modelos'
#ruta de imagenes de celebA
img_path='data/img_align_celeba/'
### archivo procesado de descripciones textuales
caption_file = 'data/caps_es_proc.txt'
image_label_pairs=load_normalised_img_and_cap(img_path, caption_file)
shuffle(image_label_pairs)

print("data cargada....")

dcgan=DCGan()
dcgan.img_width=64
dcgan.img_height=64
dcgan.img_channels=3
dcgan.random_input_dim=100
dcgan.text_input_dim=1024
batch_size=1024
epochs = 600 #definir la cantidad de epocas
dcgan.create_model()

print("modelo generado...")

with tf.device('/GPU:1'):
	dcgan.fit(model_dir_path=model_dir_path,
		image_label_pairs=image_label_pairs,
		snapshot_dir_path='snapshots',
		batch_size=batch_size,
		epochs=epochs)

dcgan.generator.load_weights(model_dir_path+'/dc_ganES-generator-weights.h5')
print("Entrenamiento finalizado")
