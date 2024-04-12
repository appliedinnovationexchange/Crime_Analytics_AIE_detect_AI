from os import listdir
import os
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
import numpy as np
import cv2

# # load all images in a directory into memory
# def load_images(path, size=(256,512)):
# 	src_list, tar_list = list(), list()
# 	# enumerate filenames in directory, assume all are images
# 	for filename in listdir(path):
# 		# load and resize the image
# 		pixels = load_img(path + filename, target_size=size)
# 		# convert to numpy array
# 		pixels = img_to_array(pixels)
# 		# split into satellite and map
# 		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
# 		src_list.append(sat_img)
# 		tar_list.append(map_img)
# 	return [asarray(src_list), asarray(tar_list)]
#
# def src_load_images(path, size=(0,255)):
# 	src_list = list()
# 	for filename in listdir(path):
# 		pixels = load_img(path + filename, target_size=size)
# 		# convert to numpy array
# 		pixels = img_to_array(pixels)
# 		src_list.append(pixels)
# 	return asarray(src_list)
#
# def tar_load_images(path, size=(0,255)):
# 	tar_list = list()
# 	for filename in listdir(path):
# 		pixels = load_img(path + filename, target_size=size)
# 		# convert to numpy array
# 		pixels = img_to_array(pixels)
# 		tar_list.append(pixels)
# 	return asarray(tar_list)

def src_load_images(path, size=(256, 256)):
    src_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        src_list.append(pixels)
    return asarray(src_list)

def tar_load_images(path, size=(256, 256)):
    tar_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        tar_list.append(pixels)
    return asarray(tar_list)


def load_images(src_path, tar_path):
	src_data = src_load_images(src_path)
	tar_data = tar_load_images(tar_path)
	return [src_data, tar_data]

def load_images_with_names(path):
	path_path = path
	images = []
	image_names = []

	for filename in sorted(os.listdir(path_path)):
		img = cv2.imread(os.path.join(path_path, filename))

		if img is not None:
			images.append(img)
			image_names.append(filename)

	return np.array(images), image_names



# src_path = 'named_dataset/256dataset_masked/'
src_path = '50pictures/masked/'
# tar_path = 'named_dataset/256dataset/'
tar_path = '50pictures/unmasked/'
# Load source images with their filenames
src_images, src_image_names = load_images_with_names(path=src_path)
# Load target images with their filenames
tar_images, tar_image_names = load_images_with_names(path=tar_path)

# Sort images and names based on names
src_images_sorted = [img for _, img in sorted(zip(src_image_names, src_images))]
src_image_names_sorted = sorted(src_image_names)
tar_images_sorted = [img for _, img in sorted(zip(tar_image_names, tar_images))]
tar_image_names_sorted = sorted(tar_image_names)
# Check the sorted images and their names
#
# print('Sorted Source Images:', len(src_images_sorted))
#
# print('Sorted Source Image Names:', src_image_names_sorted)
#
# print('Sorted Target Images:', len(tar_images_sorted))
#
# print('Sorted Target Image Names:', tar_image_names_sorted)

# dataset path
# # path = 'maps/train/'
# src_path = '256dataset_masked/'
# tar_path = '256dataset/'
# load dataset
# [src_images, tar_images] = load_images(src_path, tar_path)
print('Loaded: ', src_images.shape, tar_images.shape)


n_samples = 5
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()

#######################################

from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#Define data
# load and prepare training images
data = [src_images, tar_images]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from datetime import datetime
start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=5, n_batch=10)
#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

stop1 = datetime.now()
#Execution time of the model
execution_time = stop1-start1
print("Execution time is: ", execution_time)

#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

#################################################

#Test trained model on a few images...

from keras.models import load_model
from numpy.random import randint
model = load_model('saved_model_10epochs.h5')

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()



[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)
