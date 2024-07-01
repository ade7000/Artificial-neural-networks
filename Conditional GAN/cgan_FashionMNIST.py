# Bringing dependencies
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot as plt
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Model
# Bring in the layers for the neural network
from tensorflow.keras.layers import Input, Conv2D, Dense, Embedding, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, Concatenate, Conv2DTranspose
import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint

# scale from [0,255] to [-1, 1]
def scale_images(data): 
    image = data['image']
    return image -127.5 / 127.5

if tf.test.is_gpu_available():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

# Bring the Fashion dataset.
# Only the train is needed, as we are not interested in testing the discriminator
ds = tfds.load('fashion_mnist', split='train', data_dir = 'data', batch_size = 128, shuffle_files = True, download = True)
ds = ds.map(scale_images) 
ds = ds.cache()
ds = ds.prefetch(64)

ds.as_numpy_iterator().next()['label']

ds.as_numpy_iterator().next().shape

# Setup connection aka iterator
dataiterator = ds.as_numpy_iterator()

# Getting data out of the pipeline
dataiterator.next()['image']

# Setup the subplot formatting 
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# Loop four times and get images 
for idx in range(4): 
    # Grab an image and label
    sample = dataiterator.next()
    # Plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(sample['image']))
    # Appending the image label as the plot title 
    ax[idx].title.set_text(sample['label'])


def build_generator(random_values, nr_classes = 10): 

    in_label = Input(shape=(1,))

    label_image = Embedding(nr_classes,49)(in_label)
    label_image = Reshape((7,7,1))(label_image)

    in_lat = Input(shape=(random_values))
    
    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    gen = Dense(7*7*128, input_dim=128)(in_lat)
    gen = LeakyReLU(0.2)(gen)
    gen = Reshape((7,7,128))(gen)

    merge = Concatenate()([gen, label_image]) 
    # Upsampling block 1 
    gen = UpSampling2D()(merge)
    gen = Conv2D(128, 5, padding='same')(gen)
    gen = LeakyReLU(0.2)(gen)
    
    # Upsampling block 2 
    gen = UpSampling2D()(gen)
    gen= Conv2D(128, 5, padding='same')(gen)
    gen = LeakyReLU(0.2)(gen)
    
    # Convolutional block 1
    gen = Conv2D(128, 4, padding='same')(gen)
    gen = LeakyReLU(0.2)(gen)
    
    # Convolutional block 2
    gen = Conv2D(128, 4, padding='same')(gen)
    gen = LeakyReLU(0.2)(gen)
    
    # Conv layer to get to one channel
    out_layer = Conv2D(1, 4, padding='same', activation='tanh')(gen)
    
    model = Model([in_lat, in_label], out_layer)
    return model

generator = build_generator()

generator.summary()

def build_discriminator(in_shape=(28,28,1), nr_classes = 10): 
    in_label = Input(shape=(1,))  #Shape 1
    
    label_image = Embedding(nr_classes, 50)(in_label)
    nr_nodes = in_shape[0] * in_shape[1]  #32x32 = 1024. 
    label_image = Dense(nr_nodes)(label_image)  #Shape = 1, 1024
	# reshape to additional channel
    label_image = Reshape((in_shape[0], in_shape[1], 1))(label_image)

    # image input
    in_image = Input(shape=in_shape) #28x28x1
	# concat label as a channel
    merge = Concatenate()([in_image, label_image]) #28x28x2
    
    # First Conv Block
    dis = Conv2D(32, 5)(merge)
    dis = LeakyReLU(0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # Second Conv Block
    dis = Conv2D(64, 5)(dis)
    dis = LeakyReLU(0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # Third Conv Block
    dis = Conv2D(128, 5)(dis)
    dis = LeakyReLU(0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # Fourth Conv Block
    dis = Conv2D(256, 5)(dis)
    dis = LeakyReLU(0.2)(dis)
    dis = Dropout(0.4)(dis)
    
    # Flatten then pass to dense layer
    dis = Flatten()(dis)
    dis = Dropout(0.4)(dis)

    out_layer = Dense(1, activation='sigmoid')(dis) 
    
	# define model
    ##Combine input label with input image and supply as inputs to the model. 
    model = Model([in_image, in_label], out_layer)
	# compile model
    return model

discriminator = build_discriminator()

discriminator.summary()

def generate_latent_points(latent_dim, nr_samples, nr_classes=10):
	# generate points in the latent space
	x_input = tf.random.normal((nr_samples, latent_dim, 1))
	# generate labels
	labels = randint(0, nr_classes, nr_samples)
	return [x_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	return [images, labels_input]

# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers import Adam
# Binary cross entropy is going to be the loss for both 
from tensorflow.keras.losses import BinaryCrossentropy

g_opt = Adam(learning_rate=0.0002, beta_1=0.5) 
d_opt = Adam(learning_rate=0.00001) 
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


class FashionGAN(Model): 
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class 
        super().__init__(*args, **kwargs)
        
        # Create attributes for gen and disc
        self.generator = generator 
        self.discriminator = discriminator 
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs): 
        # Compile with base class
        super().compile(*args, **kwargs)
        
        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 

    def train_step(self, batch):
        # Get the data 
        real_images = batch
        fake_images = self.generator(generate_latent_points(128, 128), training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True) 
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            
            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
            
            # Add some noise to the TRUE outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply backpropagation - nn learn 
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as g_tape: 
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128,128,1)), training=True)
                                        
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
            
        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        return {"d_loss":total_d_loss, "g_loss":total_g_loss}

# Create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)

# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)


import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))


######################################################
hist = fashgan.fit(ds, epochs=2, callbacks=[ModelMonitor()])

######################################################
## Save the model
generator.save_weights('generator.h5')
discriminator.save_weights('discriminator.h5')

######################################################
## Plot the loss
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()

######################################################
generator = load_weights('generatormodel.h5)

# generate multiple images

latent_points = generate_latent_points(128, 128)
# specify labels - generate 10 sets of labels each gping from 0 to 9
labels = asarray([x for _ in range(10) for x in range(10)])
# generate images
X  = generator.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)
# plot the result (10 sets of images, all images in a column should be of same class in the plot)
# Plot generated images 
def show_plot(examples, n):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
	plt.show()
    
show_plot(X, 10)


