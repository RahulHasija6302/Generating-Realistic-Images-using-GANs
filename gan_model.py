import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the CelebA dataset
(X_train, _), (_, _) = tf.keras.datasets.celeba.load_data()

# Preprocess the data
X_train = (X_train.astype('float32') - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

# Define the generator network
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*256, input_shape=(100,)),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
])

# Define the discriminator network
discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 1]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define the GAN model
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Train the GAN model
epochs = 50
batch_size = 128
steps_per_epoch = len(X_train) // batch_size

for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    for step in range(steps_per_epoch):
        # Generate random noise as input to the generator network
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        
        # Generate fake images using the generator network
        fake_images = generator.predict(noise)
        
        # Select a batch of real images from the dataset
        real_images = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
        
        # Concatenate the real and fake images into a single batch
        X = np.concatenate([real_images, fake_images])
        
        # Create labels for the real and fake images
        y = np.zeros(2*batch_size)
        y[:batch_size] = 0.9
        
        # Train the discriminator network on the real and fake images
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y)
        
        # Train the generator network to generate images that can fool the discriminator
        noise = np.random.normal(0, 1, size=(batch_size, 100))
        y = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y)
        
    # Generate a sample of fake images after each epoch
    noise = np.random.normal(0, 1, size=(16, 100))
    fake_images = generator.predict(noise)
    
    # Plot the fake images and save them to disk
    fig, axs = plt.subplots(4, 4)
    count = 0
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(fake_images[count,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            count += 1
    plt.savefig('gan_images_epoch_{}.png'.format(epoch+1))
    plt.close()
