import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self,num_output_channels):
        self.batch_size = 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.lambda = .01 # regularization


        def create_layer_with_batch_norm_and_relu(num_filters,kernel_size,batch_norm=True,dropout=False,downsample=True):
            layer = tf.keras.Sequential()
            if encoder:
                conv_output_layer = tf.keras.layers.Conv2D(num_filters,kernel_size,strides=2, padding='same')
            else:
                conv_output_layer = tf.keras.layers.conv2DTranspose(num_filters,kernel_size,strides=2,padding='same')
            if batch_norm:
                batch_norm_layer = tf.keras.layers.BatchNormalization()

            leaky_relu_layer = tf.keras.layers.LeakyReLU(0.2)

            layer.add(conv_output_layer)
            layer.add(batch_norm_layer)
            if dropout:
                layer.add(tf.keras.layers.Dropout(0.5))
            layer.add(leaky_relu_layer)

            return layer


        ### TODO add skip connections (U Net). also modify decoder to be consistent with unet. section 6.1.1
        self.encoder = tf.keras.Sequential()
        self.encoder.add(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False,dropout=False))
        self.encoder.add(create_layer_with_batch_norm_and_relu(128,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(256,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(512,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(512,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(512,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(512,4))
        self.encoder.add(create_layer_with_batch_norm_and_relu(512,4))
        self.decoder = tf.keras.Sequential()
        self.decoder.add(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        self.decoder.add(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        self.decoder.add(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        self.decoder.add(create_layer_with_batch_norm_and_relu(256,4,downsample=False))
        self.decoder.add(create_layer_with_batch_norm_and_relu(128,4,downsample=False))
        self.decoder.add(create_layer_with_batch_norm_and_relu(64,4,downsample=False))

        self.generator = tf.keras.Sequential()
        self.generator.add(self.encoder)
        self.generator.add(self.decoder)
        self.generator.add(tf.keras.layers.Conv2D(num_output_channels,4,strides=2,padding='same'))
        self.generator.add(tf.keras.layers.Activation('tanh'))

        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(128,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(256,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(512,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(1,4,batch_norm=False))
        self.discriminator.add(tf.keras.layers.Activation('sigmoid'))

    def call(self,original_images,real_transformed_images):
        gen_transformed_images = self.generator(original_images)
        prob_real_given_real = self.discriminator(original_images,transformed_images)
        prob_gen_given_gen = self.discriminator(original_images,gen_transformed_images)
        return gen_transformed_images,prob_real_given_real,prob_gen_given_gen

    def loss(self):
        pass


def train(model,train_inputs,train_labels):
    # repeatedly update discriminator then generator (or vice versa?)
    # going to have to multiply by -1 for one of them since one is maximizing and other is minimizing
