import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self,num_output_channels):
        batch_size = 1
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.999)
        lambda = .01 # regularization


        def create_layer_with_batch_norm_and_relu(num_filters,kernel_size,batch_norm=True,dropout=False,downsample=True):
            layer = tf.keras.Sequential()
            if encoder:
                conv_output_layer = tf.keras.layers.Conv2D(num_filters,kernel_size,strides=2, padding='same')
            else:
                conv_output_layer = tf.keras.layers.conv2DTranspose(num_filters,kernel_size,strides=2,padding='same')
            if batch_norm:
                batch_norm_layer = tf.keras.layers.BatchNormalization()

            leaky_relu_layer = tf.keras.layers.LeakyReLU(0.2)

            layer.append(conv_output_layer)
            layer.append(batch_norm_layer)
            if dropout:
                layer.append(tf.keras.layers.Dropout(0.5))
            layer.append(leaky_relu_layer)

            return layer


        ### TODO add skip connections (U Net). also modify decoder to be consistent with unet. section 6.1.1
        self.generator = []

        encoder = []
        encoder.append(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False,dropout=False))
        encoder.append(create_layer_with_batch_norm_and_relu(128,4))
        encoder.append(create_layer_with_batch_norm_and_relu(256,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        decoder = []
        decoder.append(create_layer_with_batch_norm_and_relu(256,4,dropout=True,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(256,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(128,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(64,4,downsample=False))

        generator.append(encoder)
        generator.append(decoder)
        generator.append(tf.keras.layers.Conv2D(num_output_channels,4,strides=2,padding='same'))
        generator.append(tf.keras.layers.Activation('tanh'))


        self.discriminator = tf.Sequential.keras()
        discriminator.add(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False))
        discriminator.add(create_layer_with_batch_norm_and_relu(128,4,batch_norm=False))
        discriminator.add(create_layer_with_batch_norm_and_relu(256,4,batch_norm=False))
        discriminator.add(create_layer_with_batch_norm_and_relu(512,4,batch_norm=False))
        discriminator.add(create_layer_with_batch_norm_and_relu(1,4,batch_norm=False))
        discriminator.add(tf.keras.layers.Activation('sigmoid'))

    def call(self,original_images,real_transformed_images):
        # call the generator
        encoder = self.generator[0]
        decoder = self.generator[1]
        final_conv = self.generator[2]
        final_tanh = self.generator[3]

        encoder_outputs = [] # will ignore the last encoder_output
        curr_output = original_images
        for i,encoder_layer in enumerate(encoder):
            curr_output = encoder_layer(curr_output)
            encoder_outputs.append(curr_output)

        for i,decoder_layer in enumerate(decoder):
            decoder_output = decoder_layer(curr_output)
            curr_output = tf.nn.Concatenate()([decoder_output,encoder_outputs[i]])

        logits_real_given_real = self.discriminator(tf.nn.Concatenate()([original_images,curr_output])) # check what concatenating them like this does
        logits_gen_given_gen = self.discriminator(tf.nn.Concatenate()([original_images,real_transformed_images]))

        return logits_real_given_real,logits_gen_given_gen

    def loss_func(self,logits_real_given_real,logits_gen_given_gen):
        prob_real_given_real = tf.math.reduce_mean(tf.math.sigmoid(logits_real_given_real))
        prob_gen_given_gen = 1 - tf.math.reduce_mean(tf.math.sigmoid(logits_gen_given_gen))
        return prob_real_given_real + prob_gen_given_gen


def train(model,original_images,real_transformed_images):
    # repeatedly update discriminator then generator (or vice versa?)
    # going to have to multiply by -1 for one of them since one is maximizing and other is minimizing
    for i in range(0,len(original_images),model.batch_size):
        original_images_batch = original_images[i:i+model.batch_size]
        real_transformed_images_batch = real_transformed_images[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            logits_real_given_real,logits_gen_given_gen = model.call(original_images_batch,real_transformed_images_batch)
            loss = model.loss(logits_real_given_real,logits_gen_given_gen)
        gradients = tape.gradient(loss,model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
