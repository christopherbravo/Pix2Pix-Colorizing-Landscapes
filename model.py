from preprocess import get_data
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class Generator(tf.keras.Model):
    pass

class Model(tf.keras.Model):
    def __init__(self,num_output_channels):
        super().__init__()
        self.batch_size = 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.999)
        self.lambda_param = 100 # regularization

        def create_layer_with_batch_norm_and_relu(num_filters,kernel_size,batch_norm=True,dropout=False,downsample=True,input_shape=None):
            layer = tf.keras.Sequential()

            if downsample:
                if input_shape:
                    conv_output_layer = tf.keras.layers.Conv2D(num_filters,kernel_size,strides=2, padding='same',input_shape=input_shape)
                else:
                    conv_output_layer = tf.keras.layers.Conv2D(num_filters,kernel_size,strides=2, padding='same')
            else:
                if input_shape:
                    conv_output_layer = tf.keras.layers.Conv2DTranspose(num_filters,kernel_size,strides=2,padding='same',input_shape=input_shape)
                else:
                    conv_output_layer = tf.keras.layers.Conv2DTranspose(num_filters,kernel_size,strides=2,padding='same')
            layer.add(conv_output_layer)

            if batch_norm:
                batch_norm_layer = tf.keras.layers.BatchNormalization()
                layer.add(batch_norm_layer)

            if dropout:
                layer.add(tf.keras.layers.Dropout(0.5))

            leaky_relu_layer = tf.keras.layers.LeakyReLU(0.2)
            layer.add(leaky_relu_layer)

            return layer


        ### TODO add skip connections (U Net). also modify decoder to be consistent with unet. section 6.1.1
        self.generator = []

        encoder = []
        encoder.append(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False,dropout=False,input_shape=(256,256,3)))
        encoder.append(create_layer_with_batch_norm_and_relu(128,4))
        encoder.append(create_layer_with_batch_norm_and_relu(256,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        encoder.append(create_layer_with_batch_norm_and_relu(512,4))
        decoder = []
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,dropout=True,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(512,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(256,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(128,4,downsample=False))
        decoder.append(create_layer_with_batch_norm_and_relu(64,4,downsample=False))

        self.generator.append(encoder)
        self.generator.append(decoder)
        self.generator.append(tf.keras.layers.Conv2DTranspose(num_output_channels,4,strides=2,padding='same'))
        self.generator.append(tf.keras.layers.Activation('tanh'))


        self.discriminator = tf.keras.Sequential()
        self.discriminator.add(create_layer_with_batch_norm_and_relu(64,4,batch_norm=False,input_shape=(256,256,6)))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(128,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(256,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(512,4,batch_norm=False))
        self.discriminator.add(create_layer_with_batch_norm_and_relu(1,4,batch_norm=False))
        # self.discriminator.add(tf.keras.layers.Activation('sigmoid'))

    def generate(self,original_images):
        encoder = self.generator[0]
        decoder = self.generator[1]
        final_conv = self.generator[2]
        final_tanh = self.generator[3]

        encoder_outputs = [] # will ignore the last encoder_output
        curr_output = original_images

        for i,encoder_layer in enumerate(encoder):
            curr_output = encoder_layer(curr_output)
            encoder_outputs.append(curr_output)

        encoder_outputs = encoder_outputs[:-1]

        for i,decoder_layer in enumerate(decoder):
            decoder_output = decoder_layer(curr_output)
            curr_output = tf.keras.layers.Concatenate()([decoder_output,encoder_outputs[-(i+1)]])

        curr_output = final_conv(curr_output)
        curr_output = final_tanh(curr_output)
        return curr_output

    def discriminate(self,original_images,real_transformed_images):
        pass

    def call(self,original_images,real_transformed_images):
        # call the generator
        curr_output = self.generate(original_images)

        original_and_real_transformed = tf.keras.layers.Concatenate()([original_images[0],real_transformed_images[0]])
        logits_real_given_real = self.discriminator(original_and_real_transformed) # check what concatenating them like this does
        original_and_generated_transformed = tf.keras.layers.Concatenate()([original_images[0],curr_output])
        logits_real_given_gen = self.discriminator(original_and_generated_transformed)

        return logits_real_given_real,logits_real_given_gen,curr_output

    def discriminator_loss(self,probs_real_given_real,probs_real_given_gen):
        # off by negative sign. WHY?
        log_prob_real_given_real = -1 * tf.math.reduce_mean(tf.math.log(probs_real_given_real))
        log_prob_gen_given_gen = -1 * tf.math.reduce_mean(tf.math.log(1 - probs_real_given_gen))
        my_discriminator_loss = (log_prob_real_given_real + log_prob_gen_given_gen)

        # log_prob_real_given_real = tf.math.reduce_mean(tf.math.log(tf.math.sigmoid(logits_real_given_real)))
        # probs_gen_given_gen = tf.math.sigmoid(logits_gen_given_gen)
        # log_prob_gen_given_gen = tf.math.reduce_mean(tf.math.log(1 - probs_gen_given_gen))
        # return -1 * (log_prob_real_given_real + log_prob_gen_given_gen) / 2

        # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # real_loss = loss_object(tf.ones_like(probs_real_given_real), probs_real_given_real)
        # generated_loss = loss_object(tf.zeros_like(probs_real_given_gen), probs_real_given_gen)
        # total_disc_loss = real_loss + generated_loss
        # print('DISCRIMINATIVE')
        # print(f'real: {log_prob_real_given_real} {real_loss}')
        # print(f'gen: {log_prob_gen_given_gen} {generated_loss}')
        # print(f'discriminator total: {my_discriminator_loss} {total_disc_loss}')
        return my_discriminator_loss / 2 # need to divide by 2 for training

    def generator_loss(self,probs_real_given_gen,y,generated):
        log_prob_real_given_gen = -1 * tf.math.reduce_mean(tf.math.log(probs_real_given_gen))

        regularization = self.lambda_param * tf.norm(tf.reshape(y,[-1]) - tf.reshape(generated,[-1]),ord=1)/tf.cast(tf.math.reduce_prod(generated.shape[1:]),tf.float32)

        my_generator_loss = log_prob_real_given_gen + regularization
        #return my_generator_loss

        # probs_real_given_gen = tf.math.sigmoid(logits_real_given_gen)
        # log_prob_gen_given_gen = tf.math.reduce_mean(tf.math.log(1 - probs_real_given_gen))
        # # tf tutorial uses l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) instead of tf.norm(y - generated,ord=1)
        # regularization = self.lambda_param * tf.norm(y - generated,ord=1) # idk if this will work with batch_size > 1
        # return log_prob_gen_given_gen + regularization

        # loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # gan_loss = loss_object(tf.ones_like(probs_real_given_gen), probs_real_given_gen)
        # l1_loss = self.lambda_param * tf.reduce_mean(tf.abs(y - generated))
        # total_gen_loss = gan_loss + l1_loss
        #
        # print('GENERATIVE')
        # print(f'gen: {log_prob_real_given_gen} {gan_loss}')
        # print(f'regularization: {regularization} {l1_loss}')
        # print(f'generative total: {my_generator_loss} {total_gen_loss}')
        return my_generator_loss


def train(model,original_images,real_transformed_images):
    # repeatedly update discriminator then generator (or vice versa?)
    # going to have to multiply by -1 for one of them since one is maximizing and other is minimizing
    print(len(original_images))
    for i in range(0,len(original_images),model.batch_size):
        print(i)
        original_images_batch = original_images[i:i+model.batch_size]
        real_transformed_images_batch = real_transformed_images[i:i+model.batch_size]
        with tf.GradientTape() as discriminator_tape,tf.GradientTape() as generator_tape:
            probs_real_given_real,probs_real_given_gen,generated = model.call(original_images_batch,real_transformed_images_batch)

            disc_loss = model.discriminator_loss(probs_real_given_real,probs_real_given_gen)
            gen_loss = model.generator_loss(probs_real_given_gen,real_transformed_images_batch,generated)


        encoder_vars = [sequential.trainable_variables for sequential in model.generator[0]]
        encoder_vars = [var for sequential in encoder_vars for var in sequential]
        decoder_vars = [sequential.trainable_variables for sequential in model.generator[1]]
        decoder_vars = [var for sequential in decoder_vars for var in sequential]
        generator_vars = []
        generator_vars.extend(encoder_vars)
        generator_vars.extend(decoder_vars)
        generator_vars.extend(model.generator[2].trainable_variables)
        generator_vars.extend(model.generator[3].trainable_variables)
        gradients_generator = generator_tape.gradient(gen_loss,generator_vars)
        model.optimizer.apply_gradients(zip(gradients_generator,generator_vars))

        gradients_discriminator = discriminator_tape.gradient(disc_loss,model.discriminator.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients_discriminator,model.discriminator.trainable_variables))


def main():
    train_data,test_data = get_data('./data/facades/')
    real_transformed_images,original_images = zip(*train_data)
    original_images = list(original_images)
    real_transformed_images = list(real_transformed_images)

    model = Model(3)
    train(model,original_images[:50],real_transformed_images[:50])

    def generate_images(model, test_input, tar):
      prediction = model.generate(test_input)
      plt.figure(figsize=(15, 15))

      display_list = [test_input[0], tar[0], prediction[0]]
      title = ['Input Image', 'Ground Truth', 'Predicted Image']
      for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
      plt.show()

    for example_target, example_input  in test_data.take(3):
        generate_images(model, example_input, example_target)


if __name__ == '__main__':
    main()
