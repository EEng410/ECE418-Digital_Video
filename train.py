import tensorflow as tf
import numpy as np
from SCFpyr_TF import SCFpyr_TF

import os
from phasenet import PhaseNet, Dataloader, CombinedLoss, process_batch, output_convert
import time
import pickle

def save_model(model, name):
    with open(name, "wb") as file: # file is a variable for storing the newly created file, it can be anything.
        pickle.dump(model, file) # Dump function is used to write the object into the created file in byte format.

if __name__ == "__main__":
    # create log
    log_dir = './log/'
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_name = '{}_train.txt'.format(now)

    # define net parameter
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 4
    # pyr parameter
    height = 6
    nbands = 4
    scale_factor = 2**(1/2)

    dataloader = Dataloader("DAVIS/JPEGImages/480p")

    model = PhaseNet()
    combined_loss = CombinedLoss(v = 0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Training loop

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    pyr = SCFpyr_TF(height=height, nbands=nbands, scale_factor=scale_factor)

    loss = CombinedLoss(v=0.1)

    # There are 6028 samples in the dataset
    step = 0
    num_samples = 6028
    for epoch in range(num_epochs):
        # randomly generate a sequence from 0 to num_samples-1, use these in order as a whole epoch
        iter = 0
        rand_seq = tf.random.shuffle(tf.constant(range(num_samples)))
        for loop in range(753):
            x_batch = []
            for i in range(batch_size):
                if iter <= num_samples:
                    x_batch.append(dataloader.get_item(rand_seq[iter]))
                    iter += 1
            for channel in range(3):
                image_stack = []
                batch_coeffs = []
                for n, triplet in enumerate(x_batch):
                    # Reformat triplets into image stack
                    image_stack.append(tf.stack([triplet['start'], triplet['inter'], triplet['end']]))
                    # Each element in image_stack is [3, 256, 256, 3], i.e. [N, H, W, C]
                    # Turn into complex tensor for steerable pyramid
                    img_complex = tf.dtypes.complex(image_stack[n][:, :, :, channel][:, :, :, tf.newaxis], tf.zeros(shape = image_stack[n][:, :, :, channel][:, :, :, tf.newaxis].shape))
                    coeffs = pyr.build(img_complex)
                    batch_coeffs.append(coeffs)
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(model.alpha)
                    train, truth = process_batch(batch_coeffs)
                    predicted_coeff = model(train)
                    # truth_img = triplet['inter'][:, :, channel]
                    # predicted_img = pyr.reconstruct(output_convert(predicted_coeff))
                    loss = combined_loss(truth, predicted_coeff)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                step+=1
                if step % 10 == 0:
                    print('Epoch [{}/{}], Channel [{}], Step [{}], Loss: {:.4f}'
                        .format(epoch+1, num_epochs, channel, step, loss))
                    # save log
                    with open(os.path.join(log_dir, log_name), 'at') as f:
                        f.write('Epoch [{}/{}], Channel [{}], Step [{}], Loss: {:.4f}\n'.format(
                            epoch+1, num_epochs, channel, step, loss))
    breakpoint()
            
            
            
            