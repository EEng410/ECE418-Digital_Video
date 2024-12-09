import tensorflow as tf
from steerablepyramid import SteerablePyramid
import os
from PIL import Image
import sys
import numpy as np
# Create PhaseNet Block

def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_amplitude_phase(input):
    # Assume the input is a tensor in the norm of [batch, 256, 256, 1], Complex numbers
    real = tf.math.real(input)
    imag = tf.math.imag(input)
    
    amplitude = tf.math.sqrt(real**2+imag**2)
    phase = tf.math.atan2(imag, real)

    return amplitude, phase 

# def steerable_pyr_norm(input):
#     for i in range(len(input)):
#         for j in range(len(input[i])):
#             AP = get_amplitude_phase(input[i][j]['s'])

def process_coeff(coeff):
    # Reverse order since we want highest decomposition level first
    coeff = coeff[::-1]
    train = []
    truth = []
    # exception for low level residual
    train.append(tf.stack([tf.math.abs(coeff[0][0, :, :]), tf.math.abs(coeff[0][2, :, :])]))
    truth.append(tf.math.abs(coeff[0][1, :, :][tf.newaxis, :, :]))
    for level in range(1, len(coeff)-1):
        AP = []
        for subband in coeff[level]:
            img = subband
            AP.append(get_amplitude_phase(img))
        # AP comes out as [band][A/P][3, 256, 256]
        # Reshape AP 
        train.append(tf.stack([
            AP[0][0][0,:,:],AP[1][0][0,:,:],
            AP[2][0][0,:,:],AP[3][0][0,:,:],
            AP[0][1][0,:,:],AP[1][1][0,:,:],
            AP[2][1][0,:,:],AP[3][1][0,:,:],
            AP[0][0][2,:,:],AP[1][0][2,:,:],
            AP[2][0][2,:,:],AP[3][0][2,:,:],
            AP[0][1][2,:,:],AP[1][1][2,:,:],
            AP[2][1][2,:,:],AP[3][1][2,:,:]]))
        truth.append(tf.stack([
            AP[0][0][1,:,:],AP[1][0][1,:,:],
            AP[2][0][1,:,:],AP[3][0][1,:,:],
            AP[0][1][1,:,:],AP[1][1][1,:,:],
            AP[2][1][1,:,:],AP[3][1][1,:,:]]))

    return train, truth

def process_batch(batch_coeff):
    # Assume we have an incoming list of length batch_size
    temp = []
    train = []
    truth = []
    for coeff in batch_coeff:
        temp.append(process_coeff(coeff))
    
    for i in range(len(temp[0][0])):
        # Reshape so the batch is now treated as an inner dimension while the outer dimension corresponds to levels of decomposition
        train.append(tf.stack([item[0][i] for item in temp]))
        truth.append(tf.stack([item[1][i] for item in temp]))
    return train, truth

def normalize(coeff_input):
    
    # temp = coeff_input.numpy()
    # # Residual Case
    # if coeff_input.shape[1] == 2:
    #     max = tf.math.reduce_max(coeff_input, axis = [2, 3], keepdims = True).numpy()
    #     temp = temp/max
    # else:
    #     num_bands = int(temp.shape[1]/4)
    #     # normalize phase 
    #     for i in range(num_bands):
    #         temp[:, i+num_bands, :, :] = temp[:, i+num_bands, :, :]/np.pi
    #         temp[:, i+3*num_bands, :, :] = temp[:, i+3*num_bands, :, :]/np.pi
    #     # normalize amplitude 
    #     for i in range(coeff_input.shape[0]):
    #         for j in range(num_bands):
    #             temp[i, j, :, :] = temp[i, j, :, :]/tf.math.reduce_max(coeff_input[i, j, :, :][tf.newaxis, tf.newaxis, :, :], axis = [2, 3], keepdims=True).numpy()
    #             temp[i, j+2*num_bands, :, :] = temp[i, j+2*num_bands, :, :] / tf.math.reduce_max(coeff_input[i, j+2*num_bands, :, :][tf.newaxis, tf.newaxis, :, :], axis = [2, 3], keepdims = True).numpy()
    # temp = tf.convert_to_tensor(temp)
    temp = coeff_input.numpy()
    # Residual Case
    if coeff_input.shape[1] == 2:
        max = tf.math.reduce_max(coeff_input, axis = [2, 3], keepdims = True).numpy()
        temp = temp/max
    else:
        num_bands = int(temp.shape[1]/4)
        # normalize phase 
        for i in range(num_bands):
            temp[:, i+num_bands, :, :] = temp[:, i+num_bands, :, :]/np.pi
            temp[:, i+3*num_bands, :, :] = temp[:, i+3*num_bands, :, :]/np.pi
        # normalize amplitude 
        for i in range(coeff_input.shape[0]):
            for j in range(num_bands):
                temp[i, j, :, :] = temp[i, j, :, :]/tf.math.reduce_max(coeff_input[i, j, :, :][tf.newaxis, tf.newaxis, :, :], axis = [2, 3], keepdims=True).numpy()
                temp[i, j+2*num_bands, :, :] = temp[i, j+2*num_bands, :, :] / tf.math.reduce_max(coeff_input[i, j+2*num_bands, :, :][tf.newaxis, tf.newaxis, :, :], axis = [2, 3], keepdims = True).numpy()
    temp = tf.convert_to_tensor(temp)

    return temp

def output_convert(pre_coeff):
    coeff = []
    bands_num = int(pre_coeff[1].shape[1]/2)
    
    coeff.append(tf.squeeze(pre_coeff[0], axis = 1))
    for i in range(1,len(pre_coeff)):
        band = []
        for j in range(bands_num):
            amp = pre_coeff[i][:,j,:,:]
            phase = pre_coeff[i][:,j+bands_num,:,:]
            real = amp*tf.math.cos(phase)
            imag = amp*tf.math.sin(phase)
            band.append(tf.dtypes.complex(real,imag))
        coeff.insert(0,band)
    coeff.insert(0,tf.zeros(shape=(pre_coeff[-1].shape[0],pre_coeff[-1].shape[2],pre_coeff[-1].shape[3])))
    return coeff

class PhaseNetBlock(tf.Module):
    def __init__(self, in_channels = 81, out_channels = 64, kernel_size = 3, padding = 'same'):
        super(PhaseNetBlock, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_channels, (kernel_size, kernel_size), padding = padding, data_format="channels_first"),
            tf.keras.layers.Conv2D(out_channels, (kernel_size, kernel_size), padding = padding, data_format="channels_first"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha = 0.2)
        ])

    def __call__(self, inputs):
        x = self.layer(inputs)
        return x
    
class PredBlock(tf.Module):
    def __init__(self, in_channels = 64, out_channels = 8, kernel_size = 1, padding = 'same'):
        super(PredBlock, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_channels, (kernel_size, kernel_size), padding = padding, data_format="channels_first"),
            ])
    def __call__(self, inputs):
        x = tf.keras.activations.tanh(self.layer(inputs))
        return x
    
class Dataloader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self.get_classes()
        self.triplets = self.get_triplets()

        self.sizing = tf.keras.Sequential(
            tf.keras.layers.Resizing(256, 256)
        )

    def get_classes(self):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def get_triplets(self):
        # a modified version from torchvision.datasets.ImageFolder source code
        triplets = []
        dir = os.path.expanduser(self.root_dir)
        # Iterate through each subdirectory
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(dir, target)
            # Get a list of all images in subdirectory d
            images = []
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
            for i in range(len(images) - 2):
                triplets.append(images[i:i+3])
        return triplets
    
    def get_item(self, idx):
        # process each triplet into images
        imgs = self.triplets[idx]
        # load image based on stored paths
        sample = [self.sizing(tf.convert_to_tensor(load_image(img[0]))) for img in imgs]
        
        
        sample = {'start': sample[0], 'inter': sample[1],
                  'end': sample[2], 'class_index': imgs[0][1]}
        
        return sample
    



class PhaseNet(tf.Module):
    def __init__(self):
        super(PhaseNet, self).__init__()
        # Two parameters needed to learn a reconstruction of the inter frame
        rng = tf.random.get_global_generator()
        
        self.alpha = tf.Variable(
            rng.normal(shape=[1]), 
            trainable = True
        )
        self.beta = tf.Variable(
            rng.normal(shape=[1]), 
            trainable= True
        )

        # Generate PhaseNet blocks
        self.phase_layers = []
        self.preds = []
        
        # Take two channel input, block it up to 64 channel with kernel size 1 and stride 0
        self.phase_layers.append(PhaseNetBlock(2, 64, 1, 'valid'))
        self.preds.append(PredBlock(64, 1))

        # Use a 1x1 kernel
        self.phase_layers.append(PhaseNetBlock(81, 64, 1, 'valid'))
        self.preds.append(PredBlock())

        for i in range(8):
            self.phase_layers.append(PhaseNetBlock())
            self.preds.append(PredBlock())        

    def __call__(self, x):
        # Depending on the length of x, we will apply the network to only some of the layers

        # We technically only use the first layer for inference!
        feature_map = []
        pred_map = []
        output = []
        feature_map.append(self.phase_layers[0](normalize(x[0])))
        pred_map.append(self.preds[0](feature_map[0]))
        amp = self.alpha*x[0][:, 0, :, :] + (1-self.alpha)*x[0][:, 1, :, :]
        output.append(amp[:, tf.newaxis, :, :])

        # Iterate through each decomposition level
        for i in range(1, len(x)):
            img_dim = (x[i].shape[2], x[i].shape[3])
            feature_map.append(self.phase_layers[i](tf.concat([
                    normalize(x[i]),
                    tf.transpose(tf.image.resize(tf.transpose(feature_map[i-1], perm = [0, 2, 3, 1]), img_dim), perm = [0, 3, 1, 2]), # defaults to bilinear
                    tf.transpose(tf.image.resize(tf.transpose(pred_map[i-1], perm = [0, 2, 3, 1]), img_dim), perm = [0, 3, 1, 2])], axis = 1
                    ))) # concatenate along the 1st axis (0th is batch)
            pred_map.append(self.preds[i](feature_map[i]))
            amp = self.beta*x[i][:, 0:4, :, :] + (1-self.beta)*x[i][:, 8:12, :, :]
            phase = pred_map[i][:,4:8,:,:]    
            output.append(tf.concat([amp, phase], axis = 1))
        return output

class CombinedLoss(tf.Module):
    def __init__(self, v = 0.1):
        super(CombinedLoss, self).__init__()
        self.v = v
    
    def __call__(self, truth_coeff, pred_coeff):
        # img_loss = tf.norm(truth_img - pred_img, ord = 1)
        dphase = [truth_coeff[i][:, 4:, :, :]-pred_coeff[i][:, 4:, :, :] for i in range(len(truth_coeff))]
        differential_phase = [tf.math.atan2(tf.math.sin(d), tf.math.cos(d)) for d in dphase]
        phase_loss = 0
        for i in range(len(differential_phase)):
            phase_loss += tf.norm(differential_phase[i], ord = 1)
        loss = phase_loss
        return loss
        