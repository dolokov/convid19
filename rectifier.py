"""
    learn to cut and rectify a xray image by self supervision
    

    take random image or monochrome background and paste a random affine transformation
    model is an encoder predicting the random parameters
    
"""

import os 
import sys 
from random import shuffle  
from datetime import datetime 
import time 

import numpy as np 
import cv2 as cv 
import tensorflow as tf 
import tensorflow_addons as tfa 

from classifier import downsample, dense

def create_background_dataset(config):
    print('[*] create_background_dataset',config)
    url = "http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz"
    fraw = '/tmp/%s' % url.split('/')[-1]
    
    if not os.path.isfile(fraw):
        import subprocess
        subprocess.call(['wget',url,'-O',fraw])
    import tarfile
    tar = tarfile.open(fraw, "r:gz")
    print('[*] extracting SUN2012 dataset ...')
    tar.extractall(config['background_dir'])
    tar.close()

def load_dataset(config):
    ## load background and image folders to combine them at runtime
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    datasets = {'train':{},'test':{}}
    
    '''bgtrain_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range = 9,
        horizontal_flip=True)
    bgtest_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    bgtrain_generator = bgtrain_datagen.flow_from_directory(
            '%s/SUN2012pascalformat' % config['background_dir'],
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'])
    bgvalidation_generator = bgtest_datagen.flow_from_directory(
            '%s/SUN2012pascalformat' % (config['background_dir']),
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'])
    bgtrain_generator = tf.data.Dataset.from_generator( 
     bgtrain_generator, 
     (tf.float32, tf.int64), 
     (tf.TensorShape([None,config['img_height'],config['img_width'],3]), tf.TensorShape([None]))) 
    datasets['train']['background'] = bgtrain_generator
    datasets['test']['background'] = bgvalidation_generator'''

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range = 9,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(
            '%s/train' % config['image_dir'],
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'],
            class_mode = None)
    
    #train_generator = tf.data.Dataset.from_generator( 
    #  train_generator, 
    #  tf.float32) 
    datasets['train']['xray'] = train_generator
    

    #test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    '''validation_generator = test_datagen.flow_from_directory(
            '%s/test' % (config['image_dir']),
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'])'''
    
    #datasets['test']['xray'] = validation_generator

    return datasets 

def make_model(config):
    inp = tf.keras.layers.Input(shape=[config['img_height'],config['img_width'], 3], name='input_image')
    
    x = inp
    reg_l2 = 0.0#1
    kernel_initializer=["he_normal","glorot_uniform"][1]
    cl = 0  
    layers = []
    x = downsample(32,size=3,initializer = kernel_initializer)(x)
    x = downsample(64,size=3,initializer = kernel_initializer)(x)
    while x.get_shape().as_list()[1] > 4:
        fs = min(512, 128 * 2**cl)
        xx = tf.keras.layers.Conv2D(fs, 1, strides=1, padding='same',kernel_regularizer=tf.keras.regularizers.l2(reg_l2),kernel_initializer=kernel_initializer)(x) 

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(fs, 3, strides=1, padding='same',kernel_regularizer=tf.keras.regularizers.l2(reg_l2),kernel_initializer=kernel_initializer)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(fs, 3, strides=1, padding='same',kernel_regularizer=tf.keras.regularizers.l2(reg_l2),kernel_initializer=kernel_initializer)(x)
        x = xx + x
        x = tf.keras.layers.MaxPool2D()(x)
        layers.append(x)
        cl += 1 
    x = tf.keras.layers.Flatten()(x)
    x = dense(512,dropout_rate=0.5)(x)
    x = dense(8,apply_lrelu=False,apply_norm=False,apply_softmax=False)(x)
    model = tf.keras.Model(inputs = inp,outputs = x, name = 'Resnet Rectifier')

    model.summary()

    return model

def get_transformed_points(im, transformation_matrix):
    points = np.array(im.shape[0] * [
        [0,0],
        [im.shape[2],0],
        [im.shape[2],im.shape[1]],
        [0,im.shape[1]]
    ]).reshape((im.shape[0],4,2))
    # eucl -> homo
    trans_points = tf.concat((points,tf.ones((im.shape[0],4,1))),axis=2)
    # transform points
    trans_points = tf.matmul(transformation_matrix, trans_points,transpose_b=True)
    trans_points = tf.transpose(trans_points,[0,2,1])
    # norm and homo->euclid
    trans_points = trans_points / tf.expand_dims(trans_points[:,:,2],axis=-1)
    trans_points = trans_points[:,:,:2]
    return trans_points

def random_transform_im(im,M=None):
    is_batched = True 
    if len(im.shape) == 3:
        im = tf.expand_dims(im, axis=0)
        is_batched = False 
    ss = [im.shape[0],1]
    if M is None:
        scale = (tf.random.uniform(ss)+0.5 )*2.5
        theta = tf.random.uniform(ss)-0.5
        tx = -(tf.random.uniform(ss))*im.shape[2]
        ty = -(tf.random.uniform(ss))*im.shape[1]
        shearx = tf.random.uniform(ss)+0.5
        sheary = tf.random.uniform(ss)+0.5

        if 0:
            # identity transform
            theta = tf.zeros(ss)
            scale = tf.ones(ss)
            tx = tf.zeros(ss)
            ty = tf.zeros(ss)
            shearx = tf.ones(ss)
            sheary = tf.ones(ss)

        ### Projective transform matrix/matrices. A vector of length 8 or tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
        # then it maps the output point (x, y) to a transformed input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), where k = c0 x + c1 y + 1
        a0 = tf.cos(theta) * scale 
        a1 = tf.sin(theta) * shearx 
        a2 = tx 
        b0 =-tf.sin(theta) * sheary
        b1 = tf.cos(theta) * scale 
        b2 = ty 
        c0 = tf.zeros(ss)#(tf.random.uniform(ss)-0.5)/4. 
        c1 = tf.zeros(ss)#(tf.random.uniform(ss)-0.5)/4. 
        
        M = tf.stack((a0,a1,a2,b0,b1,b2,c0,c1),axis=1)[:,:,0]
        #print('M',M.shape,tf.ones(im.shape[0],1).shape)
        #print(M)
    affine_im = tfa.image.transform(im, M)

    # transform matrix inversion
    trans = tf.concat([M,tf.reshape([tf.ones(im.shape[0])],(im.shape[0],1))],axis=1)
    trans = tf.reshape(trans,(im.shape[0],3,3))
    inv_trans = tf.linalg.inv(trans) # tfaddons does inversion internally
    inv_trans_flat = tf.reshape(inv_trans,(im.shape[0],9))
    inv_trans_flat = inv_trans_flat[:,:8]

    # calculate 4 corner points 
    points = get_transformed_points(im, trans)
    # todo: visualize points to verify position
    print('im',im.shape,'aff',affine_im.shape,'points',points)

    if not is_batched:
        affine_im = affine_im[0,:,:,:]
    return affine_im, inv_trans_flat

def train_rectifier(config):
    print('[*] train_rectifier start with config',config)

    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/convid19/rectifier/%s" % now)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    datasets = load_dataset(config)
    
    model = make_model(config)
    optimizer=tf.keras.optimizers.Adam(lr=config['lr'])

    step = 0 
    writer = tf.summary.create_file_writer(checkpoint_path+'/train')
    writer_test = tf.summary.create_file_writer(checkpoint_path+'/test')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    last_ckpt_time = time.time()
    last_summary_step = 0
    
    moving_loss, moving_accuracy, k = 1., 0., .95
    def train_step(im,bg, last_ckpt_time, should_summarize=True):

        affine_im, inv_trans = random_transform_im(im)
        with tf.GradientTape(persistent=True) as tape:
            predicted = model(affine_im)#,training=True)
            loss = tf.norm(inv_trans - predicted)
            #moving_loss = k * moving_loss + (1-k) * loss 
        
        if step < 50:
            print('loss %i: %f' %(step,loss.numpy()))
        # summary
        if step % 50 == 0 and should_summarize:
            with writer.as_default():
                tf.summary.scalar("loss",loss,step=step)
                tf.summary.flush(writer)
                writer.flush()
                
        if step % 100 == 0 and should_summarize:
            recon_im = random_transform_im(affine_im,predicted)[0]
            vis = tf.concat((im,affine_im, recon_im),axis=2)
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.image('reconstructions',vis,step=step)
            
            print(step,'loss',loss.numpy(),'truth',tuple(inv_trans[0].numpy()),'<===>',tuple(predicted[0].numpy()))

        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))


        if time.time() - last_ckpt_time > 60. * 30.: # each hour #epoch % 100 == 0:
            ckpt_save_path = ckpt_manager.save()
            last_ckpt_time = time.time()
            model.save(os.path.join(checkpoint_path,'classifier.h5'))
            print('[*] saved model to %s.' % ckpt_save_path)
        return last_ckpt_time

    while step < config['max_steps']:
        for im in datasets['train']['xray']:#tf.data.Dataset.zip((datasets['train']['xray'],datasets['train']['background'])):
            bg= None 
            for _ in range(15):
                last_ckpt_time = train_step(im,bg,last_ckpt_time, should_summarize= 1)
                step += 1 

def show_examples(config):
    # load some images, show the input, the warped and the unwarped image
    example_urls = ["https://prod-images-static.radiopaedia.org/images/30134883/94d0d52b219b0cae507d3bbe19d470_jumbo.jpeg",
        "https://i1.wp.com/radiologykey.com/wp-content/uploads/2019/03/f003-002a-9781455774838.jpg?w=960",
        "https://prod-images-static.radiopaedia.org/images/381278/5757c0787c96e99ee6f7a3cdb48ed1_jumbo.jpeg"]
    import subprocess
    files = []
    for i, url in enumerate(example_urls):
        files.append('/tmp/%i.%s'%(i,url.split('.')[-1]))
        if not os.path.isfile(files[-1]):
            subprocess.call(['wget',url,'-O',files[-1]])  
    for i,f in enumerate(files):
        im = cv.imread(f)
        #print(f,im.shape)  
        for j in range(10):
            affine_im, inv_trans = random_transform_im(im)
            #print('affine_im',affine_im.shape,affine_im.numpy().min(),affine_im.numpy().max())
            recon_im = random_transform_im(affine_im,inv_trans)[0]
            print(im.shape,affine_im.shape, recon_im.shape)
            vis = np.hstack((im,affine_im, recon_im))
            fout = '/tmp/out_%i-%i.%s' % ( i,j, f.split('.')[-1])
            cv.imwrite(fout, vis )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',default='~/data/convid/findings')
    parser.add_argument('--background_dir',default='~/data/convid/backgrounds')
    parser.add_argument('--predict',default = None)
    parser.add_argument('--model', default=None, help = "model file.h5")
    args = parser.parse_args()

    image_dir = os.path.expanduser(args.image_dir)
    background_dir = os.path.expanduser(args.background_dir)

    k=4
    config = {
        'num_classes': 3,
        "lr": 1e-5,
        "batch_size": 2,
        'epochs': 1000,
        "img_width": k*128,
        "img_height": k*128,
        'max_steps': int(1e6),
        'image_dir': image_dir,
        'background_dir': background_dir
    }
    show_examples(config)
    # predict
    if args.predict is not None:
        config['resample_test'] = True 
        config['batch_size'] = 8
        #predict(config, args.model, args.predict)
        
    else:
        # train 
        # create dataset if not there
        if not os.path.isdir(config['background_dir']):
            os.makedirs(config['background_dir'])
            create_background_dataset(config)
        train_rectifier(config)