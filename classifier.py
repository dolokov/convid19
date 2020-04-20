import cv2 as cv 
import numpy as np
import tensorflow as tf
from datetime import datetime
import os 
import json 
import shutil

def load_dataset(config,mode = ["pad","crop"][0]):
    image_dir = os.path.expanduser('~/data/convid/findings')
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    findings_labels, annotation_data = load_annotation_data(config)
    for p in annotation_data.keys():
        print(p)#.keys())
    
    heights,widths = [],[]
    for im_meta in annotation_data['images']:
        heights.append(im_meta['height'])
        widths.append(im_meta['width'])
        path = os.path.expanduser(os.path.join('~/github/convid19/covid-chestxray-dataset/images',im_meta['file_name']))
        finding = im_meta['metadata']['finding']

        new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%s.png'%(str(findings_labels.index(finding)),im_meta['file_name'])))
        if not os.path.isfile(new_path):
            cv.imwrite(new_path,cv.imread(path))
            
            #shutil.copyfile(path,new_path)
        #im = cv.imread(path)
    heights, widths = np.array(heights), np.array(widths)
    print('heights mean/std min/max %i/%i %i/%i' % (heights.mean(),heights.std(),heights.min(), heights.max()))
    print('widths mean/std min/max %i/%i %i/%i' % (widths.mean(),widths.std(),widths.min(), widths.max()))
        

    def load_im(image_file,mode="crop"):
        image = tf.io.read_file(image_file)
        class_id = tf.strings.split(image_file,sep='#$#')[0]
        label = tf.one_hot([class_id],11)
        #print('class_id',class_id)
        image = tf.image.decode_png(image,channels=3)
        if mode == "pad":
            image = tf.image.resize_with_pad(image,config['img_height'],config['img_width'],antialias=True)
        if mode == "crop":
            rr = np.random.uniform(1.01,3.)
            image = tf.image.resize(image, (int(rr*config['img_height']),int(rr*config['img_width'])), preserve_aspect_ratio=True,antialias=True)
            image = tf.image.random_crop(image,(config['img_height'],config['img_width'],3))
        image = tf.cast(image,tf.float32)
        image = (image / 127.5) - 1
        return image, label     
    
    data = tf.data.Dataset.list_files(os.path.join(image_dir,'*.png'))
    # https://github.com/tensorflow/tensorflow/issues/32052 data = data.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    data = data.map(load_im)
    data = data.shuffle(1024)
    #data = data.repeat()
    data = data.batch(config['batch_size'])#.prefetch(tf.data.experimental.AUTOTUNE)#.cache()
    print('[*] loaded images from disk')
    return findings_labels, data, [] 


def load_annotation_data(config):
    file_annotations = "covid-chestxray-dataset/annotations/imageannotation_ai_lung_bounding_boxes.json"
    with open(file_annotations, 'r') as f:
        data = json.load(f)
    
    count_findings, all_findings = {},[]
    for i,p in enumerate(data['annotations']):
        if 'attributes' in p and 'X-Ray' in p['attributes']['Modality'] and 'Finding' in p['attributes']:
            findings = p['attributes']['Finding']
            findings = '+'.join(findings)
            if not findings in all_findings:
                all_findings.append(findings)
            if not findings in count_findings:
                count_findings[findings] = 0
            count_findings[findings] += 1
    count_findings_ims, all_findings_ims = {},[]
    for i,p in enumerate(data['images']):
        findings = p['metadata']['finding']
        #findings = p['attributes']['Finding']
        #findings = '+'.join(findings)
        if not findings in all_findings_ims:
            all_findings_ims.append(findings)
        if not findings in count_findings_ims:
            count_findings_ims[findings] = 0
        count_findings_ims[findings] += 1
    print('Findings',all_findings)
    print('annotations:')
    for k,v in count_findings.items():
        print(k,'\t',v)
    print('- - - - - - - - - ')
    print('images')
    for k,v in count_findings_ims.items():
        print(k,'\t',v)
    return all_findings_ims, data
    

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

def make_model(config, metrics = METRICS):
    def downsample(filters,size = 3,strides=2, dilation_rate=(1, 1), apply_norm=True, dropout_rate = None, apply_lrelu = True):
        initializer = 'he_normal'
        #initializer = ["he_normal","orthogonal", tf.random_normal_initializer(0., init_std)][0]

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides,
                                        padding='same',
                                        #kernel_constraint=wconst,
                                        dilation_rate=dilation_rate,
                                        kernel_initializer=initializer))#,
                                        #use_bias=False))  
        if dropout_rate is not None and dropout_rate > 0.0:
            result.add(tf.keras.layers.Dropout(dropout_rate))
        if apply_norm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_lrelu:
            result.add(tf.keras.layers.LeakyReLU())
        return result

    inp = tf.keras.layers.Input(shape=[config['img_height'],config['img_width'], 3], name='input_image')

    x = inp
    cl = 0  
    while x.get_shape().as_list()[1] > 8:
        fs = 32 * 2**cl
        x = downsample(fs)(x)
        cl += 1 

    x = downsample(config['num_classes'])(x)
    x = tf.keras.layers.AveragePooling2D(x.get_shape().as_list()[1:3])(x)
    
    model = tf.keras.Model(inputs = inp,outputs = x, name = 'Classifier')
    '''model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=config['lr']),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics)'''

    model.summary()

    return model

def train_classifier(config):
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/convid19/classification/%s" % now)
    #if not os.path.isdir(checkpoint_path):
    #    os.makedirs(checkpoint_path)

    labels, data_train, data_val = load_dataset(config)
    
    model = make_model(config)
    
if __name__ == "__main__":
    config = {
        'num_classes': 11,
        "lr": 1e-4,
        "batch_size": 4,
        "img_width": 256,
        "img_height": 256           
    }
    train_classifier(config)