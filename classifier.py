import cv2 as cv 
import numpy as np
import tensorflow as tf
from datetime import datetime
import os 
import json 
import shutil
import time 
from glob import glob


from tensorflow.keras.mixed_precision import experimental as mixed_precision
dtype = tf.float32
if 0:
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  print('Compute dtype: %s' % policy.compute_dtype)
  print('Variable dtype: %s' % policy.variable_dtype)
  dtype = tf.bfloat16

tf.summary.trace_on(
    graph=True,
    profiler=False
)

#tf.keras.backend.set_floatx('float16')

def write_im(ddata):
    np.random.seed()
    path, new_path, img_height,img_width, label = ddata.split('/$%#/')

    # extend new path to put into subdirectory
    if int(label) == 0:
        newlabel = "0"
    elif int(label) == 3:
        newlabel = "1"
    elif int(label) == 4:
        newlabel = "2"

    new_dir, new_fn = os.path.split(new_path)
    mode = "train"
    if np.random.random() < 0.1:
        mode = "test"
    new_path = os.path.join(new_dir,mode,newlabel,new_fn)

    im = cv.imread(path)
    ddd = max(int(img_height),int(img_width))
    ss = .75
    if im.shape[0] > ddd /ss or im.shape[1] > ddd /ss: # file image too big
        while im.shape[0] > ddd /ss or im.shape[1] > ddd /ss:
            im = cv.resize(im,None,None,fx=ss,fy=ss)
    
    if im.shape[0] < ddd or im.shape[1] < ddd: # file image too small
        while im.shape[0] < ddd or im.shape[1] < ddd:
            im = cv.resize(im,None,None,fx=1.5,fy=1.5)
        
    #if not (int(label) == 4 and np.random.uniform() > 0.5):
    cv.imwrite(new_path,im)
    
    # h/vflip
    #cv.imwrite(new_path.replace('.png','h.png'),np.fliplr(im))
    #cv.imwrite(new_path.replace('.png','v.png'),np.flipud(im))
    #cv.imwrite(new_path.replace('.png','hv.png'),np.flipud(np.fliplr(im)))
    
    ## resample covid images 
    if 0 and int(label) == 0:
        for j in range(80):
            random_im = im + np.random.normal(0,10,np.prod(im.shape)).reshape(im.shape)
            new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%i#%s.png'%(str(label),int(1e5*np.random.uniform()),os.path.split(path)[1])))
            cv.imwrite(new_path,im)

    
def create_dataset(image_dir, config, findings_labels, annotation_data):
    use_chexpert = 1 
    use_kaggle = 1
    use_coviddata = 1
    use_dummydata = False
    use_catsvsdogs = 0
    counts = []
    for p in findings_labels:
        counts.append(0)

    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    
    writeim_jobs = []
    ## cheXpert
    if use_chexpert:
        print('[*] creating dataset from cheXpert data')
        directory_chexpert = os.path.expanduser("~/data/convid/chexpert")
        train_csv = os.path.join(directory_chexpert,"CheXpert-v1.0-small/train.csv")
        # Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,
        # Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
        with open(train_csv,'r') as f:
            lines = f.readlines()
            i = -1
            while i < len(lines) -1:
                i += 1
                ss = lines[i]
                if i > 0:
                    ss = ss.replace('\n','')
                    ssl = ss.split(',')
                    path = os.path.join(directory_chexpert, ssl[0])
                    label = -1
                    if ssl[5] == "1.0": # No Finding
                        label = 4
                    elif ssl[12] == "1.0": # Pneumo
                        label = 3
                    if label > 0 and 'frontal' in path: # only no finding, pneumo on frontal images
                        #print('huhu',label,path,":::",ssl)
                        new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%i#%s.png'%(str(label),int(1e5*np.random.uniform()),os.path.split(path)[1])))
                        counts[label] += 1
                        if not os.path.isfile(new_path):
                            writeim_jobs.append('/$%#/'.join([path, new_path, str(config['img_height']),str(config['img_width']),str(label) ] ))
                            '''im = cv.imread(path)
                            ddd = max(config['img_height'],config['img_width'])
                            if im.shape[0] < ddd or im.shape[1] < ddd:
                                im = cv.resize(im,None,None,fx=2.,fy=2.)
                            cv.imwrite(new_path,im)
                            # h/vflip
                            #cv.imwrite(new_path.replace('.png','h.png'),np.fliplr(im))
                            #cv.imwrite(new_path.replace('.png','v.png'),np.flipud(im))
                            #cv.imwrite(new_path.replace('.png','hv.png'),np.flipud(np.fliplr(im)))
                            '''
                            
                    if i % 1000 == 0:
                        print('[*] create dataset chexpert %i/%i'%(i,len(lines)))

    ## </ cheXpert>

    if use_coviddata:
        print('[*] creating dataset from covid data')
        heights,widths = [],[]
        for im_meta in annotation_data['images']:
            heights.append(im_meta['height'])
            widths.append(im_meta['width'])
            path = os.path.expanduser(os.path.join('~/github/convid19/covid-chestxray-dataset/images',im_meta['file_name']))
            finding = im_meta['metadata']['finding']
            label = findings_labels.index(finding)
            counts[label] += 1
            ## only take covid, normal or pneuno images!
            if label == 0 or label == 3 or label == 4:
                new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%s.png'%(str(label),im_meta['file_name'])))
                if not os.path.isfile(new_path):
                    writeim_jobs.append('/$%#/'.join([path, new_path, str(config['img_height']),str(config['img_width']),str(label) ] ))
                            
        heights, widths = np.array(heights), np.array(widths)
        print('heights mean/std min/max %i/%i %i/%i' % (heights.mean(),heights.std(),heights.min(), heights.max()))
        print('widths mean/std min/max %i/%i %i/%i' % (widths.mean(),widths.std(),widths.min(), widths.max()))
            

    if use_kaggle:
        print('[*] creating dataset from kaggle data')
        # now add kaggle chest xray dataset https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
        directory_kaggle_chestxray = os.path.expanduser('~/data/convid/kaggle/chest_xray')
        # in folder NORMAL are images with label 
        for subdir,label in [('NORMAL',4),('PNEUMONIA',3)]:
            for mode in ['train','test']:
                _files = glob(os.path.join(directory_kaggle_chestxray,mode,subdir,'*.jpeg'))
                for path in _files:
                    new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%s.png'%(str(label),os.path.split(path)[1])))
                    if not os.path.isfile(new_path):
                        writeim_jobs.append('/$%#/'.join([path, new_path, str(config['img_height']),str(config['img_width']),str(label)]))        
                    counts[label] += 1
                
    if use_dummydata:
        # make 50 random mono rgb images with a bit of noise for each label
        
        for label in [0,3,4]:
            color = [(0,0,255),0,0,(0,255,0),(255,0,0)][label]
            for n in range(150):
                im = np.uint8(color * np.ones((700,700,3)))
                path = '/tmp/%i.png' % int(1e6 * np.random.random())
                cv.imwrite(path,im)
                random_im = im + np.random.normal(0,5,np.prod(im.shape)).reshape(im.shape)
                new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%i#%s.png'%(str(label),int(1e5*np.random.uniform()),os.path.split(path)[1])))
                writeim_jobs.append('/$%#/'.join([path, new_path, str(config['img_height']),str(config['img_width']),str(label)]))            
        
    if use_catsvsdogs: # https://www.kaggle.com/c/dogs-vs-cats/data
        _dir = os.path.expanduser("~/data/convid/catsvsdogs/train")
        for path in glob(os.path.join(_dir,'*.jpg')):
            ff = path.split('/')[-1]
            label = 0
            if 'cat' in ff:
                label = 1
            new_path = os.path.expanduser(os.path.join(image_dir,'%s#$#%i#%s.png'%(str(label),int(1e5*np.random.uniform()),os.path.split(path)[1])))
            writeim_jobs.append('/$%#/'.join([path, new_path, str(config['img_height']),str(config['img_width']),str(label)]))
    
    # do actual copy jobs
    if 1:
        pool.map(write_im, writeim_jobs)
    else:
        from write_tfrecord import write_files
        write_files(writeim_jobs, config)

    print('[*] counts of labels')
    for label_name, count in zip(findings_labels,counts):
        print(label_name,':',count)

def count_dataset(image_dir):
    files = glob(os.path.join(image_dir,'*.png'))
    counts = {}
    for f in files:
        label, rest = f.split('#$#')
        label = label.split('/')[-1]
        if not label in counts:
            counts[label]=0
        else:
            counts[label]+=1
    print('[*] counts:')
    for k, v in counts.items():
        print(k,v)

def vis_gradcam(grad_model, im, label, classidx ):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(im)
        loss = predictions[:, classidx]

    output = conv_outputs
    grads = tape.gradient(loss, conv_outputs)

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(1, 2))

    
    cam = []
    for b in range(output.shape[0]):
        ca = tf.ones(output.shape[1: 3], tf.float32)
        for i, w in enumerate(weights[b]):
            #print('ABC',b,i,w,'weights',len(weights))
            #rint('shapes',ca.shape, output.shape)
            ca += w * output[b, :, :, i]
        cam.append(ca)
    cam = tf.convert_to_tensor(cam)
    #cam = tf.nn.relu(cam)
    #rint('cam shape',cam.shape)
    heatmap = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))
    heatmap = tf.transpose(heatmap,[2,1,0])
    heatmap = tf.image.resize(heatmap,im.shape[1:3])
    heatmap = tf.transpose(heatmap,[2,1,0])
    heatmap = tf.expand_dims(heatmap,3)
    green = tf.convert_to_tensor([[[1.,0,0]]])
    red = tf.convert_to_tensor([[[0,0,1.]]])
    heatmap = heatmap * green + (1.-heatmap) * red  

    #vis = .5 * (im+1)*.5 + heatmap * .5
    #vis = np.float16(0.5) * (im+np.float16(1))*np.float16(0.5) + np.float16(0.5) * heatmap
    
    # guided grad cam
    guided_grad_cam = guided_grads
    guided_grad_cam = tf.reduce_mean(guided_grad_cam,[3])
    
    #guided_grad_cam = guided_grad_cam * (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam))
    #guided_grad_cam = (guided_grad_cam - tf.reduce_min(guided_grad_cam)) / (tf.reduce_max(guided_grad_cam) - tf.reduce_min(guided_grad_cam))
    guided_grad_cam = guided_grad_cam * (cam - tf.reduce_min(cam,axis=[1,2],keepdims=True)) / (tf.reduce_max(cam,axis=[1,2],keepdims=True) - tf.reduce_min(cam,axis=[1,2],keepdims=True))
    guided_grad_cam = (guided_grad_cam - tf.reduce_min(guided_grad_cam,axis=[1,2],keepdims=True)) / (tf.reduce_max(guided_grad_cam,axis=[1,2],keepdims=True) - tf.reduce_min(guided_grad_cam,axis=[1,2],keepdims=True))

    guided_grad_cam = tf.expand_dims(guided_grad_cam,3)
    guided_grad_cam = tf.image.resize(guided_grad_cam,im.shape[1:3])
    
    guided_grad_cam = guided_grad_cam * green + (1.-guided_grad_cam) * red
    vis = .5 * (im+1)*.5 + guided_grad_cam * .5
    sstacked = tf.concat(((im+1)*np.float16(0.5),heatmap,guided_grad_cam,vis),1)    
    return sstacked  

def read_im(path):
    np.random.seed()
    h, w = 512,512
    #[np.float32(cv.resize(cv.imread(p),(config['img_width'],config['img_height'])))/255. for p in files]
    im = cv.imread(path)
    while im.shape[0] < h or im.shape[1] < w:
        im = cv.resize(im, None, None, fx = 1.1, fy=1.1)
        #print('upscaled',path.split('/')[-1],im.shape)
    y = int(np.random.random() * (im.shape[0]-h))
    x = int(np.random.random() * (im.shape[1]-w))
    im = im[y:y+h,x:x+w,:]

    if 0 and np.random.random() > 0.5:
        ## CLAHE
        lab = cv.cvtColor(im, cv.COLOR_BGR2LAB)
        gridsize= 7
        clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
        lab[...,0] = clahe.apply(lab[...,0])
        im = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    im = np.float16(im)
    im /= 255.
    return im 

    #,(config['img_width'],config['img_height'])))/255.

def read_ims(paths):
    for path in paths:
        yield read_im(path)

def parse_paths(paths):
    for path in paths:
        labels = []
        if '0#$' in f:
            labels.append([1.,0.,0.])
        if '3#$' in f:
            labels.append([0.,1.,0.])
        if '4#$' in f:
            labels.append([0.,0.,1.])
        label = np.array(labels[0])
        yield label

def load_dataset(config,mode = ["pad","crop"][0]):
    #load_datasetFULLRAM(config,mode = mode)
    return load_datasetPUREKERAS(config,mode = mode)
    #return load_datasetTFRECORD(config,mode = mode)

def load_datasetTFRECORD(config,mode = ["pad","crop"][0]):
    #http://digital-thinking.de/tensorflow-vs-keras-or-how-to-speed-up-your-training-for-image-data-sets-by-factor-10/
    findings_labels, annotation_data = load_annotation_data(config)
    record_files = glob(os.path.join(config['image_dir'],'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames = record_files )
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.shuffle(512)
    nrepeat = 8 
    test_dataset = dataset.take(nrepeat*32 // config['batch_size']).repeat() 
    train_dataset = dataset.skip(nrepeat*32 // config['batch_size'])
    return findings_labels, train_dataset, test_dataset
 

def load_datasetPUREKERAS(config,mode = ["pad","crop"][0]):
    findings_labels, annotation_data = load_annotation_data(config)
    
    if 1:
        count_dataset(image_dir)
    
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range = 9,
        horizontal_flip=True)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            '%s/train' % image_dir,
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'])
    validation_generator = test_datagen.flow_from_directory(
            '%s/test' % image_dir,
            target_size=(config['img_height'],config['img_width']),
            batch_size=config['batch_size'])
    
    return findings_labels, train_generator, validation_generator

def load_datasetFULLRAM(config,mode = ["pad","crop"][0]):
    findings_labels, annotation_data = load_annotation_data(config)
    
    if 1:
        count_dataset(image_dir)
    
    files = glob(os.path.join(config['image_dir'],'*.png'))
    from random import shuffle 
    shuffle(files)
    files = files[:min(len(files),3000)]
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    images = pool.map(read_im, files)
    images = np.array(images).astype(np.float32)
    print('iiims',images.shape)
    images = tf.data.Dataset.from_tensor_slices(images)
    print('[*] loaded images')
    labels = []
    for f in files:
        if '0#$' in f:
            labels.append([1.,0.,0.])
        if '3#$' in f:
            labels.append([0.,1.,0.])
        if '4#$' in f:
            labels.append([0.,0.,1.])

    labels = tf.data.Dataset.from_tensor_slices(labels)#[[[0.,1.],[1.,0.]][int('4#$' in p.split('/')[-1])] for p in files])
    print('[*] loaded labels')
    data = tf.data.Dataset.zip((images,labels))
    
    data = data.shuffle(512//4)
    #data = data.repeat()
    data = data.batch(config['batch_size'])#.prefetch(tf.data.experimental.AUTOTUNE)#.cache()
    nrepeat = 8 
    #data = data.repeat(nrepeat)
    print('[*] loaded images from disk')
    test_dataset = data.take(nrepeat*32 // config['batch_size']).repeat() 
    train_dataset = data.skip(nrepeat*32 // config['batch_size'])
    return findings_labels, train_dataset, test_dataset



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
    '''print('Findings',all_findings)
    print('annotations:')
    for k,v in count_findings.items():
        print(k,'\t',v)
    print('- - - - - - - - - ')
    print('images')
    for k,v in count_findings_ims.items():
        print(k,'\t',v)'''
    

    return all_findings_ims, data

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=[False,True][0])

METRICS = [
    tf.keras.metrics.Accuracy(name='accuracy')
]

def make_model(config, metrics = METRICS):
    def downsample(filters,size = 3,strides=2, dilation_rate=(1, 1), apply_norm=True, dropout_rate = 0.2, apply_lrelu = True, apply_softmax = False):
        initializer = 'he_normal'
        #initializer = ["he_normal","orthogonal", tf.random_normal_initializer(0., init_std)][0]

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides,
                                        padding='same',
                                        #kernel_constraint=wconst,
                                        dilation_rate=dilation_rate,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                        activity_regularizer=tf.keras.regularizers.l2(0.01),
                                        kernel_initializer=initializer))#,
                                        #use_bias=False))  
        if dropout_rate is not None and dropout_rate > 0.0:
            result.add(tf.keras.layers.Dropout(dropout_rate))
        if apply_norm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_lrelu:
            result.add(tf.keras.layers.LeakyReLU())
        if apply_softmax:
            result.add(tf.keras.layers.Softmax())
        return result

    def dense(filters,apply_norm=True, dropout_rate = None, apply_lrelu=True,apply_softmax=False):
        initializer = 'he_normal'
        #initializer = ["he_normal","orthogonal", tf.random_normal_initializer(0., init_std)][0]

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Dense(filters, 
                                        #kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                        #activity_regularizer=tf.keras.regularizers.l2(0.01),
                                        kernel_initializer=initializer))#,
                                        #use_bias=False))  
        if dropout_rate is not None and dropout_rate > 0.0:
            result.add(tf.keras.layers.Dropout(dropout_rate))
        if apply_norm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_lrelu:
            result.add(tf.keras.layers.LeakyReLU())
        if apply_softmax:
            result.add(tf.keras.layers.Softmax())
        return result

    inp = tf.keras.layers.Input(shape=[config['img_height'],config['img_width'], 3], name='input_image')

    #inp = tf.keras.layers.Lambda(lambda x: tf.image.random_flip_left_right(x))(inp)
    #inp = tf.keras.layers.Lambda(lambda x: tf.image.random_flip_up_down(x))(inp)

    if 0:
        x = inp
        cl = 0  
        layers = []
        while x.get_shape().as_list()[1] > 4:
            fs = min(512, 32 * 2**cl)
            x = downsample(fs,size=3,strides=1)(x)
            layers.append(x)
            x = downsample(fs,size=3)(x)
            cl += 1 
        x = tf.keras.layers.Flatten()(x)
        x = dense(512,dropout_rate=0.5)(x)
        x = dense(config['num_classes'],apply_lrelu=False,apply_norm=False,apply_softmax=True)(x)
        model = tf.keras.Model(inputs = inp,outputs = x, name = 'Classifier')
    else:
        x = inp
        cl = 0  
        layers = []
        x = downsample(32,size=3)(x)
        x = downsample(64,size=3)(x)
        while x.get_shape().as_list()[1] > 4:
            fs = min(512, 128 * 2**cl)
            xx = tf.keras.layers.Conv2D(fs, 1, strides=1, padding='same')(x) 

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(fs, 3, strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(fs, 3, strides=1, padding='same')(x)
            x = xx + x
            x = tf.keras.layers.MaxPool2D()(x)
            layers.append(x)
            cl += 1 
        x = tf.keras.layers.Flatten()(x)
        x = dense(512,dropout_rate=0.5)(x)
        x = dense(config['num_classes'],apply_lrelu=False,apply_norm=False,apply_softmax=True)(x)
        model = tf.keras.Model(inputs = inp,outputs = x, name = 'Resnet Classifier')

    model.summary()

    return layers,model




def train_classifier(config, image_dir):
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/convid19/classification/%s" % now)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    labels, data_train, data_val = load_dataset(config)
    
    layers, model = make_model(config)
    grad_model = tf.keras.models.Model([model.inputs], [layers[1], model.output])

    optimizer=tf.keras.optimizers.Adam(lr=config['lr'])

    ## <train callbacks>
    cb_save = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=False, mode='auto', save_freq=500 )
    cb_tb = tf.keras.callbacks.TensorBoard(
        log_dir=checkpoint_path, histogram_freq=0, write_graph=True, write_images=True,
        update_freq=5 )

    # Define the per-epoch callback.
    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(data_val)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
        
    cb_cm = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    
    file_writer_cm = tf.summary.create_file_writer(checkpoint_path )
    def show_images(epoch, logs):
        with tf.device("cpu:0"):
            with file_writer_cm.as_default():
                tf.summary.image("Input Image", model.inputs[0], step=epoch)
        
    cb_im = tf.keras.callbacks.LambdaCallback(on_epoch_end=show_images)
    callbacks = [
        cb_save, cb_tb#, cb_im#cb_cm
    ]
    ## </callbacks>
    #model.fit(data_train,epochs=config['epochs'],
    #            validation_data=data_val,
    #            callbacks = callbacks)

    step = 0 
    writer = tf.summary.create_file_writer(checkpoint_path+'/train')
    writer_test = tf.summary.create_file_writer(checkpoint_path+'/test')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    last_ckpt_time = time.time()
    last_summary_step = 0
    
    moving_loss, moving_accuracy, k = 1., 0., .95
    def train_step(im,label, last_ckpt_time, last_summary_step, moving_accuracy, moving_loss, data_val, should_summarize=True):
        with tf.GradientTape(persistent=True) as tape:
            predicted = model(im)#,training=True)
            #loss = cce(label,predicted)
            #print(label, predicted)
            #print('----')
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, predicted))
            moving_loss = k * moving_loss + (1-k) * loss 
        gradients = tape.gradient(loss,model.trainable_variables)
        grad_magn = np.linalg.norm(gradients[0].numpy())
        #print(step,'grad',grad_magn)
        # update weights
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        # summary
        if step % 100 == 0 and should_summarize:
            def get_accuracy(_lab,_pred):
                _pred = tf.reshape(_pred,[_lab.shape[0],config['num_classes']])
                correct_prediction = tf.equal(tf.argmax(_lab,1), tf.argmax(_pred,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                return accuracy

            with tf.device("cpu:0"):
                def im_summary(name,data):
                    tf.summary.image(name,(data+1)/2,step=step)
            
            with writer.as_default():
                accuracy = get_accuracy(label,predicted)
                moving_accuracy = k * moving_accuracy + (1.-k) * accuracy
                tf.summary.scalar("accuracy",moving_accuracy,step=step)
                tf.summary.scalar("loss",moving_loss,step=step)
                
                if step % 200 == 0:
                    vis_grad_covid = vis_gradcam(grad_model, im, label,0)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam covid',vis_grad_covid,step=step)
                    vis_grad_pneumonia = vis_gradcam(grad_model, im, label,1)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam pneumonia',vis_grad_pneumonia,step=step)
                    vis_grad_nofinding = vis_gradcam(grad_model, im, label,2)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam no finding',vis_grad_nofinding,step=step)

                writer.flush() 
            
            if step % 200 == 0:
                # test data
                with writer_test.as_default():
                    loss_test = 0.
                    accuracy_test = 0.
                    ctest = 0
                    for (testim,testlabel) in data_val:
                        predicted_test = model( testim, training=False )
                        accuracy_test+= get_accuracy(testlabel, predicted_test)
                        _loss_test = cce(testlabel, predicted_test).numpy()
                        if not np.isnan(_loss_test):
                            loss_test += _loss_test
                        ctest += 1
                        if ctest == 4:
                            break
                    loss_test = loss_test / ctest 
                    accuracy_test = accuracy_test / ctest
                    tf.summary.scalar("loss",loss_test,step=step)
                    tf.summary.scalar("accuracy",accuracy_test,step=step)
                    print('accuracy train/test %f/%f \t loss train/test %f/%f'%(accuracy,accuracy_test,loss,loss_test))
                    
                    vis_grad_covid = vis_gradcam(grad_model, testim, testlabel,0)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam covid',vis_grad_covid,step=step)
                    vis_grad_pneumonia = vis_gradcam(grad_model, testim, testlabel,1)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam pneumonia',vis_grad_pneumonia,step=step)
                    vis_grad_nofinding = vis_gradcam(grad_model, testim, testlabel,2)
                    with tf.device("cpu:0"):
                        tf.summary.image('gradcam no finding',vis_grad_nofinding,step=step)

                    writer_test.flush() 
                    

            tf.summary.flush(writer); tf.summary.flush(writer_test)
            last_summary_step = step

        if time.time() - last_ckpt_time > 60. * 30.: # each hour #epoch % 100 == 0:
            ckpt_save_path = ckpt_manager.save()
            last_ckpt_time = time.time()
            model.save(os.path.join(checkpoint_path,'classifier.h5'))
            print('[*] saved model to %s.' % ckpt_save_path)
            if 0:
                del labels 
                del data_train 
                del data_val 
                labels, data_train, data_val = load_dataset(config)
        return moving_loss, moving_accuracy, last_ckpt_time, last_summary_step

    print('[*] starting training at %s with config' % now, config)
    use_mixup = True 
    for epoch in range(config['epochs']):
        if 1:#try:
            for (imo, labelo) in data_train: 
                #   for ddata in data_train:
                #print(ddata)
                #feature = ddata.features.feature
                #ddata = ddata.numpy()
                #print(ddata)
                #print('keys',ddata.shape)
                #example = tf.train.Example()
                #example.ParseFromString(ddata.numpy())
                #print(example)
                #raw_img = example['image/encoded'].bytes_list.value[0]
                #imo = tf.image.decode_png(raw_img)
                #labelo = None
                use_mixup = np.random.random() < 0.3 #0#step < 2500
                if use_mixup:
                    for _ in range(3):
                        # mixup augmentation
                        _seed = 1337
                        mixup_r = tf.random.uniform([config['batch_size']],minval=0.0,maxval=0.2)
                        mixup_ri = tf.expand_dims(mixup_r,1)
                        mixup_ri = tf.expand_dims(mixup_ri,1)
                        mixup_ri = tf.expand_dims(mixup_ri,1)
                        mixup_rl = tf.expand_dims(mixup_r,1)
                        
                        #imshu = tf.random.shuffle(imo,seed=_seed)
                        #labelshu = tf.random.shuffle(labelo,seed=_seed)
                        imshu = imo[::-1,:,:,:]
                        labelshu = labelo[::-1,:]
                        #print('mixups',mixup_r.get_shape().as_list(),mixup_ri.get_shape().as_list(),mixup_rl.get_shape().as_list())

                        try:
                            im = mixup_ri * imo + (1.0 - mixup_ri) * imshu
                            label = mixup_rl * labelo + (1.0 - mixup_rl) * labelshu
                            #print('im',im.get_shape().as_list())
                            #print('label',label.get_shape().as_list())
                            _, _, last_ckpt_time, last_summary_step = train_step(im, label, last_ckpt_time, last_summary_step, moving_accuracy, moving_loss, data_val, should_summarize=False)
                            #step += 1
                        except Exception as e:
                            print('exception mixup train step')
                            print(e) 

                
                if 1:#try:
                    moving_loss, moving_accuracy, last_ckpt_time, last_summary_step = train_step(imo, labelo, last_ckpt_time, last_summary_step, moving_accuracy, moving_loss, data_val )
                    step += 1
                #except Exception as e:
                #    print('exception normal train step')
                #   print(e) 

                
        #except Exception as e:
        #    print('epoch fail',epoch)
        #    print(e)
        
        print('[*] epoch %i/%i done with %i total steps at %s.'%(epoch,config['epochs'],step, str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]))

def train_classifier2(config, image_dir):
    #labels, data_train, data_val = load_dataset(config)
    

    #data = tf.data.Dataset.list_files(os.path.join(image_dir,'*.png'),seed=1902)
    # https://github.com/tensorflow/tensorflow/issues/32052 data = data.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
    files = glob(os.path.join(image_dir,'*.png'))
    from random import shuffle 
    shuffle(files)
    files = files[:500]
    images = tf.data.Dataset.from_tensor_slices([np.float32(cv.resize(cv.imread(p),(config['img_width'],config['img_height'])))/255. for p in files])
    labels = tf.data.Dataset.from_tensor_slices([[[0.,1.],[1.,0.]][int('dog' in p.split('/')[-1])] for p in files])
    data = tf.data.Dataset.zip((images,labels))
    
    data = data.shuffle(512//4)
    data = data.batch(config['batch_size'])#.prefetch(tf.data.experimental.AUTOTUNE)#.cache()
    nrepeat = 8 
    print('[*] loaded images from disk')
    test_dataset = data.take(nrepeat*32 // config['batch_size']).repeat() 
    train_dataset = data.skip(nrepeat*32 // config['batch_size'])
    
    inp = tf.keras.layers.Input(shape=[config['img_height'],config['img_width'], 3], name='input_image')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu',strides=2)(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu',strides=2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu',strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dense(config['num_classes'])(x)
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inp, x)
    model.summary()

    print('trainable variables:')
    for var in model.trainable_variables:
        print(var.name)

    step = 0
    optimizer=tf.keras.optimizers.Adam(lr=config['lr'])
    for epoch in range(100):
        for (im, label) in train_dataset: 
            #print()
            with tf.GradientTape(persistent=True) as tape:
                predicted = model(im,training=True)
                #loss = cce(label,predicted)
                
                #print('----')
                loss = tf.keras.losses.categorical_crossentropy(label, predicted)

            gradients = tape.gradient(loss,model.trainable_variables)
            grad_magn = np.linalg.norm(gradients[0].numpy())
            #print('im mean',tf.reduce_mean(im,axis=[1,2]).numpy(),'  ===>   ',label.numpy())
            #print(label.numpy(), predicted.numpy())
            print(step,'loss',loss.numpy(),'grad',grad_magn)
            
            # update weights
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))

            step += 1 

if __name__ == "__main__":
    k=4
    config = {
        'num_classes': 3,
        "lr": 2e-5,
        "batch_size": 16,
        'epochs': 1000,
        "img_width": k*128,
        "img_height": k*128           
    }

    image_dir = os.path.expanduser('~/data/convid/findings')
    config['image_dir'] = image_dir

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
        for m in ['train','test']:
            for l in [0,1,2]:
                os.makedirs(os.path.join(image_dir,m,str(l)))

        findings_labels, annotation_data = load_annotation_data(config)
        create_dataset(image_dir, config, findings_labels, annotation_data)

    train_classifier(config, image_dir)