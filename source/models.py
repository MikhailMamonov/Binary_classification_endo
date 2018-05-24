import keras.backend.tensorflow_backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop,SGD, Adagrad, Adam
from keras.utils import multi_gpu_model
from utils import CLASSES


def get_gpus(gpus):
    return list(map(int, gpus.split(',')))


def get_model(model, gpus=1, **kwargs):
    """
    Returns compiled keras parallel model ready for training
    and base model that must be used for saving weights

    Params:
    - model - model type
    - gpus - a list with numbers of GPUs
    """
    if model == 'vgg16' or model == 'vgg19':
        return vgg(gpus, model)
    if model == 'skin_rec':
        return skin_rec(gpus, model)
    if model == 'lung_rec':
        return lung_rec(gpus, model)
    if model == 'alex_net':
        return alex_net(gpus, model)
    if model == 'incresnet':
        return inception_res_net_v2(gpus)
    if model == 'incv3':
        return inception_v3(gpus)
    if model == 'xcept':
        return xception(gpus)
    if model == 'resnet50':
        return resnet50(gpus)
    if model == 'densenet':
        return dense_net(gpus)
    if model == 'nasnet':
        return nasnet(gpus)
    raise ValueError('Wrong model value!')

def alex_net(gpus,model):
    frozen = 0

    model = Sequential()

    # Layer 1
    model.add(Convolution2D(32, 3, 3, input_shape = (141, 141, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 
    model.add(Convolution2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    # Layer 6
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    
    # Layer 7
    model.add(Dense(output_dim = 64, activation = 'relu'))
    model.add(Dropout(0.5))
    
    # Layer 8
    
    output = Dense(len(CLASSES), init='glorot_normal',activation='softmax')(model.output)
    
    return _compile(gpus, model.input, output, frozen)

def vgg(gpus, model):
    """
    Returns compiled keras vgg16 model ready for training
    """

    gpu = get_gpus(gpus)
    if model == 'vgg16':
        

        base_model = VGG16(
            weights= 'imagenet', include_top=False, input_shape=(224, 224, 3))
        frozen = 14
    elif model == 'vgg19':
        base_model = VGG19(
            weights= None, include_top=False, input_shape=(224, 224, 3))
        frozen = 16
    else:
        raise ValueError('Wrong VGG model type!')
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASSES), activation='softmax')(x)

    # x = Flatten(name='flatten')(base_model.output)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # output = Dense(1, activation='sigmoid')(x)
    return _compile(gpus, base_model.input, output, 0)

def skin_rec(gpus, model):
    nb_filters = 64
    k_size = (3, 3)
    pl_size = (2, 2)
    gpu = get_gpus(gpus)
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size=k_size, activation='relu', input_shape=(141, 141, 3)))
    model.add(Conv2D(nb_filters-4, k_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Conv2D(nb_filters-8, kernel_size=k_size, activation='relu'))
    model.add(Conv2D(nb_filters-12, kernel_size=k_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters-16, kernel_size=k_size, activation='relu'))
    model.add(Conv2D(nb_filters-20, kernel_size=k_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))	)

	
    x = Flatten(name='flatten')(model.output) 
    x = Dense(128, activation='relu', name='fc1')(x)

    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
#model.add(Dense(1, activation='sigmoid'))
	      
    print('Model flattened out to ', model.output_shape) 
    print(type(gpus))
    print(type(model.input))
    print(type(output))
    return _compile(gpus, model.input, output, 0)

def lung_rec(gpus, model):
    k_size = (3, 3)
    pl_size = (2, 2)
    gpu = get_gpus(gpus)
    model = Sequential()
    model.add(Conv2D(50, kernel_size=(11,11), activation='relu', input_shape=(141, 141, 3)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(120, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    x = Flatten(name='flatten')(model.output) 
    x = Dense(10, activation='relu', name='fc1')(x)

    output = Dense(len(CLASSES), activation='softmax')(x)

	      
    print('Model flattened out to ', model.output_shape) 
    print(type(gpus))
    print(type(model.input))
    print(type(output))
    return _compile(gpus, model.input, output, 0)



def inception_v3(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 29
	base_model = InceptionV3(
		weights='imagenet', include_top=False, input_shape=(141, 141, 3))

	x = GlobalAveragePooling2D()(base_model.output)
	x = Dense(1024, activation='relu')(x)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)
	return _compile(gpus, base_model.input, output, frozen)


def inception_res_net_v2(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 0  # TODO
	base_model = InceptionResNetV2(
		weights='imagenet', include_top=False, input_shape=(299, 299, 3))

	x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

	return _compile(gpus, base_model.input, output, frozen)


def xception(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 125
	base_model = Xception(
		weights='imagenet', include_top=False, input_shape=(299, 299, 3))

	x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
	x = Dense(1024, activation='relu')(x)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

	return _compile(gpus, base_model.input, output, frozen)


def resnet50(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 0
	base_model = ResNet50(
		weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	x = Flatten()(base_model.output)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

	return _compile(gpus, base_model.input, output, frozen)


def dense_net(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 0
	base_model = DenseNet201(
		weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

	return _compile(gpus, base_model.input, output, frozen)


def nasnet(gpus):
	"""
	Returns compiled keras vgg16 model ready for training
	"""
	frozen = 0
	base_model = NASNetLarge(
		weights='imagenet', include_top=False, input_shape=(331, 331, 3))

	x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
	output = Dense(len(CLASSES), activation='softmax', name='predictions')(x)

	return _compile(gpus, base_model.input, output, frozen)


def _compile(gpus, input, output, frozen):
	gpus = get_gpus(gpus)
	if len(gpus) == 1:
		with K.tf.device('/gpu:{}'.format(gpus[0])):
			model = Model(input, output)
			for layer in model.layers[:frozen]:
				layer.trainable = False
			parallel_model = model
	else:
		with K.tf.device('/cpu:0'):
			model = Model(input, output)
			for layer in model.layers[:frozen]:
				layer.trainable = False
		parallel_model = multi_gpu_model(model, gpus=gpus)
	parallel_model.compile(
		loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])
	return parallel_model, model
