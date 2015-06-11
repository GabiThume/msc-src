
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gabi/Desktop/src/caffe/python/')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

In [10]: net = caffe.Net('examples/praia_montanha_original/deploy.prototxt', 'examples/praia_montanha_original/caffe_train_iter_5000.caffemodel')
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0603 15:21:10.775430   832 net.cpp:39] Initializing net from parameters: 
name: "CaffeNet"
layers {
  bottom: "data"
  top: "conv1"
  name: "conv1"
  type: CONVOLUTION
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
layers {
  bottom: "conv1"
  top: "conv1"
  name: "relu1"
  type: RELU
}
layers {
  bottom: "conv1"
  top: "pool1"
  name: "pool1"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  bottom: "pool1"
  top: "norm1"
  name: "norm1"
  type: LRN
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layers {
  bottom: "norm1"
  top: "conv2"
  name: "conv2"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
  }
}
layers {
  bottom: "conv2"
  top: "conv2"
  name: "relu2"
  type: RELU
}
layers {
  bottom: "conv2"
  top: "pool2"
  name: "pool2"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  bottom: "pool2"
  top: "norm2"
  name: "norm2"
  type: LRN
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layers {
  bottom: "norm2"
  top: "conv3"
  name: "conv3"
  type: CONVOLUTION
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv3"
  top: "conv3"
  name: "relu3"
  type: RELU
}
layers {
  bottom: "conv3"
  top: "conv4"
  name: "conv4"
  type: CONVOLUTION
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layers {
  bottom: "conv4"
  top: "conv4"
  name: "relu4"
  type: RELU
}
layers {
  bottom: "conv4"
  top: "conv5"
  name: "conv5"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layers {
  bottom: "conv5"
  top: "conv5"
  name: "relu5"
  type: RELU
}
layers {
  bottom: "conv5"
  top: "pool5"
  name: "pool5"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  bottom: "pool5"
  top: "fc6"
  name: "fc6"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
  }
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "relu6"
  type: RELU
}
layers {
  bottom: "fc6"
  top: "fc6"
  name: "drop6"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc6"
  top: "fc7"
  name: "fc7"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 4096
  }
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "relu7"
  type: RELU
}
layers {
  bottom: "fc7"
  top: "fc7"
  name: "drop7"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc7"
  top: "fc8"
  name: "fc8"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 2
  }
}
layers {
  bottom: "fc8"
  top: "prob"
  name: "prob"
  type: SOFTMAX
}
input: "data"
input_dim: 1
input_dim: 3
input_dim: 256
input_dim: 256
I0603 15:21:10.776654   832 net.cpp:358] Input 0 -> data
I0603 15:21:10.776705   832 net.cpp:67] Creating Layer conv1
I0603 15:21:10.776721   832 net.cpp:394] conv1 <- data
I0603 15:21:10.776739   832 net.cpp:356] conv1 -> conv1
I0603 15:21:10.776759   832 net.cpp:96] Setting up conv1
I0603 15:21:10.776903   832 net.cpp:103] Top shape: 1 96 62 62 (369024)
I0603 15:21:10.776938   832 net.cpp:67] Creating Layer relu1
I0603 15:21:10.776953   832 net.cpp:394] relu1 <- conv1
I0603 15:21:10.776971   832 net.cpp:345] relu1 -> conv1 (in-place)
I0603 15:21:10.776988   832 net.cpp:96] Setting up relu1
I0603 15:21:10.777003   832 net.cpp:103] Top shape: 1 96 62 62 (369024)
I0603 15:21:10.777019   832 net.cpp:67] Creating Layer pool1
I0603 15:21:10.777032   832 net.cpp:394] pool1 <- conv1
I0603 15:21:10.777048   832 net.cpp:356] pool1 -> pool1
I0603 15:21:10.777065   832 net.cpp:96] Setting up pool1
I0603 15:21:10.777088   832 net.cpp:103] Top shape: 1 96 31 31 (92256)
I0603 15:21:10.777108   832 net.cpp:67] Creating Layer norm1
I0603 15:21:10.777122   832 net.cpp:394] norm1 <- pool1
I0603 15:21:10.777137   832 net.cpp:356] norm1 -> norm1
I0603 15:21:10.777153   832 net.cpp:96] Setting up norm1
I0603 15:21:10.777168   832 net.cpp:103] Top shape: 1 96 31 31 (92256)
I0603 15:21:10.777184   832 net.cpp:67] Creating Layer conv2
I0603 15:21:10.777197   832 net.cpp:394] conv2 <- norm1
I0603 15:21:10.777212   832 net.cpp:356] conv2 -> conv2
I0603 15:21:10.777230   832 net.cpp:96] Setting up conv2
I0603 15:21:10.778079   832 net.cpp:103] Top shape: 1 256 31 31 (246016)
I0603 15:21:10.778110   832 net.cpp:67] Creating Layer relu2
I0603 15:21:10.778125   832 net.cpp:394] relu2 <- conv2
I0603 15:21:10.778143   832 net.cpp:345] relu2 -> conv2 (in-place)
I0603 15:21:10.778159   832 net.cpp:96] Setting up relu2
I0603 15:21:10.778172   832 net.cpp:103] Top shape: 1 256 31 31 (246016)
I0603 15:21:10.778187   832 net.cpp:67] Creating Layer pool2
I0603 15:21:10.778199   832 net.cpp:394] pool2 <- conv2
I0603 15:21:10.778236   832 net.cpp:356] pool2 -> pool2
I0603 15:21:10.778259   832 net.cpp:96] Setting up pool2
I0603 15:21:10.778275   832 net.cpp:103] Top shape: 1 256 15 15 (57600)
I0603 15:21:10.778295   832 net.cpp:67] Creating Layer norm2
I0603 15:21:10.778307   832 net.cpp:394] norm2 <- pool2
I0603 15:21:10.778322   832 net.cpp:356] norm2 -> norm2
I0603 15:21:10.778337   832 net.cpp:96] Setting up norm2
I0603 15:21:10.778352   832 net.cpp:103] Top shape: 1 256 15 15 (57600)
I0603 15:21:10.778367   832 net.cpp:67] Creating Layer conv3
I0603 15:21:10.778380   832 net.cpp:394] conv3 <- norm2
I0603 15:21:10.778395   832 net.cpp:356] conv3 -> conv3
I0603 15:21:10.778410   832 net.cpp:96] Setting up conv3
I0603 15:21:10.780201   832 net.cpp:103] Top shape: 1 384 15 15 (86400)
I0603 15:21:10.780246   832 net.cpp:67] Creating Layer relu3
I0603 15:21:10.780263   832 net.cpp:394] relu3 <- conv3
I0603 15:21:10.780289   832 net.cpp:345] relu3 -> conv3 (in-place)
I0603 15:21:10.780318   832 net.cpp:96] Setting up relu3
I0603 15:21:10.780339   832 net.cpp:103] Top shape: 1 384 15 15 (86400)
I0603 15:21:10.780366   832 net.cpp:67] Creating Layer conv4
I0603 15:21:10.780387   832 net.cpp:394] conv4 <- conv3
I0603 15:21:10.780412   832 net.cpp:356] conv4 -> conv4
I0603 15:21:10.780442   832 net.cpp:96] Setting up conv4
I0603 15:21:10.782467   832 net.cpp:103] Top shape: 1 384 15 15 (86400)
I0603 15:21:10.782528   832 net.cpp:67] Creating Layer relu4
I0603 15:21:10.782557   832 net.cpp:394] relu4 <- conv4
I0603 15:21:10.782588   832 net.cpp:345] relu4 -> conv4 (in-place)
I0603 15:21:10.782618   832 net.cpp:96] Setting up relu4
I0603 15:21:10.782634   832 net.cpp:103] Top shape: 1 384 15 15 (86400)
I0603 15:21:10.782655   832 net.cpp:67] Creating Layer conv5
I0603 15:21:10.782676   832 net.cpp:394] conv5 <- conv4
I0603 15:21:10.782713   832 net.cpp:356] conv5 -> conv5
I0603 15:21:10.782742   832 net.cpp:96] Setting up conv5
I0603 15:21:10.784080   832 net.cpp:103] Top shape: 1 256 15 15 (57600)
I0603 15:21:10.784127   832 net.cpp:67] Creating Layer relu5
I0603 15:21:10.784143   832 net.cpp:394] relu5 <- conv5
I0603 15:21:10.784160   832 net.cpp:345] relu5 -> conv5 (in-place)
I0603 15:21:10.784178   832 net.cpp:96] Setting up relu5
I0603 15:21:10.784193   832 net.cpp:103] Top shape: 1 256 15 15 (57600)
I0603 15:21:10.784209   832 net.cpp:67] Creating Layer pool5
I0603 15:21:10.784224   832 net.cpp:394] pool5 <- conv5
I0603 15:21:10.784242   832 net.cpp:356] pool5 -> pool5
I0603 15:21:10.784270   832 net.cpp:96] Setting up pool5
I0603 15:21:10.784286   832 net.cpp:103] Top shape: 1 256 7 7 (12544)
I0603 15:21:10.784306   832 net.cpp:67] Creating Layer fc6
I0603 15:21:10.784319   832 net.cpp:394] fc6 <- pool5
I0603 15:21:10.784335   832 net.cpp:356] fc6 -> fc6
I0603 15:21:10.784351   832 net.cpp:96] Setting up fc6
I0603 15:21:10.904543   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.904602   832 net.cpp:67] Creating Layer relu6
I0603 15:21:10.904614   832 net.cpp:394] relu6 <- fc6
I0603 15:21:10.904626   832 net.cpp:345] relu6 -> fc6 (in-place)
I0603 15:21:10.904639   832 net.cpp:96] Setting up relu6
I0603 15:21:10.904649   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.904660   832 net.cpp:67] Creating Layer drop6
I0603 15:21:10.904669   832 net.cpp:394] drop6 <- fc6
I0603 15:21:10.904683   832 net.cpp:345] drop6 -> fc6 (in-place)
I0603 15:21:10.904695   832 net.cpp:96] Setting up drop6
I0603 15:21:10.904703   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.904714   832 net.cpp:67] Creating Layer fc7
I0603 15:21:10.904723   832 net.cpp:394] fc7 <- fc6
I0603 15:21:10.904733   832 net.cpp:356] fc7 -> fc7
I0603 15:21:10.904754   832 net.cpp:96] Setting up fc7
I0603 15:21:10.939723   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.939772   832 net.cpp:67] Creating Layer relu7
I0603 15:21:10.939784   832 net.cpp:394] relu7 <- fc7
I0603 15:21:10.939797   832 net.cpp:345] relu7 -> fc7 (in-place)
I0603 15:21:10.939810   832 net.cpp:96] Setting up relu7
I0603 15:21:10.939820   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.939831   832 net.cpp:67] Creating Layer drop7
I0603 15:21:10.939837   832 net.cpp:394] drop7 <- fc7
I0603 15:21:10.939846   832 net.cpp:345] drop7 -> fc7 (in-place)
I0603 15:21:10.939856   832 net.cpp:96] Setting up drop7
I0603 15:21:10.939863   832 net.cpp:103] Top shape: 1 4096 1 1 (4096)
I0603 15:21:10.939873   832 net.cpp:67] Creating Layer fc8
I0603 15:21:10.939880   832 net.cpp:394] fc8 <- fc7
I0603 15:21:10.939889   832 net.cpp:356] fc8 -> fc8
I0603 15:21:10.939908   832 net.cpp:96] Setting up fc8
I0603 15:21:10.939934   832 net.cpp:103] Top shape: 1 2 1 1 (2)
I0603 15:21:10.939949   832 net.cpp:67] Creating Layer prob
I0603 15:21:10.939956   832 net.cpp:394] prob <- fc8
I0603 15:21:10.939965   832 net.cpp:356] prob -> prob
I0603 15:21:10.939975   832 net.cpp:96] Setting up prob
I0603 15:21:10.939983   832 net.cpp:103] Top shape: 1 2 1 1 (2)
I0603 15:21:10.939991   832 net.cpp:172] prob does not need backward computation.
I0603 15:21:10.939999   832 net.cpp:172] fc8 does not need backward computation.
I0603 15:21:10.940007   832 net.cpp:172] drop7 does not need backward computation.
I0603 15:21:10.940014   832 net.cpp:172] relu7 does not need backward computation.
I0603 15:21:10.940021   832 net.cpp:172] fc7 does not need backward computation.
I0603 15:21:10.940028   832 net.cpp:172] drop6 does not need backward computation.
I0603 15:21:10.940037   832 net.cpp:172] relu6 does not need backward computation.
I0603 15:21:10.940043   832 net.cpp:172] fc6 does not need backward computation.
I0603 15:21:10.940050   832 net.cpp:172] pool5 does not need backward computation.
I0603 15:21:10.940057   832 net.cpp:172] relu5 does not need backward computation.
I0603 15:21:10.940064   832 net.cpp:172] conv5 does not need backward computation.
I0603 15:21:10.940071   832 net.cpp:172] relu4 does not need backward computation.
I0603 15:21:10.940078   832 net.cpp:172] conv4 does not need backward computation.
I0603 15:21:10.940085   832 net.cpp:172] relu3 does not need backward computation.
I0603 15:21:10.940093   832 net.cpp:172] conv3 does not need backward computation.
I0603 15:21:10.940100   832 net.cpp:172] norm2 does not need backward computation.
I0603 15:21:10.940107   832 net.cpp:172] pool2 does not need backward computation.
I0603 15:21:10.940114   832 net.cpp:172] relu2 does not need backward computation.
I0603 15:21:10.940121   832 net.cpp:172] conv2 does not need backward computation.
I0603 15:21:10.940129   832 net.cpp:172] norm1 does not need backward computation.
I0603 15:21:10.940136   832 net.cpp:172] pool1 does not need backward computation.
I0603 15:21:10.940143   832 net.cpp:172] relu1 does not need backward computation.
I0603 15:21:10.940150   832 net.cpp:172] conv1 does not need backward computation.
I0603 15:21:10.940157   832 net.cpp:208] This network produces output prob
I0603 15:21:10.940174   832 net.cpp:467] Collecting Learning Rate and Weight Decay.
I0603 15:21:10.940186   832 net.cpp:219] Network initialization done.
I0603 15:21:10.940193   832 net.cpp:220] Memory required for data: 8110864

In [11]: 

In [11]: 

In [11]: [(k, v.data.shape) for k, v in net.blobs.items()]
Out[11]: 
[('data', (1, 3, 256, 256)),
 ('conv1', (1, 96, 62, 62)),
 ('pool1', (1, 96, 31, 31)),
 ('norm1', (1, 96, 31, 31)),
 ('conv2', (1, 256, 31, 31)),
 ('pool2', (1, 256, 15, 15)),
 ('norm2', (1, 256, 15, 15)),
 ('conv3', (1, 384, 15, 15)),
 ('conv4', (1, 384, 15, 15)),
 ('conv5', (1, 256, 15, 15)),
 ('pool5', (1, 256, 7, 7)),
 ('fc6', (1, 4096, 1, 1)),
 ('fc7', (1, 4096, 1, 1)),
 ('fc8', (1, 2, 1, 1)),
 ('prob', (1, 2, 1, 1))]

In [13]: [(k, v[0].data.shape) for k, v in net.params.items()]
Out[13]: 
[('conv1', (96, 3, 11, 11)),
 ('conv2', (256, 48, 5, 5)),
 ('conv3', (384, 256, 3, 3)),
 ('conv4', (384, 192, 3, 3)),
 ('conv5', (256, 192, 3, 3)),
 ('fc6', (1, 1, 4096, 12544)),
 ('fc7', (1, 1, 4096, 4096)),
 ('fc8', (1, 1, 2, 4096))]



# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()


# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


##################################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/gabi/Desktop/src/caffe/python/')
import caffe
import  matplotlib.cm as cm

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


MODEL_FILE= 'examples/praia_montanha_original-/deploy.prototxt'
PRETRAINED= 'examples/praia_montanha_original-/caffe_train_iter_5000.caffemodel'
IMAGE_FILE='/home/gabi/Desktop/src/Corel1000/2/11.jpg'

# def montanha(n):
# IMAGE_FILE='../imagefeatureextraction/Desbalanced/original/2/'+str(n)+'.jpg'
# IMAGE_FILE='examples/images/cat.jpg'
net = caffe.Classifier(MODEL_FILE, PRETRAINED, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
input_image = caffe.io.load_image(IMAGE_FILE)
n_iterations = 5000
label_index = 0  # Index for beach class
# label_index = 1  # Index for mountains class
caffe_data = np.random.random((1,3,256,256))
caffeLabel = np.zeros((1,2,1,1))
caffeLabel[0,label_index,0,0] = 1;

# out = net.forward()
# print("Predicted class is #{}.".format(out['prob'].argmax()))

#Perform a forward pass with the data as the input image
prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
bw = net.backward(**{net.outputs[0]: caffeLabel})
diff = bw['data']
print net.blobs['prob'].data[0]
# print("Predicted class is #{}.".format(out['prob'].argmax()))


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data,cmap=cm.gray)
    plt.show()


####################################
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.title('conv1')
plt.savefig('/home/gabi/Dropbox/vis/conv1.png')
plt.clf()

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)
plt.title('conv1-blobs')
plt.savefig('/home/gabi/Dropbox/vis/conv1-blob.png')
plt.clf()

feat = net.blobs['pool1'].data[0]
vis_square(feat, padval=1)
plt.title('pool1')
plt.savefig('/home/gabi/Dropbox/vis/pool1.png')
plt.clf()

feat = net.blobs['norm1'].data[0]
vis_square(feat, padval=1)
plt.title('norm1')
plt.savefig('/home/gabi/Dropbox/vis/norm1.png')
plt.clf()
####################################
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))
plt.title('conv2')
plt.savefig('/home/gabi/Dropbox/vis/conv2.png')
plt.clf()

feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)
plt.title('conv2-blobs')
plt.savefig('/home/gabi/Dropbox/vis/conv2-blob.png')
plt.clf()

feat = net.blobs['pool2'].data[0]
vis_square(feat, padval=1)
plt.title('pool2')
plt.savefig('/home/gabi/Dropbox/vis/pool2.png')
plt.clf()

feat = net.blobs['norm2'].data[0]
vis_square(feat, padval=1)
plt.title('norm2')
plt.savefig('/home/gabi/Dropbox/vis/norm2.png')
plt.clf()
####################################
filters = net.params['conv3'][0].data
vis_square(filters[:25].reshape(48**2, 5, 5))
plt.title('conv3-filters')
plt.savefig('/home/gabi/Dropbox/vis/conv3-filters.png')
plt.clf()

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv3-blobs')
plt.savefig('/home/gabi/Dropbox/vis/conv3-blob.png')
plt.clf()
####################################
feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv4')
plt.savefig('/home/gabi/Dropbox/vis/conv4.png')
plt.clf()
####################################
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv5')
plt.savefig('/home/gabi/Dropbox/vis/conv5.png')
plt.clf()

feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
plt.title('pool5')
plt.savefig('/home/gabi/Dropbox/vis/pool5.png')
plt.clf()
#################################
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc6')
plt.savefig('/home/gabi/Dropbox/vis/fc6.png')
plt.clf()
#################################
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc7')
plt.savefig('/home/gabi/Dropbox/vis/fc7.png')
plt.clf()
#################################
feat = net.blobs['fc8'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc8')
plt.savefig('/home/gabi/Dropbox/vis/fc8.png')
plt.clf()
#################################
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.title('prob')
plt.savefig('/home/gabi/Dropbox/vis/prob.png')
plt.clf()
#################################


# feat = net.blobs['pool3'].data[0]
# vis_square(feat, padval=1)
# plt.title('pool3')
# plt.savefig('/home/gabi/Dropbox/vis/pool3.png')

# feat = net.blobs['pool4'].data[0]
# vis_square(feat, padval=1)
# plt.title('pool4')
# plt.savefig('/home/gabi/Dropbox/vis/pool4.png')

feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)


# Plot each derivative of each layer and save each fig.
# feat = net.blobs['conv1'].diff[0]
# vis_square(feat, padval=1)
# plt.show()
# plt.title('conv1')
# plt.savefig('ps3part3_conv1.png')

#probability output
prob = net.blobs['prob'].data[0]
plt.plot(prob.flat)
plt.show()