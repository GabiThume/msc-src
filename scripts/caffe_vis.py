#  net = caffe.Net('examples/praia_montanha_original/deploy.prototxt', 'examples/praia_montanha_original/caffe_train_iter_5000.caffemodel')

# [(k, v.data.shape) for k, v in net.blobs.items()]
# Out[11]:
# [('data', (1, 3, 256, 256)),
#  ('conv1', (1, 96, 62, 62)),
#  ('pool1', (1, 96, 31, 31)),
#  ('norm1', (1, 96, 31, 31)),
#  ('conv2', (1, 256, 31, 31)),
#  ('pool2', (1, 256, 15, 15)),
#  ('norm2', (1, 256, 15, 15)),
#  ('conv3', (1, 384, 15, 15)),
#  ('conv4', (1, 384, 15, 15)),
#  ('conv5', (1, 256, 15, 15)),
#  ('pool5', (1, 256, 7, 7)),
#  ('fc6', (1, 4096, 1, 1)),
#  ('fc7', (1, 4096, 1, 1)),
#  ('fc8', (1, 2, 1, 1)),
#  ('prob', (1, 2, 1, 1))]

# [(k, v[0].data.shape) for k, v in net.params.items()]
# Out[13]:
# [('conv1', (96, 3, 11, 11)),
#  ('conv2', (256, 48, 5, 5)),
#  ('conv3', (384, 256, 3, 3)),
#  ('conv4', (384, 192, 3, 3)),
#  ('conv5', (256, 192, 3, 3)),
#  ('fc6', (1, 1, 4096, 12544)),
#  ('fc7', (1, 1, 4096, 4096)),
#  ('fc8', (1, 1, 2, 4096))]

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


MODEL_FILE= 'examples/praia_montanha_original/deploy.prototxt'
PRETRAINED= 'examples/praia_montanha_original/caffe_train_iter_5000.caffemodel'
FIGURES_PATH = '/home/gabi/Desktop/Dropbox/vis'

# def montanha(n):
# IMAGE_FILE='../imagefeatureextraction/Desbalanced/original/2/'+str(n)+'.jpg'
# IMAGE_FILE='examples/images/cat.jpg'
IMAGE_FILE='/home/gabi/Desktop/src/Corel1000/2/11.jpg'
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

    # plt.imshow(data)
    plt.imshow(data,cmap=cm.gray)
    plt.show()

####################################
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.title('conv1')
plt.savefig(FIGURES_PATH+'/conv1.png')
plt.clf()

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)
plt.title('conv1-blobs')
plt.savefig(FIGURES_PATH+'/conv1-blob.png')
plt.clf()

feat = net.blobs['pool1'].data[0]
vis_square(feat, padval=1)
plt.title('pool1')
plt.savefig(FIGURES_PATH+'/pool1.png')
plt.clf()

feat = net.blobs['norm1'].data[0]
vis_square(feat, padval=1)
plt.title('norm1')
plt.savefig(FIGURES_PATH+'/norm1.png')
plt.clf()
####################################
filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))
plt.title('conv2')
plt.savefig(FIGURES_PATH+'/conv2.png')
plt.clf()

feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)
plt.title('conv2-blobs')
plt.savefig(FIGURES_PATH+'/conv2-blob.png')
plt.clf()

feat = net.blobs['pool2'].data[0]
vis_square(feat, padval=1)
plt.title('pool2')
plt.savefig(FIGURES_PATH+'/pool2.png')
plt.clf()

feat = net.blobs['norm2'].data[0]
vis_square(feat, padval=1)
plt.title('norm2')
plt.savefig(FIGURES_PATH+'/norm2.png')
plt.clf()
####################################
filters = net.params['conv3'][0].data
vis_square(filters[:25].reshape(48**2, 5, 5))
plt.title('conv3-filters')
plt.savefig(FIGURES_PATH+'/conv3-filters.png')
plt.clf()

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv3-blobs')
plt.savefig(FIGURES_PATH+'/conv3-blob.png')
plt.clf()
####################################
feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv4')
plt.savefig(FIGURES_PATH+'/conv4.png')
plt.clf()
####################################
feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)
plt.title('conv5')
plt.savefig(FIGURES_PATH+'/conv5.png')
plt.clf()

feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)
plt.title('pool5')
plt.savefig(FIGURES_PATH+'/pool5.png')
plt.clf()
#################################
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc6')
plt.savefig(FIGURES_PATH+'/fc6.png')
plt.clf()
#################################
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc7')
plt.savefig(FIGURES_PATH+'/fc7.png')
plt.clf()
#################################
feat = net.blobs['fc8'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=10)
plt.show()
plt.title('fc8')
plt.savefig(FIGURES_PATH+'/fc8.png')
plt.clf()
#################################
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.title('prob')
plt.savefig(FIGURES_PATH+'/prob.png')
plt.clf()
#################################


# feat = net.blobs['pool3'].data[0]
# vis_square(feat, padval=1)
# plt.title('pool3')
# plt.savefig(FIGURES_PATH+'/pool3.png')

# feat = net.blobs['pool4'].data[0]
# vis_square(feat, padval=1)
# plt.title('pool4')
# plt.savefig(FIGURES_PATH+'/pool4.png')


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
