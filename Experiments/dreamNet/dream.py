# coding: utf-8

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import cv2

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

model_path = '../examples/face_detection/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'face_tunning_train_iter_3000.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

#####################################################################################

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    


def deepdream(name, net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            # vis = deprocess(net, src.data[0])
            # if not clip: # adjust image contrast if clipping is disabled
            #     vis = vis*(255.0/np.percentile(vis, 99.98))
            # # showarray(vis)
            # # cv2.imwrite('vis/%d_%s_%02d.jpg'%(name, end,i),vis)
            # # print octave, i, end, vis.shape
            # #PIL.Image.fromarray(np.uint8(vis)).save("vis/%d_%s_%02d.jpg"%(name, end, i))
            # clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

print net.blobs.keys()

# showarray(img)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#_=deepdream(net, img)

# _=deepdream(net, img, end='inception_3b/5x5_reduce')

ends = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}
end = 'conv4'
i = 36
for image in range(50,62):
    # for end in ends:
        img = np.float32(PIL.Image.open('praia_montanha/Base/training/%d.jpg'%image))
        _=deepdream(image, net, img, end=end)
        PIL.Image.fromarray(np.uint8(_)).save("conv/%d.jpg"%(i))
        i+=1

# end = 'conv5'
# for image in range(0,100):
#     # for end in ends:
#         img = np.float32(PIL.Image.open('praia_montanha/Base/training/%d.jpg'%image))
#         frame = img
#         frame_i = 0
#         h, w = frame.shape[:2]
#         s = 0.05 # scale coefficient
#         for i in xrange(24):
#             frame = deepdream(image, net, frame, end=end)
#             PIL.Image.fromarray(np.uint8(frame)).save("frames/%s_%s_%04d.jpg"%(image, end, frame_i))
#             frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
#             frame_i += 1


