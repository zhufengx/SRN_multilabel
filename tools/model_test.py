import matplotlib 
matplotlib.use('Agg')
import sys
sys.path.append('/home/fzhu/multi_label/caffe-fzhu/python')
import caffe
import numpy as np
import skimage.io
from skimage.transform import resize
import time
import argparse
import multiprocessing
# import matplotlib.pyplot as plt

from srn_io import ImageTransformer, load_image

parser = argparse.ArgumentParser()
parser.add_argument('img_root', type=str, help="root directory holding the images")
parser.add_argument('imglist_test', type=str, help="a list for tested images")
parser.add_argument('prototxt', type=str, help="deploy net prototxt")
parser.add_argument('caffemodel', type=str, help="trained caffe model")
parser.add_argument('save_name', type=str, help="prefix of output file name")
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument("--gpus", type=int, nargs='+', default=0, help='specify list of gpu to use')
parser.add_argument('--num_test', type=int, default=None)
args = parser.parse_args()
print args
num_worker = len(args.gpus)


def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] if num_worker > 1 else 1
    caffe.set_mode_gpu()
    caffe.set_device(args.gpus[(my_id-1)%num_worker])
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    input_dims = net.blobs['data'].data.shape
    net.blobs['data'].reshape(args.batch_size, input_dims[1], input_dims[2], input_dims[3])


def eval_im_batch(input_data):
    global net
    output_layer = net.outputs[0]
    imlist_batch = input_data[0]
    batch_id = input_data[1]

    # image pre-processing settings
    transformer = ImageTransformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))    # h*w*c -> c*h*w
    transformer.set_mean('data', np.array((104, 117, 123)))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the net operates on images in [0,255]
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB -> BGR 

    # testing current batch
    start = time.time()
    input_dims = net.blobs['data'].data.shape
    input_batch = np.empty( (args.batch_size, input_dims[1], input_dims[2], input_dims[3]) ).astype(np.float32)
    for idx in range(0,len(imlist_batch)):
        line_split = imlist_batch[idx].split(' ')
        img_name = line_split[0]
        im = load_image(args.img_root + img_name)
        im = resize(im, input_dims[2:]).astype(np.float32)
        im_preprocessed = transformer.preprocess('data', im)
        input_batch[idx] = im_preprocessed.copy()
    # net forward
    net.blobs['data'].data[...] = input_batch
    out = net.forward()
    end = time.time()
    print 'batch: {} / {}, time: {:.3f} s'.format(batch_id + 1, len(start_idx), end - start)
    return out[output_layer].reshape(args.batch_size, -1).copy()

# load list of testing images
f = open(args.imglist_test, 'r')
im_name_list = f.readlines()
f.close()
if args.num_test is not None:
    im_name_list = im_name_list[0:args.num_test]
# devide testing images to batches
num_imgs = len(im_name_list)
start_idx = np.array(range(0,num_imgs,args.batch_size))
end_idx = start_idx + args.batch_size -1 
end_idx[-1] = num_imgs-1 if end_idx[-1] > num_imgs-1 else end_idx[-1]
eval_batch_list = []
for batch_idx in range(0,len(start_idx)):
    batch_temp = []
    for im_idx in range(start_idx[batch_idx],end_idx[batch_idx]+1):
        batch_temp.append(im_name_list[im_idx])
    eval_batch_list.append((batch_temp,batch_idx))
# parellel on batches
if len(args.gpus) > 1:
    pool = multiprocessing.Pool(len(args.gpus), initializer=build_net)
    im_scores_all = pool.map(eval_im_batch, eval_batch_list)
else:
    build_net()
    im_scores_all = map(eval_im_batch, eval_batch_list)


outputs = np.vstack(im_scores_all) 
outputs = outputs[0:num_imgs,:]
# save
np.savetxt(args.save_name, outputs, fmt='%.6f')