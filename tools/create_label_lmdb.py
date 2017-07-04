import matplotlib 
matplotlib.use('Agg')
from caffe.proto import caffe_pb2
import lmdb
import os, sys
import hashlib
import numpy
import scipy.io


def array_to_datum(arr, label=0):
    """Converts a 3-dimensional array to datum. If the array has dtype uint8,
    the output data will be encoded as a string. Otherwise, the output data
    will be stored in float format.
    """
    if arr.ndim != 3:
        raise ValueError('Incorrect array shape.')
    datum = caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = arr.shape
    if arr.dtype == numpy.uint8:
        datum.data = arr.tostring()
    else:
        datum.float_data.extend(arr.flat)
    datum.label = label
    return datum


if __name__ == '__main__':
    # argv[1]: label txt file
    # argv[2]: image txt file, for keys of datum
    # argv[3]: label lmdb file

    labels_array = numpy.loadtxt(sys.argv[1])
    num_sps, num_cls = labels_array.shape
    f_keys = open(sys.argv[2])
    keys_list = f_keys.readlines()
    f_keys.close()
    assert num_sps == len(keys_list)

    labels_db = lmdb.open(sys.argv[3], map_size=20 * 1024 * 1024 * 1024)
    with labels_db.begin(write=True) as txn:
        for idx in range(0,num_sps):
            # label data
            labels_cur = labels_array[idx,:]
            # while labels_cur.ndim < 3:
            #     labels_cur = numpy.expand_dims(labels_cur, axis=0)
            num_cat = labels_cur.size
            labels_cur = labels_cur.reshape(num_cat, 1, 1)
            # keys for label
            key_split = keys_list[idx].split(' ')
            key_cur = key_split[0]
            key_cur = "{:0>8d}".format(idx) + '_' + key_cur
            tag_cur = int(key_split[1])
            assert tag_cur == labels_cur.sum()
            # create datum
            labels_datum = array_to_datum(labels_cur, tag_cur)
            # write datum to lmdb
            txn.put(key_cur.encode('ascii'), labels_datum.SerializeToString())
            if (idx+1)%5000 == 0:
                print 'Processed ', idx + 1, 'files in total.'

    n_label_db = 0
    with labels_db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            n_label_db = n_label_db + 1
    print 'Total # of item in label lmdb:', n_label_db