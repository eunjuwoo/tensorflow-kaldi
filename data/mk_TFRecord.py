from pyparsing import Or
import os, glob, argparse
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import tensorflow as tf

class convertState:
    def __init__(self, trgnum=4, PHON_TRG=dict(), PHON_SIL=[]):
        '''
            class 0 : silence phone
            class 1 : etc phone
            class 2 ~ : trigger phone
        '''
        self.TRG_NUM=trgnum
        self.PHON_SIL=[1, 2, 3, 4, 5]
        # 명령어 : hey snips
        # HH_B(96) EY_I(86) S_I(150) N_I(126) IH_I(102) P_I(142) S_E(149)
        self.PHON_TRG={ '2': [96, 86],
                        '3': [150],
                        '4': [126, 102, 142],
                        '5': [149]}
        if self.TRG_NUM != self.PHON_TRG.__len__():
            print('[ERROR] : Check the number of TRG_NUM.')
            exit(1)

    def convertLabel(self, state_data=[]):
        label = []
        for idx, lab in enumerate(state_data):
            tmp_label = '1000'      # temp value
            if int(lab) in self.PHON_SIL:
                tmp_label = '0'
                label.append(tmp_label)
                continue
            for trg in self.PHON_TRG.keys():
                if int(lab) in self.PHON_TRG[trg]:
                    tmp_label = trg
                    label.append(tmp_label)
                    break
            if tmp_label == '1000':
                label.append('1')
        return label

class createTFrecord:
    def __init__(self, res=OrderedDict(), num_class=4, feat_dim=140, tfr_name=''):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.res = res
        self.tfr_name = tfr_name
        self.ctx_L = 12
        self.ctx_R = 12
        
    def write_record(self):
        cs = convertState(trgnum=self.num_class)
        writer = tf.io.TFRecordWriter(os.getcwd()+'/tfrecord/'+self.tfr_name+'.tfrecord')
        print('\n=====+=====+=====+====+=====+=====+=====+==== [Write TFRecord] =====+=====+=====+====+=====+=====+=====+==== ')
        for fid in tqdm(self.res.keys()):
            if not len(self.res[fid]['feature1']) == len(self.res[fid]['feature2']) == len(self.res[fid]['align']):
                continue
            if len(self.res[fid]['feature1']) == 0 or len(self.res[fid]['feature2']) == 0 or len(self.res[fid]['align']) == 0:
                print('[WARNING] : pass {}'.format(fid))
                continue
            feat = [self.res[fid]['feature1'][idx] + self.res[fid]['feature2'][idx] for idx in range(len(self.res[fid]['feature1']))]
            label = np.array([int(i) for i in cs.convertLabel(self.res[fid]['align'])], dtype=np.int64)
            
            data_padded = np.pad(feat, ((self.ctx_L, self.ctx_R), (0,0)), 'constant')
            x = list()
            for idx in range(self.ctx_L, len(feat)+self.ctx_R):
                x.append(data_padded[idx-self.ctx_L : idx+self.ctx_R+1].reshape(-1))
            for idx in range(len(label)):
                example = self.serialize_example(x[idx], np.array([label[idx]]))
                writer.write(example)
        writer.close()
        return
    
    def read_record(self, tfrecord_fn=''):
        def deserialize_example(serialized_string):
            feature_description = {
                'feature' : tf.io.FixedLenFeature([], tf.string),
                'label' : tf.io.FixedLenFeature([], tf.string),
                }
            example = tf.io.parse_example(serialized_string, feature_description)
            feature = tf.io.decode_raw(example["feature"], tf.float32)
            label = tf.io.decode_raw(example["label"], tf.float32)

            return feature, label
    
        dataset = tf.data.TFRecordDataset(tfrecord_fn).map(deserialize_example).batch(4)
        for x in dataset:
            print(x)
            break
        return
       
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def serialize_example(self, feat, label):
        feat_binary = feat.astype(np.float32).reshape(-1, self.feat_dim*(self.ctx_L+self.ctx_R+1)).tobytes()
         ## label one-hot encoding ==> tobytes() 
        label_binary = np.eye(self.num_class+2).astype(np.float32)[label]
        label_binary = label_binary.tobytes()
        feat_list = self._bytes_feature(feat_binary)
        label_list = self._bytes_feature(label_binary)
        proto = tf.train.Example(features=tf.train.Features(feature={
            "feature": feat_list,
            "label": label_list
        }))
        return proto.SerializeToString()

def get_args():
    parser=argparse.ArgumentParser(description='')
    
    parser.add_argument('--OPTION', required=True, default='', help='')
    args=parser.parse_args()
    return args

def extract_feature(fn, feat_dim=0, feature=''):
    fid='none'
    for ln in open(fn):
        if ln.find('[') > -1 and ln.replace('[','').strip() in res.keys(): fid=ln.replace('[','').strip()
        else: 
            if fid=='none': continue     
            feats=ln.replace(']', '').strip().split(' ')
            if len(feats)!=feat_dim:
                print('[WARNING] : feature dimesion is not matched. Current feature dimension : {}'.format(len(feats)))
            if feature=='mfcc_LDA': res[fid]['feature1'].append(feats)
            elif feature=='i-vector': res[fid]['feature2'].append(feats)
            else: print('[ERROR] : It is unknown feature. please check option.')
            if ln.find(']') > -1: fid='none'
    return

def extract_label(fn):
    for ln in open(fn):
        wns = ln.strip().split(' ')
        if not wns[0] in res.keys():
            continue 
        start_t = int(float(wns[2])*100)
        end_t = start_t + int(float(wns[3])*100)
        state=wns[4]
        for idx in range(start_t, end_t):
            res[wns[0]]['align'].append(state)
    return


################################################################################################
#####                                      Main                                            #####
##### ------------------------------------------------------------------------------------ #####
#####    python3 mk_TFRecord.py --OPTION [train/test/validation]                           #####
################################################################################################

args = get_args()
option = args.OPTION  # choose [train / test / validation]

if not option in {'train', 'test', 'validation'}:
    print('[ERROR] : Check your option(train/test/validation). ')
    exit(1)
else:
    print('[INFO] : selected OPTION : ', option)

## 사용할 wav list 먼저 정하고
res = OrderedDict()
for ln in open(os.getcwd()+'/'+option+'.scp'):
    fid = ln.strip().split(' ')[0]
    res[fid]=OrderedDict({'feature1':list(), 'feature2':list(), 'align':list()})
    
## feature
feat_list = [file for file in glob.glob(os.getcwd()+'/feature/*') if file.endswith('.txt')]
print('=====+=====+=====+====+=====+=====+=====+====  [Processing feature] =====+=====+=====+====+=====+=====+=====+====  ')
for fn in tqdm(sorted(feat_list)):
    if fn.find('_mfcc_') > -1:
        extract_feature(fn, feat_dim=40, feature='mfcc_LDA')
    else:
        extract_feature(fn, feat_dim=100, feature='i-vector')
## align
algt_list = [file for file in glob.glob(os.getcwd()+'/align/*') if file.endswith('.ctm')]
print('\n=====+=====+=====+====+=====+=====+=====+==== [Processing align] =====+=====+=====+====+=====+=====+=====+==== ')
for fn in tqdm(sorted(algt_list)):
    extract_label(fn)
    
## write tf-record file.
createTf = createTFrecord(res=res, num_class=4, feat_dim=140, tfr_name=option)
createTf.write_record()

## read tf-record file
createTf.read_record(tfrecord_fn=os.getcwd()+'/tfrecord/'+option+'.tfrecord')
