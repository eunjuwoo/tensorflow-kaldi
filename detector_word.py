from multiprocessing.sharedctypes import Value
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants

class detector_word:
    def __init__(self, wmax=50, _cm_threshold=10, threshold=0.7, keyword_num=6):
        self.wmax = wmax
        self._cm_threshold = _cm_threshold
        self.threshold = threshold
        self.keyword_num = keyword_num
        self.wsmooth_length = 20
        self.past_prob = np.zeros((self.keyword_num, self.wsmooth_length))
        self.keyword_prob_ring = np.zeros((self.keyword_num-2, self.wmax))
        self.proc_count = -1
        self.trigger_word_count = 0
        self.FLT_MIN = 1.175494351e-38

    def reset(self):
        self.proc_count = -1
        self.trigger_word_count = 0
        
        self.past_prob = np.zeros((self.keyword_num, self.wsmooth_length))
        self.keyword_prob_ring = np.zeros((self.keyword_num-2, self.wmax))
        return
        
    def detect(self, prob):
        self.proc_count += 1
        idx =  self.proc_count % self.wsmooth_length
        self.past_prob[0][idx] = prob[0]  # silence prob
        self.past_prob[1][idx] = prob[1]  # filler prob
        for k in range(0, self.keyword_num-2):
            self.past_prob[k+2][idx] = prob[k+2]
        
        idx_key_prob = self.proc_count % self.wmax
        for k in range(0, self.keyword_num-2):
            self.keyword_prob_ring[k][idx_key_prob] = self.posterior_smoothing_ring_buffer(self.proc_count+1, k)
        # print('{}'.format(self.keyword_prob_ring[][idx_key_prob]))
        kw_cm_score = self.calc_confidence_score_ring_buffer(self.proc_count+1)
        detected = 0
        if self.threshold < kw_cm_score:
            self.trigger_word_count += 1
            if self.trigger_word_count > self._cm_threshold and self.keyword_prob_ring[self.keyword_num-2-2][idx_key_prob] < self.keyword_prob_ring[self.keyword_num-2-1][idx_key_prob]:
                detected = 1
        else:
            detected = 0
            self.trigger_word_count = 0
        
        if detected == 0:
            return 0
        self.reset()
        return 1

    def posterior_smoothing_ring_buffer(self, frame_idx, class_idx):  #  frame_idx > 0
        accum_post_dnnprob = 0
        hsmooth_length = max(1, frame_idx - self.wsmooth_length + 1)
        hsmooth_value = 1 / (float)(frame_idx - hsmooth_length + 1)
        
        for k in range(hsmooth_length-1, frame_idx):
            accum_post_dnnprob += self.past_prob[class_idx + 2][k % self.wsmooth_length]
        return hsmooth_value * accum_post_dnnprob
    
    def calc_confidence_score_ring_buffer(self, frame_idx):
        cm_score = 1.0
        hmax = max(1, frame_idx - self.wmax +1)
        
        for i in range(0, self.keyword_num-2):
            max_class_prob = 0.0
            for k in range(hmax-1, frame_idx):
                max_class_prob = max(max_class_prob, self.keyword_prob_ring[i][k%self.wmax])
            cm_score *= max_class_prob
        if cm_score < self.FLT_MIN:
            return 0.0
        return (float)(pow(cm_score, 1.0 / (self.keyword_num-2)))
    
def load_graph(frozen_graph_fn):
    saved_model_loaded = tf.saved_model.load(frozen_graph_fn, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]   # signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"
    graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
    return graph_func

def decoder(model, test_tfr, option):
    def deserialize_example(serialized_string):
        feature_description = {
            'feature' : tf.io.FixedLenFeature([], tf.string),
            'label' : tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_example(serialized_string, feature_description)
        feature = tf.io.decode_raw(example["feature"], tf.float32)
        label = tf.io.decode_raw(example["label"], tf.float32)

        return feature, label
    
    test_ds = tf.data.TFRecordDataset(test_tfr).map(deserialize_example).batch(option['batch_size']).prefetch(option['batch_size'])
    
    detector = detector_word(wmax=option['wmax'], _cm_threshold=option['cm_threshold'], threshold=option['threshold'], keyword_num=option['keyword_num'])
    word_detect_count = 0
    detected_frame = 0 
    detection = -1
    frame_idx = 0
    for features, labels in test_ds:
        predictions = model(features)
        for i in range(len(predictions[0])):
            frame_idx += 1
            tmp_prob = list()
            for j, value in enumerate(predictions[0][i].numpy()):
                tmp_prob.append(value)
            print(tmp_prob, 'max_idx : ', tmp_prob.index(max(tmp_prob)), 'labels :', labels[i].numpy())
            detection = detector.detect(tmp_prob)
            if detection > 0:
                word_detect_count += 1
                detected_frame = frame_idx
                print('Keyword detected at {} frame. (Count : {})'.format(detected_frame, word_detect_count))
                detection = -1
    return

################################################################################################
#####                                      Main                                            #####
##### ------------------------------------------------------------------------------------ #####
#####    python3 detector_word.py                                                          #####
################################################################################################
if __name__=="__main__":
    test_TFRecord = [os.pwd()+'/tfrecord/trg_Hey_snips.tfrecord'
                     ]
    frozen_graph_dir=os.pwd()+'/exp1_Dense_Snips/model/epoch_24'
    graph = load_graph(frozen_graph_dir)

    option = {
            'wmax' : 50,
            'cm_threshold' : 10,
            'threshold' : 0.6,
            'keyword_num' : 6,
            'batch_size' : 128
            }

    decoder(model=graph, test_tfr=test_TFRecord, option=option)
