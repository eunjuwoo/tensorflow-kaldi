#!/usr/bin/env bash
set -e -o pipefail

. ./cmd.sh
. ./path.sh

stage=5
cmd=run.pl
# options for extracting feature & label
nj=30
ali_dir=~/kaldi/egs/s5/exp/tri3_ali_train_sp
feat_dir=~/kaldi/egs/s5/data/train_sp_hires
ivector_dir=~/kaldi/egs/s5/exp/nnet3_1d/ivector_train_sp_hires

if [ $stage -le 1 ]; then
    '''
      Run the Kaldi s5_r3 baseline of TEDLIUM. 
      This step is necessary to compute features and labels later used to train the Tensorflow neural network.
      We recommend running the TEDLIUM s5_r3 recipe (not including the DNN training).
    '''
  for dn in $ali_dir $feat_dir $ivector_dir; do
    if [ ! -d $dn ]; then
      echo "$0: no such directory $dn" && exit 1;
    fi
  done
fi

if [ $stage -le 3 ]; then
    echo "$0 : extract align-to-phone for labeling (with phones.txt)"
    $cmd JOB=1:$nj ${ali_dir}/log/ali2phones.JOB.log \
      ali-to-phones --ctm-output ${ali_dir}/final.mdl \
        "ark:gunzip -c ${ali_dir}/ali.JOB.gz |" ${ali_dir}/cv05_ali.JOB.ctm || exit 1;

    echo "$0 : extract feature, LDA (40-th) + ivector (100-th)"
    $cmd JOB=1:$nj ${feat_dir}/log/feats2text.JOB.log \
      copy-feats ark:${feat_dir}/data/raw_mfcc_train_sp_hires.JOB.ark ark,t:${feat_dir}/data/raw_mfcc_train_sp_hires.JOB.txt || exit 1;
    
    $cmd JOB=1:$nj ${ivector_dir}/log/feats2text.JOB.log \
      copy-feats ark:${ivector_dir}/ivector_online.JOB.ark ark,t:${ivector_dir}/ivector_online.JOB.txt || exit 1;
fi

if [ $stage -le 4 ]; then
    echo "$0 : creating TFRecords and convert label for training trigger engine."
    '''
    ** Data Structure
    [data] - [feature] - ivector_online.*.txt
                       - raw_mfcc_train_sp_hires.*.txt
           - [align]   - cv_05_ali.*.ctm
           - [tfrecord] - train.tfrecord
                      - test.tfrecord
                      - validation.tfrecord
           - train.scp
           - test.scp
           - validation.scp
           - mk_TFRecords.py
    '''
    required_files="data/train.scp data/test.scp data/validation.scp data/mk_TFRecords.py"
    for fn in $required_files; do
      if [ ! -f $fn ]; then
        echo "$0: no such file $fn" && exit 1;
      fi
    done
    
    for fn in train test validation; do
      echo "$0 : create $fn TFRecords file."
      python3 data/mk_tfrecord.py --OPTION data/$fn
    done
fi

if [ $stage -le 5 ]; then
  train_tfr='data/tfrecord/train.tfrecord'   # train_tfr : multi input, list()
  test_tfr='data/tfrecord/test.tfrecord'     # test_tfr : single input, str
  val_tfr='data/tfrecord/validation.tfrecord'      # val_tfr : single input, str

  python3 train.py \
    --feat.train-tfr $train_tfr \
    --feat.test-tfr $test_tfr \
    --feat.validation-tfr $val_tfr \
    --trainer.continue-learning false \
    --trainer.activation-function 'relu' \
    --trainer.batch-size 128 \
    --trainer.learning-rate 0.0001 \
    --trainer.epoch 25 \
    --dir "exp1_Dense_Snips"
fi

exit 0
