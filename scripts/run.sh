#!/bin/bash
# Copyright 2013   Daniel Povey
#           2015   Srikanth Madikeri (Idiap Research Institute)
#                  Subhadeep Dey (Idiap Research Institute)
# Apache 2.0.
#

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# sample directory paths. These must be point to proper
# directories for the script to work
fisherpart1dir=databases/fe_03_p1_sph
fisherpart2dir=databases/Fisher_English_Training_Part2
sre2005dir=databases/SRE05
sre2006dir=database/SRE06
sre2004dir=databases/SRE04
sre2008dir=databases/SRE08
swbc1dir=databases/swbc1    # Switchboard cellular Part 1
swbc2dir=databases/swbc2    # Switchboard cellular Part 2


bash local/make_fisher.sh $fisherpart1dir/media/audio/ /idiap/resource/database data/fisher1
local/make_fisher.sh $fisherpart2dir/media/ /idiap/home/msrikanth/temp/sre/flists/kaldi/fisherp2/trans/ data/fisher2

# 2005
# what it does: looks for  data/local/sre05-key-v7b.txt. if that doesn't exist, this
# is downloaded from Povey's webpage. For every test case, $path_to_2005_data/$test.sph
# should exist. This is gender independent. All train files are missed
local/make_sre_2005_test.pl $sre2005dir/r101_1_1/test/ data/

# 2004
local/make_sre_2004_test.pl  $sre2004dir/test/ data/sre_2004_1

# 2008
#alias find="find -L"
local/make_sre_2008_train.pl $sre2008dir/train data

# 2008 test
local/make_sre_2008_test.sh  $sre2008dir/test data

# 2006 train
local/make_sre_2006_train.pl $sre2006dir/ data

# 2005 train
# modified path from data/speech to data/data in the script
 local/make_sre_2005_train.pl $sre2005dir/train data

# swb cell 1
local/make_swbd_cellular1.pl $swbc1dir/data  data/swbd_cellular1_train
# swb cell 2
local/make_swbd_cellular2.pl $swbc2dir/      data/swbd_cellular2_train

utils/combine_data.sh \
    data/train \
    data/fisher1 data/fisher2 \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/sre05_train_3conv4w_female data/sre05_train_8conv4w_female \
    data/sre06_train_3conv4w_female data/sre06_train_8conv4w_female \
    data/sre05_train_3conv4w_male data/sre05_train_8conv4w_male \
    data/sre06_train_3conv4w_male data/sre06_train_8conv4w_male \
    data/sre_2004_1/ data/sre_2004_2/ data/sre05_test

grep -w m data/train/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/train data/train_male
grep -w f data/train/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/train data/train_female
rm foo


mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

set -e
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "queue.pl -l q1d"       data/train                      exp/make_mfcc mfcc 
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "queue.pl -l q1d"       data/sre08_train_short2_female  exp/make_mfcc mfcc
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "queue.pl -l q1d"       data/sre08_train_short2_male    exp/make_mfcc mfcc 
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "queue.pl -l q1d"       data/sre08_test_short3_female   exp/make_mfcc mfcc 
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "queue.pl -l q1d"       data/sre08_test_short3_male     exp/make_mfcc mfcc


sid/compute_vad_decision.sh --nj 4 --cmd "queue.pl  -l  q1d"           data/train                       exp/make_mfcc    mfcc
sid/compute_vad_decision.sh --nj 40 --cmd "queue.pl -l  q1d"           data/train                       exp/make_vad     mfcc
sid/compute_vad_decision.sh --nj 40 --cmd "queue.pl -l  q1d"           data/sre08_train_short2_female   exp/make_vad     mfcc
sid/compute_vad_decision.sh --nj 40 --cmd "queue.pl -l  q1d"           data/sre08_train_short2_male     exp/make_vad     mfcc
sid/compute_vad_decision.sh --nj 40 --cmd "queue.pl -l  q1d"           data/sre08_test_short3_female    exp/make_vad     mfcc
sid/compute_vad_decision.sh --nj 40 --cmd "queue.pl -l  q1d"           data/sre08_test_short3_male      exp/make_vad     mfcc


# Note: to see the proportion of voiced frames you can do,
# grep Prop exp/make_vad/vad_*.1.log 

# Get male and female subsets of training data.
grep -w m data/fisher/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/fisher data/fisher_male
grep -w f data/fisher/spk2gender | awk '{print $1}' > foo;
utils/subset_data_dir.sh --spk-list foo data/fisher data/fisher_female
rm foo

sid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/fisher_2k 2048 exp/diag_ubm_2048

sid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/fisher_4k exp/diag_ubm_2048 exp/full_ubm_2048

# Get male and female versions of the UBM in one pass; make sure not to remove
# any Gaussians due to low counts (so they stay matched).  This will be more convenient
# for gender-id.
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false --num-iters 1 --cmd "$train_cmd" \
   data/fisher_male_4k exp/full_ubm_2048 exp/full_ubm_2048_male &
sid/train_full_ubm.sh --nj 30 --remove-low-count-gaussians false --num-iters 1 --cmd "$train_cmd" \
   data/fisher_female_4k exp/full_ubm_2048 exp/full_ubm_2048_female &
wait

sid/train_ivector_extractor_conv.sh --nj 100 --cmd "$train_cmd" exp/full_ubm_2048_male/final.ubm exp/train_male exp/extractor_2048_full_ubm_male
sid/train_ivector_extractor_conv.sh --nj 100 --cmd "$train_cmd" exp/full_ubm_2048_female/final.ubm exp/train_female exp/extractor_2048_full_ubm_female


#### Extract I-vectors  for development, enrollment and test data
sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_male data/train_male exp/ivectors_male_train
sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_male data/sre08_train_short2_male exp/ivectors_sre08_train_short2_male
sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_male data/sre08_test_short3_male exp/ivectors_sre08_test_short3_male

sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_female data/train_female exp/ivectors_female_train
sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_female data/sre08_train_short2_female exp/ivectors_sre08_train_short2_female
sid/extract_ivectors_conv.sh --cmd "$train_cmd"  --nj 100  exp/extractor_2048_full_ubm_female data/sre08_test_short3_female exp/ivectors_sre08_test_short3_female

ivector-compute-lda --dim=150  --total-covariance-factor=0.1   'ark:ivector-normalize-length scp:exp/ivectors_male_train/ivector.scp    ark:- |'     ark:data/train_male/utt2spk    exp/ivectors_male_train/lda_transform.mat

ivector-compute-plda ark:data/train_male/spk2utt  'ark:ivector-transform    exp/ivectors_male_train/lda_transform.mat   scp:exp/ivectors_male_train/ivector.scp   ark:- | ivector-normalize-length  ark:- ark:- |'  exp/ivectors_male_train/plda_matrix

trials=data/sre08_trials/short2-short3-male.trials

ivector-plda-scoring --normalize-length=true    exp/ivectors_male_train/plda_matrix    'ark:ivector-transform  exp/ivectors_male_train/lda_transform.mat  scp:exp/ivectors_sre08_train_short2_male/spk_ivector.scp  ark:- |ivector-normalize-length  ark:- ark:- |'  'ark:ivector-transform   exp/ivectors_male_train/lda_transform.mat    scp:exp/ivectors_sre08_test_short3_male/ivector.scp  ark:- |  ivector-normalize-length  ark:- ark:- |'   "cat '$trials' | awk '{print \$1, \$2}'|"  foo_male_2k_400_dim
local/score_sre08.sh  $trials foo_male_2k_400_dim

#######################################################################################
### Results for male test
### Number of utterances = 39090 
### Number of speakers =  8643

##Scoring against data/sre08_trials/short2-short3-male.trials
##  Condition:      0      1      2      3      4      5      6      7      8
##        EER:   9.90   7.45   1.21   7.57   7.52   6.25   5.95   4.10   3.07
####################################################################################


ivector-compute-lda --dim=150  --total-covariance-factor=0.1   'ark:ivector-normalize-length scp:exp/ivectors_female_train/ivector.scp    ark:- |'     ark:data/train_female/utt2spk    exp/ivectors_female_train/lda_transform.mat

ivector-compute-plda ark:data/train_female/spk2utt  'ark:ivector-transform    exp/ivectors_female_train/lda_transform.mat   scp:exp/ivectors_female_train/ivector.scp   ark:- | ivector-normalize-length  ark:- ark:- |'  exp/ivectors_female_train/plda_matrix

trials=data/sre08_trials/short2-short3-female.trials

ivector-plda-scoring --normalize-length=true    exp/ivectors_female_train/plda_matrix    'ark:ivector-transform  exp/ivectors_female_train/lda_transform.mat  scp:exp/ivectors_sre08_train_short2_female/spk_ivector.scp  ark:- |ivector-normalize-length  ark:- ark:- |'  'ark:ivector-transform   exp/ivectors_female_train/lda_transform.mat    scp:exp/ivectors_sre08_test_short3_female/ivector.scp  ark:- |  ivector-normalize-length  ark:- ark:- |'   "cat '$trials' | awk '{print \$1, \$2}'|"  foo_female_2k_400_dim

local/score_sre08.sh  $trials foo_female_2k_400_dim
#######################################################################################
### Results for female test
### Number of utterances = 52079 
### Number of speakers =  12117
##  Scoring against data/sre08_trials/short2-short3-female.trials
##  Condition:      0      1      2      3      4      5      6      7      8
##        EER:  10.82   8.18   0.90   8.14   9.31   8.89   8.26   5.20   5.26

####################################################################################

