
all:

EXTRA_CXXFLAGS = -Wno-sign-compare -I${KALDI_DIR}/src
include ${KALDI_DIR}/src/kaldi.mk

BINFILES = ivector-extractor-conv-init ivector-extractor-conv-acc-stats \
		   ivector-extractor-conv-sum-acc ivector-extractor-conv-est \
		   ivector-extract-conv \
		   copy-tmat-conv 

OBJFILES =

TESTFILES =

ADDLIBS = ../ivector/kaldi-ivector-conv.a ${KALDI_DIR}/src/ivector/kaldi-ivector.a ${KALDI_DIR}/src/hmm/kaldi-hmm.a ${KALDI_DIR}/src/gmm/kaldi-gmm.a \
    ${KALDI_DIR}/src/tree/kaldi-tree.a ${KALDI_DIR}/src/thread/kaldi-thread.a ${KALDI_DIR}/src/matrix/kaldi-matrix.a \
    ${KALDI_DIR}/src/util/kaldi-util.a ${KALDI_DIR}/src/base/kaldi-base.a 

include ${KALDI_DIR}/src/makefiles/default_rules.mk
