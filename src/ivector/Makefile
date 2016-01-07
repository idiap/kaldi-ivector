
all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
EXTRA_CXXFLAGS = -I${KALDI_DIR}/src

include ${KALDI_DIR}/src/kaldi.mk

TESTFILES = 

OBJFILES = conv-ivector-extractor.o 

LIBNAME = kaldi-ivector-conv

ADDLIBS = ${KALDI_DIR}/src/gmm/kaldi-gmm.a ${KALDI_DIR}/src/tree/kaldi-tree.a ${KALDI_DIR}/src/transform/kaldi-transform.a \
		${KALDI_DIR}/src/thread/kaldi-thread.a ${KALDI_DIR}/src/matrix/kaldi-matrix.a ${KALDI_DIR}/src/base/kaldi-base.a \
        ${KALDI_DIR}/src/util/kaldi-util.a 

include ${KALDI_DIR}/src/makefiles/default_rules.mk
