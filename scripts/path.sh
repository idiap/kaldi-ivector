if [ -z "$KALDI_DIR" ]; then
    echo "WARNING: KALDI_DIR not set. You won't be able to run the scripts"
    exit 1
else
KALDI_ROOT="$KALDI_DIR"
    export PATH=$PWD/utils/:$PWD/../src/ivectorbin/:$KALDI_ROOT/egs/sre08/v1/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
fi
export LC_ALL=C
