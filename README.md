# 7/12/2015
This is the README to the Idiap's implementation of the i-vector
system for Kaldi. It contains information about the package, implementation
details, installation and compilation.

## General information

This implementation of the i-vector system is based on the
standard i-vector extraction procedure. It contains code to estimate
the T-matrix with the conventional EM algorithm for estimation of
Eigenvoice matrices, estimate i-vectors given the T-matrix, features
and corresponding posteriors. 

## Data structures

The classes for T-matrix and sufficient statistics are modifications
to the classes already present in Kaldi. Some irrelevant members are
removed.

The i-vector is still a kaldi::Vector and is compatible with the 
LDA and PLDA backends already available in Kaldi.

## Compilation

To compile the package simply follow the 2 steps

1. export the path to kaldi souce in the environment variable $KALDI_DIR

```
export KALDI_DIR=/home/username/kaldi-trunk/
```

2. Run make in the src/ directory

```
cd src/
make
```

Now, the binaries should have been created in the src/ivectorbin/
folder.

## Kaldi recipe

The recipe equivalent to the kaldi recipe to train and
test a speaker recognition system for NIST SRE 2008 dataset
is available in the scripts folder. The file scripts/run.sh
is the main recipe that calls other scripts from within. 

## References

The implementation is based on the i-vector systems in 

[1] Glembek, Ondřej, et al. "Simplification and optimization of i-vector extraction." Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International Conference on. IEEE, 2011.

[2] Madikeri, Srikanth. "A hybrid factor analysis and probabilistic pca-based system for dictionary learning and encoding for robust speaker recognition." Odyssey Workshop. 2012.
