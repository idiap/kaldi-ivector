// ivector/conv-ivector-extractor.h

// Copyright 2013    Daniel Povey
// Copyright 2015    Srikanth Madikeri (Idiap Research Institute)


// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_IVECTOR_IVECTOR_EXTRACTOR_H_
#define KALDI_IVECTOR_IVECTOR_EXTRACTOR_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "thread/kaldi-mutex.h"
#include "hmm/posterior.h"
#include "ivector/ivector-extractor.h"

namespace kaldi {

// Note, throughout this file we use SGMM-type notation because
// that's what I'm comfortable with.
// Dimensions:
//  D is the feature dim (e.g. D = 60)
//  I is the number of Gaussians (e.g. I = 2048)
//  S is the ivector dim (e.g. S = 400)

struct IvectorExtractorConvUtteranceStats {
  IvectorExtractorConvUtteranceStats(int32 num_gauss, int32 feat_dim,
                                 bool need_2nd_order_stats):
      gamma(num_gauss), X(num_gauss, feat_dim) {
    if (need_2nd_order_stats) {
      S.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++) S[i].Resize(feat_dim);
    }
  }

  void Scale(double scale) { // Used to apply acoustic scale.
    gamma.Scale(scale);
    X.Scale(scale);
    for (size_t i = 0; i < S.size(); i++) S[i].Scale(scale);
  }
  Vector<double> gamma; // zeroth-order stats (summed posteriors), dimension [I]
  Matrix<double> X; // first-order stats, dimension [I][D]
  std::vector<SpMatrix<double> > S; // 2nd-order stats, dimension [I][D][D], if
                                    // required.
};


// Options for estimating iVectors, during both training and test.  Note: the
// "acoustic_weight" is not read by any class declared in this header; it has to
// be applied by calling IvectorExtractorUtteranceStats::Scale() before
// obtaining the iVector.
struct IvectorConvEstimationOptions {
  // TODO: not really required now; try removing this later
  double acoustic_weight;
  IvectorConvEstimationOptions(): acoustic_weight(1.0) {}
  void Register(OptionsItf *po) {
    po->Register("acoustic-weight", &acoustic_weight,
                 "Weight on part of auxf that involves the data (e.g. 0.2); "
                 "if this weight is small, the prior will have more effect.");
  }
};


class IvectorExtractorConv;

struct IvectorExtractorConvOptions {
  int ivector_dim;
  IvectorExtractorConvOptions(): ivector_dim(400) { }
  void Register(OptionsItf *po) {
    po->Register("ivector-dim", &ivector_dim, "the subspace dimension");
  }
};


// Caution: the IvectorExtractor is not the only thing required
// to get an ivector.  We also need to get posteriors from a
// FullGmm.  Typically these will be obtained in a process that involves
// using a DiagGmm for Gaussian selection, followed by getting
// posteriors from the FullGmm.  To keep track of these, we keep
// them all in the same directory, e.g. final.{ubm,dubm,ie}

class IvectorExtractorConv {

 public:
  friend class IvectorConvStats;

  IvectorExtractorConv() { }
  
  IvectorExtractorConv(bool online_) {online = online_; diaginit = false; }

  IvectorExtractorConv(
      const IvectorExtractorConvOptions &opts,
      const FullGmm &fgmm,
      bool online_ = false);

  /// Gets the distribution over ivectors (or at least, a Gaussian approximation
  /// to it).  The output "var" may be NULL if you don't need it.  "mean", and
  /// "var", if present, must be the correct dimension (this->IvectorDim()).
  /// If you only need a point estimate of the iVector, get the mean only.
  void GetIvectorDistribution(
      const IvectorExtractorConvUtteranceStats &utt_stats,
      VectorBase<double> *mean,
      SpMatrix<double> *var) const;

  void GetIvectorDistribution(
      const Vector<double> fostats,
      const Vector<double> bias,
      VectorBase<double> *mean,
      const SpMatrix<double> &var) const;
  
  /// Gets the linear and quadratic terms in the distribution over iVectors, but
  /// only the terms arising from the Gaussian means (i.e. not the weights
  /// or the priors).
  /// Setup is log p(x) \propto x^T linear -0.5 x^T quadratic x.
  /// This function *adds to* the output rather than setting it.
  void GetIvectorDistMean(
      const IvectorExtractorConvUtteranceStats &utt_stats,
      VectorBase<double> *linear,
      SpMatrix<double> *quadratic) const;

  /// Adds to "stats", which are the zeroth and 1st-order
  /// stats (they must be the correct size).
  void GetStats(const MatrixBase<BaseFloat> &feats,
                const Posterior &post,
                IvectorExtractorConvUtteranceStats *stats) const;

  void GetStats(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &post,
    IvectorExtractorConvUtteranceStats *stats) const; 


  void RemoveMeans(IvectorExtractorConvUtteranceStats *stats);

  void GetSuperVector(const Vector<double> &ivec, 
                                   Vector<BaseFloat> &svec);
  int32 FeatDim() const;
  int32 IvectorDim() const;
  int32 NumGauss() const;
  bool online;
void UpdateProjections(const Matrix<BaseFloat> &R_, Matrix<BaseFloat> &Y_, int32 idx);
  
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Note: we allow the default assignment and copy operators
  // because they do what we want.
 protected:
  void ComputeDerivedVars();

  // Imagine we'll project the iVectors with transformation T, so apply T^{-1}
  // where necessary to keep the model equivalent.  Used to keep unit variance
  // (like prior re-estimation).
  void TransformIvectors(const MatrixBase<double> &T,
                         double new_ivector_offset);
  
  
  /// Weight projection vectors, if used.  Dimension is [I][S]
  // TODO: check if w_ or w_vec_ is reqd.
  // Matrix<double> w_;

  /// If we are not using weight-projection vectors, stores the Gaussian mixture
  /// weights from the UBM.  This does not affect the iVector; it is only useful
  /// as a way of making sure the log-probs are comparable between systems with
  /// and without weight projection matrices.
  Vector<double> w_vec_;
  
  /// Ivector-subspace projection matrices, dimension is [I][D][S].
  /// The I'th matrix projects from ivector-space to Gaussian mean.
  /// There is no mean offset to add-- we deal with it by having
  /// a prior with a nonzero mean.
  std::vector<Matrix<double> > M_;

  /// Means and Inverse variances of speaker-adapted model, dimension [I][D][D].
  Matrix<double> Means_;
  std::vector<SpMatrix<double> > Sigma_inv_;
  std::vector<Vector<double> > Sigma_inv_d_;
  bool diaginit;
  
  /// 1st dim of the prior over the ivector has an offset, so it is not zero.
  /// This is used to handle the global offset of the speaker-adapted means in a
  /// simple way.

  // Below are *derived variables* that can be computed from the
  // variables above.

  /// The constant term in the log-likelihood of each Gaussian (not
  /// counting any weight).
  Vector<double> gconsts_;
  
  /// U_i = M_i^T \Sigma_i^{-1} M_i is a quantity that comes up
  /// in ivector estimation.  This is conceptually a
  /// std::vector<SpMatrix<double> >, but we store the packed-data 
  /// in the rows of a matrix, which gives us an efficiency 
  /// improvement (we can use matrix-multiplies).
  Matrix<double> U_;
 private:
  // var <-- quadratic_term^{-1}, but done carefully, first flooring eigenvalues
  // of quadratic_term to 1.0, which mathematically is the least they can be,
  // due to the prior term.
  static void InvertWithFlooring(const SpMatrix<double> &quadratic_term,
                                 SpMatrix<double> *var);  
};



/// Options for training the IvectorExtractor, e.g. variance flooring.
struct IvectorExtractorConvEstimationOptions {
  double variance_floor_factor;
  double gaussian_min_count;
  int32 num_threads;
  IvectorExtractorConvEstimationOptions(): variance_floor_factor(0.1),
                                       gaussian_min_count(100.0),
                                       num_threads(1) { }
  void Register(OptionsItf *po) {
    po->Register("variance-floor-factor", &variance_floor_factor,
                 "Factor that determines variance flooring (we floor each covar "
                 "to this times global average covariance");
    po->Register("gaussian-min-count", &gaussian_min_count,
                 "Minimum total count per Gaussian, below which we refuse to "
                 "update any associated parameters.");
    po->Register("num-threads", &num_threads,
                 "Number of threads used in iVector estimation program");
  }
};

/// IvectorStats is a class used to update the parameters of the ivector estimator.
class IvectorConvStats {
 public:
  friend class IvectorExtractorConv;

  IvectorConvStats(): num_ivectors_(0) { }
  
  IvectorConvStats(const IvectorExtractorConv &extractor);
  
  void Add(const IvectorConvStats &other);
  
  void AccStatsForUtterance(const IvectorExtractorConv &extractor,
                            const MatrixBase<BaseFloat> &feats,
                            const Posterior &post);

  // This version (intended mainly for testing) works out the Gaussian
  // posteriors from the model.  Returns total log-like for feats, given
  // unadapted fgmm.  You'd want to add Gaussian pruning and preselection using
  // the diagonal, GMM, for speed, if you used this outside testing code.
  double AccStatsForUtterance(const IvectorExtractorConv &extractor,
                              const MatrixBase<BaseFloat> &feats,
                              const FullGmm &fgmm);
  void AccStatsForUtterance(
      const IvectorExtractorConv &extractor,
      const MatrixBase<BaseFloat> &feats,
      const MatrixBase<BaseFloat> &posteriors); 

  void AccStatsForUtterance(
      const IvectorExtractorConv &extractor,
      const IvectorExtractorConvUtteranceStats utt_stats); 

  void AccStatsForUtterance(
      const IvectorExtractorConv &extractor,
      const MatrixBase<BaseFloat> &feats,
      const Posterior &post,
      const MatrixBase<BaseFloat> &bias);
  void Read(std::istream &is, bool binary, bool add = false);
  void RemoveMeans(IvectorExtractorConvUtteranceStats *stats,
                   const MatrixBase<BaseFloat> &means_);

  void Write(std::ostream &os, bool binary) const;

  /// Returns the objf improvement per frame.
  double Update(const IvectorExtractorConvEstimationOptions &opts,
                IvectorExtractorConv *extractor) const;

  
  // Note: we allow the default assignment and copy operators
  // because they do what we want.
 protected:

  
  // This is called by AccStatsForUtterance
  void CommitStatsForUtterance(const IvectorExtractorConv &extractor,
                               const IvectorExtractorConvUtteranceStats &utt_stats);

  /// This is called by CommitStatsForUtterance.  We commit the stats
  /// used to update the T matrix.
  void CommitStatsForM(const IvectorExtractorConv &extractor,
                       const IvectorExtractorConvUtteranceStats &utt_stats,
                       const VectorBase<double> &ivec_mean,
                       const SpMatrix<double> &ivec_var);

  /// Commit the stats used to update the variance.
  void CommitStatsForSigma(const IvectorExtractorConv &extractor,
                           const IvectorExtractorConvUtteranceStats &utt_stats);

  
  /// Commit the stats used to update the prior distribution.
  void CommitStatsForPrior(const VectorBase<double> &ivec_mean,
                           const SpMatrix<double> &ivec_var);
  
  // Updates M.  Returns the objf improvement per frame.
  double UpdateProjections(const IvectorExtractorConvEstimationOptions &opts,
                     IvectorExtractorConv *extractor) const;

  // This internally called function returns the objf improvement
  // for this Gaussian index.  Updates one M.
  double UpdateProjection(const IvectorExtractorConvEstimationOptions &opts,
                          int32 gaussian,
                          IvectorExtractorConv *extractor) const;

  void CheckDims(const IvectorExtractorConv &extractor) const;

  /// Total auxiliary function over the training data-- can be
  /// used to check convergence, etc.

  /// This mutex guards gamma_, Y_ and R_ (for multi-threaded
  /// update)
  Mutex subspace_stats_lock_; 
  
  /// Total occupation count for each Gaussian index (zeroth-order stats)
  Vector<double> gamma_;
  
  /// Stats Y_i for estimating projections M.  Dimension is [I][D][S].  The
  /// linear term in M.
  std::vector<Matrix<double> > Y_;
  
  /// R_i, quadratic term for ivector subspace (M matrix)estimation.  This is a
  /// kind of scatter of ivectors of training speakers, weighted by count for
  /// each Gaussian.  Conceptually vector<SpMatrix<double> >, but we store each
  /// SpMatrix as a row of R_.  Conceptually, the dim is [I][S][S]; the actual
  /// dim is [I][S*(S+1)/2].
  Matrix<double> R_;

  /// This mutex guards Q_ and G_ (for multi-threaded update)
  Mutex weight_stats_lock_;
  
  /// Q_ is like R_ (with same dimensions), except used for weight estimation;
  /// the scatter of ivectors is weighted by the coefficient of the quadratic
  /// term in the expansion for w (the "safe" one, with the max expression).
  Matrix<double> Q_;

  /// G_ is the linear term in the weight projection matrix w_.  It has the same
  /// dim as w_, i.e. [I][S]
  Matrix<double> G_;

  /// This mutex guards S_ (for multi-threaded update)
  Mutex variance_stats_lock_;

  /// S_{i}, raw second-order stats per Gaussian which we will use to update the
  /// variances Sigma_inv_.
  std::vector< SpMatrix<double> > S_;


  /// This mutex guards num_ivectors_, ivector_sum_ and ivector_scatter_ (for multi-threaded
  /// update)
  Mutex prior_stats_lock_;

  /// Count of the number of iVectors we trained on.   Need for prior re-estimation.
  /// (make it double not int64 to more easily support weighting later.)
  double num_ivectors_;
  
  /// Sum of all the iVector means.  Needed for prior re-estimation.
  Vector<double> ivector_sum_;

  /// Second-order stats for the iVectors.  Needed for prior re-estimation.
  SpMatrix<double> ivector_scatter_;
};



}  // namespace kaldi


#endif


