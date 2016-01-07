// ivector/ivector-extractor.cc

// Copyright 2013     Daniel Povey
// Copyright 2015     Srikanth Madikeri (Idiap Research Institute)

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

#include <vector>

#include "ivector/conv-ivector-extractor.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi {

int32 IvectorExtractorConv::FeatDim() const {
  KALDI_ASSERT(!M_.empty());
  return M_[0].NumRows();
}
int32 IvectorExtractorConv::IvectorDim() const {
  KALDI_ASSERT(!M_.empty());
  return M_[0].NumCols();
}
int32 IvectorExtractorConv::NumGauss() const {
  return static_cast<int32>(M_.size());
}


void IvectorExtractorConv::GetSuperVector(const Vector<double> &ivec, 
                                   Vector<BaseFloat> &svec) 
{
    int32 S = IvectorDim(),
        M = NumGauss(),
        D = FeatDim();
    int32 MD = M*D;
    if(S != ivec.Dim()) {
        KALDI_ERR << "Mismatching dimensions between i-vector and T-matrix";        
    }
    if(MD != svec.Dim())
        svec.Resize(MD);
    Matrix<double> svec_mat(M,D);
    Vector<double> ivecdouble(ivec);
    for(int32 i=0; i<M; i++) {
        SubVector<double> row(svec_mat, i);
        row.AddMatVec(1.0, M_[i], kNoTrans, ivecdouble, 0.0);
    }
    svec.CopyRowsFromMat(svec_mat);
}

// including another UpdateProjections so that matrix can be
// updated when R_ and Y_ are computed elsewhere

void IvectorExtractorConv::UpdateProjections(const Matrix<BaseFloat> &R_, Matrix<BaseFloat> &Y_, int32 idx) {
        int32 S = IvectorDim();
        if(idx >= S)
            return;
        SpMatrix<double> R(S, kUndefined);
        SubVector<float> R_vec(R_, idx); // i'th row of R; vectorized form of SpMatrix.
        SubVector<double> R_sp(R.Data(), S * (S+1) / 2);
        R_sp.CopyFromVec(R_vec); // copy to SpMatrix's memory.
        Matrix<double> M(M_[idx]);
        SolverOptions solver_opts;
        solver_opts.name = "M";
        solver_opts.diagonal_precondition = true;
        SolveQuadraticMatrixProblem(R, Matrix<double>(Y_), Sigma_inv_[idx], solver_opts, &M);
        M_[idx].CopyFromMat(M);
        if (idx<=1)
            KALDI_LOG << "First element of idx "<< idx << " " << M_[idx](0,0);
}

void IvectorExtractorConv::RemoveMeans(IvectorExtractorConvUtteranceStats *stats) {
        int32 feat_dim = FeatDim();
        int32 num_gauss = NumGauss();
        KALDI_ASSERT(stats->X.NumCols() == feat_dim);
        KALDI_ASSERT(stats->X.NumRows() == num_gauss);
        for (int32 i = 0; i < num_gauss; i++) {
            stats->X.Row(i).AddVec(-stats->gamma(i), Means_.Row(i));
        }

}

void IvectorExtractorConv::GetStats(
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &post,
    IvectorExtractorConvUtteranceStats *stats) const {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  int32 num_frames = feats.NumRows(), num_gauss = NumGauss(),
      feat_dim = FeatDim();
  KALDI_ASSERT(feats.NumCols() == feat_dim);
  KALDI_ASSERT(stats->gamma.Dim() == num_gauss &&
               stats->X.NumCols() == feat_dim);
  bool update_variance = (!stats->S.empty());
  
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    SpMatrix<double> outer_prod;
    if (update_variance) {
      outer_prod.Resize(feat_dim);
      outer_prod.AddVec2(1.0, frame);
    }
    for (int32 i = 0; i < num_gauss; i++) {
      double weight = post(t,i);
      stats->gamma(i) += weight;
      stats->X.Row(i).AddVec(weight, frame);
      if (update_variance)
        stats->S[i].AddSp(weight, outer_prod);
    }
  }

  for (int32 i = 0; i < num_gauss; i++) {
      stats->X.Row(i).AddVec(-stats->gamma(i), Means_.Row(i));
  }
}

void IvectorExtractorConv::GetStats(
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    IvectorExtractorConvUtteranceStats *stats) const {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  int32 num_frames = feats.NumRows(), num_gauss = NumGauss(),
      feat_dim = FeatDim();
  KALDI_ASSERT(feats.NumCols() == feat_dim);
  KALDI_ASSERT(stats->gamma.Dim() == num_gauss &&
               stats->X.NumCols() == feat_dim);
  bool update_variance = (!stats->S.empty());
  
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    const VecType &this_post(post[t]);
    SpMatrix<double> outer_prod;
    if (update_variance) {
      outer_prod.Resize(feat_dim);
      outer_prod.AddVec2(1.0, frame);
    }
    for (VecType::const_iterator iter = this_post.begin();
         iter != this_post.end(); ++iter) {
      int32 i = iter->first; // Gaussian index.
      KALDI_ASSERT(i >= 0 && i < num_gauss &&
                   "Out-of-range Gaussian (mismatched posteriors?)");
      double weight = iter->second;
      stats->gamma(i) += weight;
      stats->X.Row(i).AddVec(weight, frame);
      if (update_variance)
        stats->S[i].AddSp(weight, outer_prod);
    }
  }

  for (int32 i = 0; i < num_gauss; i++) {
      stats->X.Row(i).AddVec(-stats->gamma(i), Means_.Row(i));
  }
}


void IvectorConvStats::RemoveMeans(IvectorExtractorConvUtteranceStats *stats,
    const MatrixBase<BaseFloat> &means_)
{
    if(means_.NumRows() != stats->X.NumRows()) {
        KALDI_ERR << "The means are incompatible " << means_.NumRows() << " " << stats->X.NumRows() ;       
    }

    int32 i = 0, 
        d = means_.NumRows();
    for(i=0; i<d; i++) 
        stats->X.Row(i).AddVec(-stats->gamma(i), means_.Row(i));
}


// This function basically inverts the input and puts it in the output, but it's
// smart numerically.  It uses the prior knowledge that the "inverse_floor" can
// have no eigenvalues less than one, so it applies that floor (in double
// precision) before inverting.  This avoids certain numerical problems that can
// otherwise occur.
// static
void IvectorExtractorConv::InvertWithFlooring(const SpMatrix<double> &inverse_var,
                                          SpMatrix<double> *var) {
  SpMatrix<double> dbl_var(inverse_var);
  int32 dim = inverse_var.NumRows();
  Vector<double> s(dim);
  Matrix<double> P(dim, dim);
  // Solve the symmetric eigenvalue problem, inverse_var = P diag(s) P^T.
  inverse_var.Eig(&s, &P);
  s.ApplyFloor(1.0);
  s.InvertElements();
  var->AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
}


void IvectorExtractorConv::GetIvectorDistribution(
    const IvectorExtractorConvUtteranceStats &utt_stats,
    VectorBase<double> *mean,
    SpMatrix<double> *var) const {
    Vector<double> linear(IvectorDim());
    SpMatrix<double> quadratic(IvectorDim());
    GetIvectorDistMean(utt_stats, &linear, &quadratic);
    if (var != NULL) {
      var->CopyFromSp(quadratic);
      var->Invert(); // now it's a variance.

      // mean of distribution = quadratic^{-1} * linear...
      mean->AddSpVec(1.0, *var, linear, 0.0);
    } else {
      quadratic.Invert();
      mean->AddSpVec(1.0, quadratic, linear, 0.0);
    }
}

void IvectorExtractorConv::GetIvectorDistribution(
    const Vector<double> fostats,
    const Vector<double> bias,
    VectorBase<double> *mean,
    const SpMatrix<double> &var) const {
    Vector<double> linear(IvectorDim());
    SpMatrix<double> quadratic(IvectorDim());
    Vector<double> temp(FeatDim());
    int32 num_gauss = NumGauss();   
    int32 dim = FeatDim();
    for (int32 i = 0; i < num_gauss; i++) {
        SubVector<double> fostats_subvec(fostats, i*dim, dim);
        SubVector<double> bias_subvec(bias, i*dim, dim);
        Vector<double> temp2(fostats_subvec);
        temp2.AddVec(-1.0, bias_subvec);
        temp.AddSpVec(1.0, Sigma_inv_[i], temp2, 0.0);
        linear.AddMatVec(1.0, M_[i], kTrans, temp, 1.0); 
    }
    mean->AddSpVec(1.0, var, linear, 0.0);
}

IvectorExtractorConv::IvectorExtractorConv(
    const IvectorExtractorConvOptions &opts,
    const FullGmm &fgmm,
    bool online_) {
  KALDI_ASSERT(opts.ivector_dim > 0);
  Sigma_inv_.resize(fgmm.NumGauss());
  Sigma_inv_d_.resize(fgmm.NumGauss());
  for (int32 i = 0; i < fgmm.NumGauss(); i++) {
    const SpMatrix<BaseFloat> &inv_var = fgmm.inv_covars()[i];
    Sigma_inv_[i].Resize(inv_var.NumRows());
    Sigma_inv_[i].CopyFromSp(inv_var);
    Sigma_inv_d_[i].Resize(inv_var.NumRows());
    Sigma_inv_d_[i].CopyDiagFromSp(Sigma_inv_[i]);
  }  

  Matrix<double> gmm_means;
  fgmm.GetMeans(&gmm_means);
  Means_.Resize(gmm_means.NumRows(), gmm_means.NumCols());
  Means_.CopyFromMat(gmm_means);

  KALDI_ASSERT(!Sigma_inv_.empty());
  int32 feature_dim = Sigma_inv_[0].NumRows(),
      num_gauss = Sigma_inv_.size();

  M_.resize(num_gauss);
  for (int32 i = 0; i < num_gauss; i++) {
    M_[i].Resize(feature_dim, opts.ivector_dim);
    // TODO: divide by variance
    M_[i].SetRandn();
  }
  w_vec_.Resize(fgmm.NumGauss());
  w_vec_.CopyFromVec(fgmm.weights());
  online = online_;
  diaginit = false;
  ComputeDerivedVars();
}

void IvectorExtractorConv::ComputeDerivedVars() {
  KALDI_LOG << "Computing derived variables for iVector extractor";
  if(online) {      
      KALDI_LOG <<"Cancelling... and returning because this is an online system";
      return;
  }
  gconsts_.Resize(NumGauss());
  for (int32 i = 0; i < NumGauss(); i++) {
    double var_logdet = -Sigma_inv_[i].LogPosDefDet();
    gconsts_(i) = -0.5 * (var_logdet + FeatDim() * M_LOG_2PI);
    // the gconsts don't contain any weight-related terms.
  }
  U_.Resize(NumGauss(), IvectorDim() * (IvectorDim() + 1) / 2);
  SpMatrix<double> temp_U(IvectorDim());
  for (int32 i = 0; i < NumGauss(); i++) {
    // temp_U = M_i^T Sigma_i^{-1} M_i
    temp_U.AddMat2Sp(1.0, M_[i], kTrans, Sigma_inv_[i], 0.0);
    SubVector<double> temp_U_vec(temp_U.Data(),
                                 IvectorDim() * (IvectorDim() + 1) / 2);
    U_.Row(i).CopyFromVec(temp_U_vec);
  }
  KALDI_LOG << "Done.";
}


void IvectorExtractorConv::GetIvectorDistMean(
    const IvectorExtractorConvUtteranceStats &utt_stats,
    VectorBase<double> *linear,
    SpMatrix<double> *quadratic) const {
  Vector<double> temp(FeatDim());
  int32 I = NumGauss();
  if(online) {
      for (int32 i = 0; i < I; i++) {
        double gamma = utt_stats.gamma(i);
        if (gamma <= 0.0)
            continue;
        Vector<double> x(utt_stats.X.Row(i)); // ==(sum post*features) - $\gamma(i) \m_i
        quadratic->AddMat2Vec(gamma, M_[i], kTrans, Sigma_inv_d_[i], 1.0);
        temp.AddVecVec(1.0, Sigma_inv_d_[i], x, 0.0);
        linear->AddMatVec(1.0, M_[i], kTrans, temp, 1.0); 
      }
  } 
  else {
      for (int32 i = 0; i < I; i++) {
        double gamma = utt_stats.gamma(i);
        if (gamma <= 0.0)
            continue;
        Vector<double> x(utt_stats.X.Row(i)); // ==(sum post*features) - $\gamma(i) \m_i
        temp.AddSpVec(1.0, Sigma_inv_[i], x, 0.0);
        linear->AddMatVec(1.0, M_[i], kTrans, temp, 1.0); 
      }
      SubVector<double> q_vec(quadratic->Data(), IvectorDim()*(IvectorDim()+1)/2);
      q_vec.AddMatVec(1.0, U_, kTrans, Vector<double>(utt_stats.gamma), 1.0);
  }

  // Merging GetIvectorDistPrior. 
  for (int32 d = 0; d < IvectorDim(); d++)
    (*quadratic)(d, d) += 1.0;
}




// TODO: check if this is really required. this function is called only
// by UpdateProjections
//
void IvectorExtractorConv::TransformIvectors(const MatrixBase<double> &T,
                                         double new_ivector_offset) {
  Matrix<double> Tinv(T);
  Tinv.Invert();
  // next: M_i <-- M_i Tinv.  (construct temporary copy with Matrix<double>(M_[i]))
  for (int32 i = 0; i < NumGauss(); i++)
    M_[i].AddMatMat(1.0, Matrix<double>(M_[i]), kNoTrans, Tinv, kNoTrans, 0.0);
  KALDI_LOG << "Setting iVector prior offset to " << new_ivector_offset;
}

// The format is different from kaldi's ivector implementation. ivector_offset_
// doesn't exist and hence not written
//
void IvectorExtractorConv::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<IvectorExtractor>");
  WriteToken(os, binary, "<w_vec>");
  w_vec_.Write(os, binary);
  WriteToken(os, binary, "<M>");  
  int32 size = M_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    M_[i].Write(os, binary);
  WriteToken(os, binary, "<Means>");
  int32 nrows = Means_.NumRows(), ncols = Means_.NumCols();
  WriteBasicType(os, binary, nrows);
  WriteBasicType(os, binary, ncols);
  Means_.Write(os, binary);
  WriteToken(os, binary, "<SigmaInv>");  
  KALDI_ASSERT(size == static_cast<int32>(Sigma_inv_.size()));
  for (int32 i = 0; i < size; i++)
    Sigma_inv_[i].Write(os, binary);
  WriteToken(os, binary, "</IvectorExtractor>");
}


void IvectorExtractorConv::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<IvectorExtractor>");
  ExpectToken(is, binary, "<w_vec>");
  w_vec_.Read(is, binary);
  ExpectToken(is, binary, "<M>");  
  int32 size;
  ReadBasicType(is, binary, &size);
  KALDI_ASSERT(size > 0);
  M_.resize(size);
  for (int32 i = 0; i < size; i++)
    M_[i].Read(is, binary);
  ExpectToken(is, binary, "<Means>");
  int32 nrows, ncols;
  ReadBasicType(is, binary, &nrows);
  ReadBasicType(is, binary, &ncols);
  Means_.Resize(nrows, ncols);
  Means_.Read(is, binary);
  ExpectToken(is, binary, "<SigmaInv>");
  Sigma_inv_.resize(size);
  Sigma_inv_d_.resize(size);
  for (int32 i = 0; i < size; i++) {
    Sigma_inv_[i].Read(is, binary);
    Sigma_inv_d_[i].Resize(Sigma_inv_[i].NumCols());
    Sigma_inv_d_[i].CopyDiagFromSp(Sigma_inv_[i]);
    //M_[i].MulRowsVec(Sigma_inv_d_[i]);
  }
  ExpectToken(is, binary, "</IvectorExtractor>");
  ComputeDerivedVars();
}


IvectorConvStats::IvectorConvStats(const IvectorExtractorConv &extractor) {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  
  gamma_.Resize(I);
  Y_.resize(I);
  for (int32 i = 0; i < I; i++)
    Y_[i].Resize(D, S);
  R_.Resize(I, S * (S + 1) / 2);
  num_ivectors_ = 0;
  ivector_sum_.Resize(S);
  ivector_scatter_.Resize(S);
}


void IvectorConvStats::CommitStatsForM(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats &utt_stats,
    const VectorBase<double> &ivec_mean,
    const SpMatrix<double> &ivec_var) {
  subspace_stats_lock_.Lock();

  // We do the occupation stats here also.
  gamma_.AddVec(1.0, utt_stats.gamma);
  
  // Stats for the linear term in M:
  for  (int32 i = 0; i < extractor.NumGauss(); i++) {
    Y_[i].AddVecVec(1.0, utt_stats.X.Row(i),
                    Vector<double>(ivec_mean));
  }

  int32 ivector_dim = extractor.IvectorDim();
  // Stats for the quadratic term in M:
  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);
  SubVector<double> ivec_scatter_vec(ivec_scatter.Data(),
                                     ivector_dim * (ivector_dim + 1) / 2);
  R_.AddVecVec(1.0, utt_stats.gamma, ivec_scatter_vec);

  subspace_stats_lock_.Unlock();
}

// TODO: no need to update variance. remove this function later
void IvectorConvStats::CommitStatsForSigma(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats &utt_stats) {
  variance_stats_lock_.Lock();
  // Storing the raw scatter statistics per Gaussian.  In the update phase we'll
  // take into account some other terms relating to the model means and their
  // correlation with the data.
  for (int32 i = 0; i < extractor.NumGauss(); i++)
    S_[i].AddSp(1.0, utt_stats.S[i]);
  variance_stats_lock_.Unlock();
}



void IvectorConvStats::CommitStatsForPrior(const VectorBase<double> &ivec_mean,
                                       const SpMatrix<double> &ivec_var) {
  SpMatrix<double> ivec_scatter(ivec_var);
  ivec_scatter.AddVec2(1.0, ivec_mean);
  prior_stats_lock_.Lock();
  num_ivectors_ += 1.0;
  ivector_sum_.AddVec(1.0, ivec_mean);
  ivector_scatter_.AddSp(1.0, ivec_scatter);
  prior_stats_lock_.Unlock();
}


void IvectorConvStats::CommitStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats &utt_stats) {
  
  int32 ivector_dim = extractor.IvectorDim();
  Vector<double> ivec_mean(ivector_dim);
  SpMatrix<double> ivec_var(ivector_dim);

  extractor.GetIvectorDistribution(utt_stats,
                                   &ivec_mean,
                                   &ivec_var);

  CommitStatsForM(extractor, utt_stats, ivec_mean, ivec_var);
  CommitStatsForPrior(ivec_mean, ivec_var);
  if (!S_.empty())
    CommitStatsForSigma(extractor, utt_stats);
}


void IvectorConvStats::CheckDims(const IvectorExtractorConv &extractor) const {
  int32 S = extractor.IvectorDim(), D = extractor.FeatDim(),
      I = extractor.NumGauss();
  KALDI_ASSERT(gamma_.Dim() == I);
  KALDI_ASSERT(static_cast<int32>(Y_.size()) == I);
  for (int32 i = 0; i < I; i++)
    KALDI_ASSERT(Y_[i].NumRows() == D && Y_[i].NumCols() == S);
  KALDI_ASSERT(R_.NumRows() == I && R_.NumCols() == S*(S+1)/2);
  //KALDI_ASSERT(Q_.NumRows() == 0);
  KALDI_ASSERT(G_.NumRows() == 0);
  // S_ may be empty or not, depending on whether update_variances == true in
  // the options.
  if (!S_.empty()) {
    KALDI_ASSERT(static_cast<int32>(S_.size() == I));
    for (int32 i = 0; i < I; i++)
      KALDI_ASSERT(S_[i].NumRows() == D);
  }
  KALDI_ASSERT(num_ivectors_ >= 0);
  KALDI_ASSERT(ivector_sum_.Dim() == S);
  KALDI_ASSERT(ivector_scatter_.NumRows() == S);
}


void IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post) {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  CheckDims(extractor);
  
  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
  }
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());

  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorConvUtteranceStats utt_stats(num_gauss, feat_dim,
                                           false);

  extractor.GetStats(feats, post, &utt_stats);
  
  CommitStatsForUtterance(extractor, utt_stats);
}

void IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const MatrixBase<BaseFloat> &feats,
    const Posterior &post,
    const MatrixBase<BaseFloat> &bias) {
  typedef std::vector<std::pair<int32, BaseFloat> > VecType;

  CheckDims(extractor);
  
  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
  }
  KALDI_ASSERT(static_cast<int32>(post.size()) == feats.NumRows());

  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorConvUtteranceStats utt_stats(num_gauss, feat_dim,
                                           false);

  extractor.GetStats(feats, post, &utt_stats);
  RemoveMeans(&utt_stats, bias);
  
  CommitStatsForUtterance(extractor, utt_stats);
}

void IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const IvectorExtractorConvUtteranceStats utt_stats) {
      CommitStatsForUtterance(extractor, utt_stats);
}

double IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const MatrixBase<BaseFloat> &feats,
    const FullGmm &fgmm) {
  int32 num_frames = feats.NumRows();
  Posterior post(num_frames);

  double tot_log_like = 0.0;
  for (int32 t = 0; t < num_frames; t++) {
    SubVector<BaseFloat> frame(feats, t);
    Vector<BaseFloat> posterior(fgmm.NumGauss(), kUndefined);
    tot_log_like += fgmm.ComponentPosteriors(frame, &posterior);
    for (int32 i = 0; i < posterior.Dim(); i++)
      post[t].push_back(std::make_pair(i, posterior(i)));
  }
  AccStatsForUtterance(extractor, feats, post);
  return tot_log_like;
}

void IvectorConvStats::AccStatsForUtterance(
    const IvectorExtractorConv &extractor,
    const MatrixBase<BaseFloat> &feats,
    const MatrixBase<BaseFloat> &posteriors) {

  int32 num_gauss = extractor.NumGauss(), feat_dim = extractor.FeatDim();
  int32 num_feats = posteriors.NumRows();

  if (feat_dim != feats.NumCols()) {
    KALDI_ERR << "Feature dimension mismatch, expected " << feat_dim
              << ", got " << feats.NumCols();
    return;
  }
  else if (num_feats != feats.NumRows()) {
    KALDI_ERR << "Number of features do not match " << num_feats 
              << " in posteriors and " << feats.NumRows() 
              << " in features";
    return;
  }
  // The zeroth and 1st-order stats are in "utt_stats".
  IvectorExtractorConvUtteranceStats utt_stats(num_gauss, feat_dim,
                                           false);
  
  extractor.GetStats(feats, posteriors, &utt_stats);
  
  CommitStatsForUtterance(extractor, utt_stats);
}

void IvectorConvStats::Add(const IvectorConvStats &other) {
  double weight = 1.0; // will later make this configurable if needed.
  gamma_.AddVec(weight, other.gamma_);
  KALDI_ASSERT(Y_.size() == other.Y_.size());
  for (size_t i = 0; i < Y_.size(); i++)
    Y_[i].AddMat(weight, other.Y_[i]);
  R_.AddMat(weight, other.R_);
  //Q_.AddMat(weight, other.Q_);
  G_.AddMat(weight, other.G_);
  KALDI_ASSERT(S_.size() == other.S_.size());
  for (size_t i = 0; i < S_.size(); i++)
    S_[i].AddSp(weight, other.S_[i]);
  num_ivectors_ += weight * other.num_ivectors_;
  ivector_sum_.AddVec(weight, other.ivector_sum_);
  ivector_scatter_.AddSp(weight, other.ivector_scatter_);
}


void IvectorConvStats::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<IvectorStats>");
  WriteToken(os, binary, "<gamma>");
  gamma_.Write(os, binary);
  WriteToken(os, binary, "<Y>");
  int32 size = Y_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Write(os, binary);
  WriteToken(os, binary, "<R>");
  R_.Write(os, binary);
  //WriteToken(os, binary, "<Q>");
  //Q_.Write(os, binary);
  WriteToken(os, binary, "<G>");
  G_.Write(os, binary);
  WriteToken(os, binary, "<S>");
  size = S_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    S_[i].Write(os, binary);
  WriteToken(os, binary, "<NumIvectors>");
  WriteBasicType(os, binary, num_ivectors_);
  WriteToken(os, binary, "<IvectorSum>");
  ivector_sum_.Write(os, binary);
  WriteToken(os, binary, "<IvectorScatter>");
  ivector_scatter_.Write(os, binary);
  WriteToken(os, binary, "</IvectorStats>");
}


void IvectorConvStats::Read(std::istream &is, bool binary, bool add) {
  ExpectToken(is, binary, "<IvectorStats>");
  ExpectToken(is, binary, "<gamma>");
  gamma_.Read(is, binary, add);
  ExpectToken(is, binary, "<Y>");
  int32 size;
  ReadBasicType(is, binary, &size);
  Y_.resize(size);
  for (int32 i = 0; i < size; i++)
    Y_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<R>");
  R_.Read(is, binary, add);
  //ExpectToken(is, binary, "<Q>");
  //Q_.Read(is, binary, add);
  ExpectToken(is, binary, "<G>");
  G_.Read(is, binary, add);
  ExpectToken(is, binary, "<S>");
  ReadBasicType(is, binary, &size);
  S_.resize(size);
  for (int32 i = 0; i < size; i++)
    S_[i].Read(is, binary, add);
  ExpectToken(is, binary, "<NumIvectors>");
  ReadBasicType(is, binary, &num_ivectors_, add);
  ExpectToken(is, binary, "<IvectorSum>");
  ivector_sum_.Read(is, binary, add);
  ExpectToken(is, binary, "<IvectorScatter>");
  ivector_scatter_.Read(is, binary, add);
  ExpectToken(is, binary, "</IvectorStats>");
}

double IvectorConvStats::Update(const IvectorExtractorConvEstimationOptions &opts,
                               IvectorExtractorConv *extractor) const {
  CheckDims(*extractor);
  
  double ans = 0.0;
  ans += UpdateProjections(opts, extractor);
  KALDI_LOG << "Overall objective-function improvement per frame was " << ans;
  extractor->ComputeDerivedVars();
  return ans;
}

double IvectorConvStats::UpdateProjection(
    const IvectorExtractorConvEstimationOptions &opts,
    int32 i,
    IvectorExtractorConv *extractor) const {
  int32 I = extractor->NumGauss(), S = extractor->IvectorDim();
  KALDI_ASSERT(i >= 0 && i < I);
  /*
    For Gaussian index i, maximize the auxiliary function
       Q_i(x) = tr(M_i^T Sigma_i^{-1} Y_i)  - 0.5 tr(Sigma_i^{-1} M_i R_i M_i^T)
   */
  if (gamma_(i) < opts.gaussian_min_count) {
    KALDI_WARN << "Skipping Gaussian index " << i << " because count "
               << gamma_(i) << " is below min-count.";
    return 0.0;
  }
  SpMatrix<double> R(S, kUndefined), SigmaInv(extractor->Sigma_inv_[i]);
  SubVector<double> R_vec(R_, i); // i'th row of R; vectorized form of SpMatrix.
  SubVector<double> R_sp(R.Data(), S * (S+1) / 2);
  R_sp.CopyFromVec(R_vec); // copy to SpMatrix's memory.

  Matrix<double> M(extractor->M_[i]);
  SolverOptions solver_opts;
  solver_opts.name = "M";
  solver_opts.diagonal_precondition = true;
  // TODO: check if inversion is sufficient?
  double impr = SolveQuadraticMatrixProblem(R, Y_[i], SigmaInv, solver_opts, &M),
      gamma = gamma_(i);
  if (i < 4) {
    KALDI_VLOG(1) << "Objf impr for M for Gaussian index " << i << " is "
                  << (impr / gamma) << " per frame over " << gamma << " frames.";
  }
  extractor->M_[i].CopyFromMat(M);
  return impr;
}

class IvectorExtractorConvUpdateProjectionClass {
 public:
  IvectorExtractorConvUpdateProjectionClass(const IvectorConvStats &stats,
                        const IvectorExtractorConvEstimationOptions &opts,
                        int32 i,
                        IvectorExtractorConv *extractor,
                        double *tot_impr):
      stats_(stats), opts_(opts), i_(i), extractor_(extractor),
      tot_impr_(tot_impr), impr_(0.0) { }
  void operator () () {
    //impr_ = stats_.UpdateProjection(opts_, i_, extractor_);
  }
  ~IvectorExtractorConvUpdateProjectionClass() { *tot_impr_ += impr_; }
 private:
  const IvectorConvStats &stats_;
  const IvectorExtractorConvEstimationOptions &opts_;
  int32 i_;
  IvectorExtractorConv *extractor_;
  double *tot_impr_;
  double impr_;
};

double IvectorConvStats::UpdateProjections(
    const IvectorExtractorConvEstimationOptions &opts,
    IvectorExtractorConv *extractor) const {
  int32 I = extractor->NumGauss();
  double tot_impr = 0.0;
  for (int32 i = 0; i < I; i++)
    tot_impr += UpdateProjection(opts, i, extractor);
  double count = gamma_.Sum();
  KALDI_LOG << "Overall objective function improvement for M (mean projections) "
            << "was " << (tot_impr / count) << " per frame over "
            << count << " frames.";
  return tot_impr / count;
}

} // namespace kaldi


