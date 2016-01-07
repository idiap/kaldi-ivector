// ivectorbin/ivector-extract-conv.cc

// Copyright 2015  Daniel Povey
//                 Srikanth Madikeri (Idiap Research Institute)

// See ../../COPYING for clarification regarding multiple authors
//
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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/conv-ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"

namespace kaldi {

// This class will be used to parallelize over multiple threads the job
// that this program does.  The work happens in the operator (), the
// output happens in the destructor.
class IvectorExtractTask {
 public:
  IvectorExtractTask(const IvectorExtractorConv &extractor,
                     std::string utt,
                     const Matrix<BaseFloat> &feats,
                     const Posterior &posterior,
                     BaseFloatVectorWriter *writer
                     ):
      extractor_(extractor), utt_(utt), feats_(feats), posterior_(posterior),
      writer_(writer)  { }

  void operator () () {
    bool need_2nd_order_stats = false;
    
    IvectorExtractorConvUtteranceStats utt_stats(extractor_.NumGauss(),
                                             extractor_.FeatDim(),
                                             need_2nd_order_stats);
      
    extractor_.GetStats(feats_, posterior_, &utt_stats);

    ivector_.Resize(extractor_.IvectorDim());
    extractor_.GetIvectorDistribution(utt_stats, &ivector_, NULL);
  }
  ~IvectorExtractTask() {
    writer_->Write(utt_, Vector<BaseFloat>(ivector_));
  }
 private:
  const IvectorExtractorConv &extractor_;
  std::string utt_;
  Matrix<BaseFloat> feats_;
  Posterior posterior_;
  BaseFloatVectorWriter *writer_;
  Vector<double> ivector_;
};
}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for utterances, using a trained iVector extractor,\n"
        "and features and Gaussian-level posteriors\n"
        "Usage:  ivector-extract [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.fgmm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extract final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = true, posteriors_are_feats = false;
    TaskSequencerConfig sequencer_config;
    po.Register("posteriors-are-feats", &posteriors_are_feats, 
                "Set when posteriors are in feature formats");
    sequencer_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        ivectors_wspecifier = po.GetArg(4);

    IvectorExtractorConv extractor(false);
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

    double tot_auxf_change = 0.0;
    int64 tot_t = 0;
    int32 num_done = 0, num_err = 0;
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    BaseFloatVectorWriter ivector_writer(ivectors_wspecifier);

    if(posteriors_are_feats) {
          RandomAccessBaseFloatMatrixReader posterior_feats_reader(posteriors_rspecifier);
          for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            if (!posteriors_reader.HasKey(key)) {
              KALDI_WARN << "No posteriors for utterance " << key;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const Matrix<BaseFloat> &posterior_mat = posterior_feats_reader.Value(key);
            if (static_cast<int32>(posterior_mat.NumRows()) != mat.NumRows()) {
              KALDI_WARN << "Size mismatch between posterior " << (posterior_mat.NumRows())
                         << " and features " << (mat.NumRows()) << " for utterance "
                         << key;
              num_err++;
              continue;
            }
            IvectorExtractorConvUtteranceStats utt_stats(extractor.NumGauss(),
                                                 extractor.FeatDim(),
                                                 false);
          
            extractor.GetStats(mat, posterior_mat, &utt_stats);

            Vector<double> ivector;
            ivector.Resize(extractor.IvectorDim());
            extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);            
            ivector_writer.Write(key, Vector<BaseFloat>(ivector));
         }
    } 
    else  {
      TaskSequencer<IvectorExtractTask> sequencer(sequencer_config);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string key = feature_reader.Key();
        if (!posteriors_reader.HasKey(key)) {
          KALDI_WARN << "No posteriors for utterance " << key;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(key);

        if(posterior.size() == 0 || mat.NumRows() == 0) {
            KALDI_WARN << "No feature/posterior vectors for " << key;
            continue;
        }
        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << (posterior.size())
                     << " and features " << (mat.NumRows()) << " for utterance "
                     << key;
          num_err++;
          continue;
        }

        sequencer.Run(new IvectorExtractTask(extractor, key, mat, posterior,
                                             &ivector_writer));
                      
        tot_t += posterior.size();
        num_done++;
      }
      // Destructor of "sequencer" will wait for any remaining tasks.
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total frames " << tot_t;

    if (compute_objf_change)
      KALDI_LOG << "Overall average objective-function change from estimating "
                << "ivector was " << (tot_auxf_change / tot_t) << " per frame "
                << " over " << tot_t << " frames.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

