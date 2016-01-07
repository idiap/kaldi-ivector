// ivectorbin/ivector-extractor-conv-acc-stats.cc

// Copyright 2013  Daniel Povey
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

// this class is used to run the command
//  stats.AccStatsForUtterance(extractor, mat, posterior);
// in parallel.
class IvectorTask {
 public:
  IvectorTask(const IvectorExtractorConv &extractor,
              const Matrix<BaseFloat> &features,
              const Posterior &posterior,
              IvectorConvStats *stats): extractor_(extractor) {
                    features_ = features;
                    posterior_=posterior;
                    stats_=stats; 
                    posteriors_are_feats = false;
            }


  IvectorTask(const IvectorExtractorConv &extractor,
              const Matrix<BaseFloat> &features,
              const Matrix<BaseFloat> &posterior,
              IvectorConvStats *stats) : extractor_(extractor) {
                    features_ = features;
                    posterior_features_ = posterior;
                    stats_ = stats; 
                    posteriors_are_feats = true;
            }

  void operator () () {
    if(posteriors_are_feats)
        stats_->AccStatsForUtterance(extractor_, features_, posterior_features_);
    else
        stats_->AccStatsForUtterance(extractor_, features_, posterior_);
  }
  ~IvectorTask() { }  // the destructor doesn't have to do anything.
 private:
  const IvectorExtractorConv &extractor_;
  Matrix<BaseFloat> features_; // not a reference, since features come from a
                               // Table and the reference we get from that is
                               // not valid long-term.
  Posterior posterior_;  // as above.
  Matrix<BaseFloat> posterior_features_;
  IvectorConvStats *stats_;
  bool posteriors_are_feats;
};
}



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Accumulate stats for iVector extractor training\n"
        "Reads in features and Gaussian-level posteriors (typically from a full GMM)\n"
        "Supports multiple threads, but won't be able to make use of too many at a time\n"
        "(e.g. more than about 4)\n"
        "Usage:  ivector-extractor-acc-stats [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <stats-out>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.fgmm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extractor-acc-stats 2.ie '$feats' ark,s,cs:- 2.1.acc\n";

    ParseOptions po(usage);
    bool binary = true,
         posteriors_are_feats = false;
    
    TaskSequencerConfig sequencer_opts;
    po.Register("post-feats", &posteriors_are_feats, "Set when posteriors are in feature format");
    po.Register("binary", &binary, "Write output in binary mode");
    sequencer_opts.Register(&po);

    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);


    // Initialize these Reader objects before reading the IvectorExtractor,
    // because it uses up a lot of memory and any fork() after that will
    // be in danger of causing an allocation failure.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    
    IvectorExtractorConv extractor(false);
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);
    
    IvectorConvStats stats(extractor);
    
    
    int64 tot_t = 0;
    int32 num_done = 0, num_err = 0;
    
    {
      TaskSequencer<IvectorTask> sequencer(sequencer_opts);
      
      if(posteriors_are_feats) {
          RandomAccessBaseFloatMatrixReader posterior_feats_reader(posteriors_rspecifier);
          for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_LOG << "Opening feature " << key;
            if (!posterior_feats_reader.HasKey(key)) {
              KALDI_WARN << "No posteriors for utterance " << key;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const Matrix<BaseFloat> &posterior_feats = posterior_feats_reader.Value(key);

            if (static_cast<int32>(posterior_feats.NumRows()) != mat.NumRows()) {
              KALDI_WARN << "Size mismatch between posterior " << (posterior_feats.NumRows())
                         << " and features " << (mat.NumRows()) << " for utterance "
                         << key;
              num_err++;
              continue;
            }

            sequencer.Run(new IvectorTask(extractor, mat, posterior_feats, &stats));

            tot_t += mat.NumRows();
            num_done++;
            KALDI_LOG << "Done feature " << key;
          }
      }
      else{
          RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
          for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_LOG << "Opening feature " << key;
            if (!posteriors_reader.HasKey(key)) {
              KALDI_WARN << "No posteriors for utterance " << key;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const Posterior &posterior = posteriors_reader.Value(key);

            if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
              KALDI_WARN << "Size mismatch between posterior " << (posterior.size())
                         << " and features " << (mat.NumRows()) << " for utterance "
                         << key;
              num_err++;
              continue;
            }

            sequencer.Run(new IvectorTask(extractor, mat, posterior, &stats));

            tot_t += mat.NumRows();
            num_done++;
            KALDI_LOG << "Done feature " << key;
          }
      }
      // destructor of "sequencer" will wait for any remaining tasks that
      // have not yet completed.
    }
    
    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total frames " << tot_t;
    
    WriteKaldiObject(stats, accs_wxfilename, binary);
    
    KALDI_LOG << "Wrote stats to " << accs_wxfilename;

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

