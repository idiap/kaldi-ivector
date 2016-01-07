// ivectorbin/copy-tmat-conv.cc
//
// Copyright 2015  Srikanth Madikeri (Idiap Research Institute)
//
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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Copy T-matrix structures from binary to ascii,\n"
        "Usage:  copy-tmat-conv <model-in> <model-out>\n"
        "e.g.: \n"
        " copy-tmat-conv final.ie final.ie.txt\n";

    ParseOptions po(usage);
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        ivector_extractor_wxfilename = po.GetArg(2);

    IvectorExtractorConv extractor(true);
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

    
    Output os(ivector_extractor_wxfilename, false);      
    extractor.Write(os.Stream(), false);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
