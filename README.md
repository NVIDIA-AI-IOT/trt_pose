TODO
----

### inference

- [x] **FindPeaks** - peak detection in cmap
- [x] **PafCostGraph** - generate part-part cost graph from peaks / PAF
- [x] **Munkres** - optimal part-part assignment from cost graph via munkres algorithm
- [x] **ConnectParts** - connect parts into objects by searching part-part assignment graphs
- [x] **ParseObjects** - combine above steps to parse objects from cmap, paf and configuration
- [ ] **GaussNewton** - gauss newton optimization for multiple NxM matrices in batches (CUDA impl)
- [ ] **GaussianFit** - fit gaussians to peaks in cmap via gauss newton optimization (CUDA impl)
- [ ] **PoseModel** - takes tensorrt engine and config used, wraps execution

# training

- [ ] **GeneratePaf** - generate part affinity field from objects
- [ ] **GenerateCmap** - generate confidence map from objects
- [ ] **CocoDataset** - wrapper for coco / generators to generate samples
- [ ] **Model** - generate model from configuration, starts from pretrained image cls model, configurable num stages, feature extractor, stage model, etc.
- [ ] **Train** - training script for selected dataset (really, coco)