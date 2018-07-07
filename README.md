TODO
----

### C 
- [x] **peak_local_max** - algorithm to find local maxima above threshold in confidence map
  - [x] CPU implementation
  - [x] Tests
- [x] **munkres** - algorithm to solve assignment problem for optimal part matching
  - [x] CPU implementation
  - [x] Tests
- [ ] **paf_score_graph** algorithm to generate part connection score graph
  - [x] CPU implementation
  - [ ] Tests
- [ ] **connected_components** algorithm to parse assignment graph into connected objects
  - [ ] CPU implementation
  - [ ] Tests
- [ ] **gauss_newton_batch** algorithm to implement gauss newton optimization for batches of NxM matrices
  - [ ] CUDA implementation
  - [ ] Tests
- [ ] **peak_gaussian_fit** algorithm to fit gaussian-ish function to data near peak in confidence map
  - [ ] CUDA residual and jacobian computation
  - [ ] CUDA gauss newton optimization
  - [ ] Tests

### C++

- [ ] **TRTModel** - trt wrapper class
- [ ] **TRTPoseModel** - pose wrapper class

### Python

- [ ] **TRTModel** - swig / python interface to TRTModel
- [ ] **TRTPoseModel** - swig / python interface to TRTPoseModel