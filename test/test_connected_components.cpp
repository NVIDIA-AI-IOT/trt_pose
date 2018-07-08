#include "gtest/gtest.h"

#include "test_utils.h"

#include "../src/connected_components.h"

TEST(connected_components, ComputesCorrectPeaks0) {

  // 3 part types, 2 connection types (T0->T1), (T1->T2)
  // PARTS
  //   T0 - 3 detected
  //   T1 - 2 detected
  //   T2 - 1 detected
  
  const int part_types = 2;
  const int connection_types = 0;
  const int max_num_indices = 10;
  int counts[3] = { 2, 1 };
  
  //imatrix_t assignment_graphs[connection_types];// = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, NULL, NULL, connection_types);
  ASSERT_EQ(count, 3);
}

TEST(connected_components, ComputesCorrectPeaks1) {

  // 3 part types, 2 connection types (T0->T1), (T1->T2)
  // PARTS
  //   T0 - 3 detected
  //   T1 - 2 detected
  //   T2 - 1 detected
  
  const int part_types = 2;
  const int connection_types = 0;
  const int max_num_indices = 10;
  int counts[3] = { 2, 2 };
  
  //imatrix_t assignment_graphs[connection_types];// = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, NULL, NULL, connection_types);
  ASSERT_EQ(count, 4);
}


TEST(connected_components, ComputesCorrectPeaks2) {

  // 3 part types, 2 connection types (T0->T1), (T1->T2)
  // PARTS
  //   T0 - 3 detected
  //   T1 - 2 detected
  //   T2 - 1 detected
  
  const int part_types = 3;
  const int connection_types = 2;
  const int max_num_indices = 10;
  int counts[3] = { 3, 2, 1};
  ivector2_t assignment_indicies[] = {
    { 0, 1 },
    { 1, 2 }
  };
  
  int connection_0_data[] = {
    1, 0, 
    0, 0, 
    0, 1, 
  };
  imatrix_t connection_0_mat;
  connection_0_mat.rows = counts[0];
  connection_0_mat.cols = counts[1];
  connection_0_mat.data = connection_0_data;

  int connection_1_data[] = {
    0, 
    1, 
  };
  imatrix_t connection_1_mat;
  connection_1_mat.rows = counts[1];
  connection_1_mat.cols = counts[2];
  connection_1_mat.data = connection_1_data;

  imatrix_t assignment_graphs[connection_types] = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, assignment_graphs, assignment_indicies, connection_types);
  ASSERT_EQ(3, count);
}

TEST(connected_components, ComputesCorrectPeaks3) {

  // 3 part types, 2 connection types (T0->T1), (T1->T2)
  // PARTS
  //   T0 - 3 detected
  //   T1 - 2 detected
  //   T2 - 1 detected
  
  const int part_types = 3;
  const int connection_types = 2;
  const int max_num_indices = 10;
  int counts[3] = { 3, 3, 3};
  ivector2_t assignment_indicies[] = {
    { 0, 1 },
    { 1, 2 }
  };
  
  int connection_0_data[] = {
    1, 0, 0,
    0, 0, 1,
    0, 1, 0
  };
  imatrix_t connection_0_mat;
  connection_0_mat.rows = counts[0];
  connection_0_mat.cols = counts[1];
  connection_0_mat.data = connection_0_data;

  int connection_1_data[] = {
    0, 1, 0,
    1, 0, 0,
    0, 0, 1
  };
  imatrix_t connection_1_mat;
  connection_1_mat.rows = counts[1];
  connection_1_mat.cols = counts[2];
  connection_1_mat.data = connection_1_data;

  imatrix_t assignment_graphs[connection_types] = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, assignment_graphs, assignment_indicies, connection_types);
  ASSERT_EQ(3, count);
}

TEST(connected_components, ComputesCorrectPeaks4) {

  const int part_types = 3;
  const int connection_types = 2;
  const int max_num_indices = 10;
  int counts[3] = { 3, 3, 3};
  ivector2_t assignment_indicies[] = {
    { 0, 1 },
    { 1, 2 }
  };
  
  int connection_0_data[] = {
    1, 0, 0,
    0, 0, 1,
    0, 0, 0
  };
  imatrix_t connection_0_mat;
  connection_0_mat.rows = counts[0];
  connection_0_mat.cols = counts[1];
  connection_0_mat.data = connection_0_data;

  int connection_1_data[] = {
    0, 1, 0,
    0, 0, 0,
    0, 0, 1
  };
  imatrix_t connection_1_mat;
  connection_1_mat.rows = counts[1];
  connection_1_mat.cols = counts[2];
  connection_1_mat.data = connection_1_data;

  imatrix_t assignment_graphs[connection_types] = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, assignment_graphs, assignment_indicies, connection_types);
  ASSERT_EQ(5, count);
}

TEST(connected_components, ComputesCorrectPeaks5) {

  const int part_types = 3;
  const int connection_types = 2;
  const int max_num_indices = 10;
  int counts[3] = { 2, 3, 3};
  ivector2_t assignment_indicies[] = {
    { 0, 1 },
    { 1, 2 }
  };
  
  int connection_0_data[] = {
    1, 0, 0,
    0, 0, 1,
  };
  imatrix_t connection_0_mat;
  connection_0_mat.rows = counts[0];
  connection_0_mat.cols = counts[1];
  connection_0_mat.data = connection_0_data;

  int connection_1_data[] = {
    0, 1, 0,
    0, 0, 0,
    0, 0, 1
  };
  imatrix_t connection_1_mat;
  connection_1_mat.rows = counts[1];
  connection_1_mat.cols = counts[2];
  connection_1_mat.data = connection_1_data;

  imatrix_t assignment_graphs[connection_types] = { connection_0_mat, connection_1_mat };

  int components_data[part_types * max_num_indices];
  memset(&components_data, 0, sizeof(components_data));

  imatrix_t components_mat;
  components_mat.rows = part_types;
  components_mat.cols = max_num_indices;
  components_mat.data = components_data;

  int count = connected_components(&components_mat, counts, assignment_graphs, assignment_indicies, connection_types);
  ASSERT_EQ(4, count);
}

#ifndef EXCLUDE_MAIN
int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
