typedef struct tensor3f {
  float *data;
  int channels;
  int height;
  int width;
} tensor3f_t;

typedef struct weighted_graph {
  float *data;
  int rows;
  int cols;
} weighted_graph_t;

typedef struct weighted_graph_list {
  weighted_graph_t *graphs;
  int size;
} weighted_graph_list_t;

typedef struct graph {
  int *data;
  int rows;
  int cols;
} graph_t;

typedef struct graph_list {
  graph_t *graphs;
  int size;
} graph_list_t;

typedef struct point2i {
  int i;
  int j;
} point2i_t;

typedef struct point2f {
 float i;
 float j; 
} point2f_t;

#define PEAK_LIST_CAPACITY 100

typedef struct peak_list {
  point2i_t peaks[PEAK_LIST_CAPACITY];
} peak_list_t;
