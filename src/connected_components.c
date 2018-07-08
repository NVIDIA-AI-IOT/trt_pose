#include "connected_components.h"
#include "stdlib.h"

#define MAX_QUEUE_SIZE 1000

typedef struct queue {
  ivector2_t nodes[MAX_QUEUE_SIZE];
  int front;
  int size;
  int capacity;
} queue_t;

void queue_init(queue_t *Q) {
  Q->front = 0;
  Q->size = 0;
  Q->capacity = MAX_QUEUE_SIZE;
}

static inline int queue_empty(queue_t *Q)
{
  return Q->size == 0;
}

static inline int queue_full(queue_t *Q)
{
  return Q->size == Q->capacity;
}

static inline int queue_push(queue_t *Q, ivector2_t node)
{
  if (queue_full(Q)) 
  {
    return 0;
  }
  else
  {
    Q->nodes[(Q->front + Q->size) % Q->capacity] = node;
    Q->size++;
    return 1;
  }
}

int queue_pop(queue_t *Q)
{
  if (Q->size > 0) 
  {
    Q->size--;
    Q->front = Q->front % Q->capacity;
    return 1;
  }
  else
  {
    return 0;
  }
}

ivector2_t queue_front(queue_t *Q)
{
  return Q->nodes[Q->front];
}

// (cmap, idx_cmap), return 1 if child found 0 otherwise
int connected_child(imatrix_t assignment_graph, ivector2_t inter_graph_connection, ivector2_t node, ivector2_t *child)
{
  if (inter_graph_connection.i == node.i) 
  {
    // search for connection in row
    child->i = inter_graph_connection.j;
    for (int j = 0; j < assignment_graph.cols; j++) 
    {
      int *connected = imatrix_at_mutable(&assignment_graph, node.j, j);
      if (*connected)
      {
        *connected = 0; 
        child->j = j;
        return 1;
      }
    }
    return 0;
  } 
  else if (inter_graph_connection.j == node.i) 
  {
    // search for connection in column
    child->i = inter_graph_connection.i;
    for (int i = 0; i < assignment_graph.rows; i++) 
    {
      int *connected = imatrix_at_mutable(&assignment_graph, i, node.j);
      if (*connected)
      {
        *connected = 0;
        child->j = i;
        return 1;
      }
    }
    return 0;
  }
  else 
  {
    return 0; // part not found or node does not belong to connection
  }
}

void connected_search(imatrix_t *components, int component_id, imatrix_t *assignment_graphs, ivector2_t *assignment_graph_indices, int num_assignment_graphs, 
    ivector2_t node)
{
  queue_t Q;
  queue_init(&Q);
  queue_push(&Q, node);
  while (!queue_empty(&Q)) 
  {
    ivector2_t node = queue_front(&Q);
    queue_pop(&Q);
    *imatrix_at_mutable(components, node.i, node.j) = component_id;
     
    for (int i = 0; i < num_assignment_graphs; i++) 
    {
      ivector2_t child;
      if (connected_child(assignment_graphs[i], assignment_graph_indices[i], node, &child))
      {
        queue_push(&Q, child);
      }
    }
  }
}

int connected_components(imatrix_t *components, int *part_counts, imatrix_t *assignment_graphs,
    ivector2_t *assignment_graph_indices, int num_assignment_graphs)
{
  int count = 0;
  for (int i = 0; i < components->rows; i++) 
  {
    for (int j = 0; j < part_counts[i]; j++) 
    {
      if (imatrix_at(components, i, j) == 0)  // unvisited
      {
        ivector2_t node;
        node.i = i;
        node.j = j;
        count++;
        connected_search(components, count, assignment_graphs, assignment_graph_indices, num_assignment_graphs, node);
      }
    }
  }
  return count;
}
