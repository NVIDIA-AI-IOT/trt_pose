#pragma once

#include "Matrix.hpp"
#include "PairGraph.hpp"
#include "CoverTable.hpp"

template<typename T>
void munkres(Matrix<T> &cost_graph, PairGraph &star_graph);
