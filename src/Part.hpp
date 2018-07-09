#pragma once

class Part
{
public:
  Part(int channel, int idx, int row, int col) : channel(channel), idx(idx), row(row), col(col) {};

  const int channel;
  const int idx;
  const int row;
  const int col;
};
