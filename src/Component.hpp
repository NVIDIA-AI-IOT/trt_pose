#pragma once

#include <vector>
#include <unordered_map>
#include "Part.hpp"

class Component
{
public:

  void addPart(const Part &part)
  {
    parts.insert({part.channel, part});
  }

  Part getPart(int channel)
  {
    return parts.at(channel);
  }

  bool hasPart(int channel)
  {
    return parts.count(channel) > 0;
  }

private:
  std::unordered_map<int, Part> parts;
};
