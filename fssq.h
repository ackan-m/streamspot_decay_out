#ifndef STREAMSPOT_FSSQ_H_
#define STREAMSPOT_FSSQ_H_

#include "param.h"
#include <iostream>
#include <math.h>
#include <vector>

namespace std{
  struct Counter{
    string item = "NULL";
    double cnt = 0;
    int ut = 0;
    double error = 0;
    bool delay = false;
  };

  struct StreamHeap{
    int t = 0;
    int size = 0;
  };

  void delayedSorting(Counter quasi_heap[][m], StreamHeap streaheap[], int root, int gid);

  void ssqAlgorithm(Counter quasi_heap[][m], StreamHeap streaheap[], string outgoing_chunks, int gid);

  void ssqPointer(Counter *qh[][m], StreamHeap *sh[]);

  void CounterInit(Counter qh[][m], vector<uint32_t> test_gids);

  void StreamHeapInit(StreamHeap sh[]);
}
#endif
