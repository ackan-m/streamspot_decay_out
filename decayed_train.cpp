#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include "graph.h"
#include "hash.h"
#include <iostream>
#include "param.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "decayed_train.h"

namespace std {
  // tuple<vector<double>, chrono::nanoseconds, chrono::nanoseconds>
  // update_streamhash_sketche(const edge& e, const vector<graph>& graphs,
  //                            vector<bitset<L>>& streamhash_sketches,
  //                            // vector<vector<double>>& streamhash_projections,
  //                            uint32_t chunk_length,
  //                            const vector<vector<uint64_t>>& H
  //                            )
 void decayed_trained_streamhash_projection(const edge& e, const vector<graph>& graphs,
                             vector<bitset<L>>& streamhash_sketches,
                             vector<vector<double>>& streamhash_projections,
                            uint32_t chunk_length,
                            const vector<vector<uint64_t>>& H,
                            deque<edge> cache
                            ){
    // source node = (src_id, src_type)
    // dst_node = (dst_id, dst_type)
    // shingle substring = (src_type, e_type, dst_type)
    //assert(K == 1 && chunk_length >= 4);

    // for timing
    // chrono::time_point<chrono::steady_clock> start;
    // chrono::time_point<chrono::steady_clock> end;
    // chrono::microseconds shingle_construction_time;
    // chrono::microseconds sketch_update_time;

    auto& src_id = get<F_S>(e);
    auto& src_type = get<F_STYPE>(e);
    auto& gid = get<F_GID>(e);

    // auto& sketch = streamhash_sketches[gid];
    auto& projection = streamhash_projections[gid];
    auto& g = graphs[gid];

    // start = chrono::steady_clock::now(); // start shingle construction

    // construct the last chunk
    auto& outgoing_edges = g.at(make_pair(src_id, src_type));
    uint32_t n_outgoing_edges = outgoing_edges.size();
    int shingle_length = 2 * (n_outgoing_edges + 1);
    int last_chunk_length = shingle_length - chunk_length *
                            (shingle_length/chunk_length);
    if (last_chunk_length == 0)
      last_chunk_length = chunk_length;
    string last_chunk("x", last_chunk_length);
    int len = last_chunk_length, i = n_outgoing_edges - 1;
    do {
      last_chunk[--len] = get<1>(outgoing_edges[i]); // dst_type
      if (len <= 0)
        break;
      last_chunk[--len] = get<2>(outgoing_edges[i]); // edge_type
      i--;
    } while (len > 0 && i >= 0);
    if (i < 0) {
      if (len == 2) {
        last_chunk[--len] = src_type;
      }
      if (len == 1) {
        last_chunk[--len] = ' ';
      }
    }

    // construct the second last chunk if it exists
    string sec_last_chunk("x", chunk_length);
    if (i >= 0) {
      len = chunk_length;

      if (last_chunk_length % 2 != 0) {
        sec_last_chunk[--len] = get<2>(outgoing_edges[i]); // edge_type
        i--;
      }

      if (i >=0 && len >= 0) {
        do {
          sec_last_chunk[--len] = get<1>(outgoing_edges[i]);
          if (len <= 0)
            break;
          sec_last_chunk[--len] = get<2>(outgoing_edges[i]);
          i--;
        } while (len > 0 && i >= 0);
      }

      if (i < 0) {
        if (len == 2) {
          sec_last_chunk[--len] = src_type;
        }
        if (len == 1) {
          sec_last_chunk[--len] = ' ';
        }
      }
    }

  #ifdef DEBUG
    string shingle(" ", 1);
    shingle.reserve(2 * (n_outgoing_edges + 1));
    shingle.push_back(src_type);
    for (uint32_t i = 0; i < n_outgoing_edges; i++) {
      shingle.push_back(get<2>(outgoing_edges[i]));
      shingle.push_back(get<1>(outgoing_edges[i]));
    }

    cout << "Shingle: " << shingle << endl;
    vector<string> chunks = get_string_chunks(shingle, chunk_length);

    cout << "Last chunk: " << last_chunk << endl;
    assert(last_chunk == chunks[chunks.size() - 1]);
    if (chunks.size() > 1) {
      cout << "Second last chunk: " << sec_last_chunk << endl;
      assert(sec_last_chunk == chunks[chunks.size() - 2]);
    }
  #endif

    string shingle(" ", 1);
    shingle.reserve(2 * (n_outgoing_edges + 1));
    shingle.push_back(src_type);
    for (uint32_t i = 0; i < n_outgoing_edges; i++) {
      shingle.push_back(get<2>(outgoing_edges[i]));
      shingle.push_back(get<1>(outgoing_edges[i]));
    }
    vector<string> chunks = get_string_chunks(shingle, chunk_length);
    // cout << "print chunks:" << endl;
    // for(auto& aaa:chunks){
    //   cout << aaa << endl;
    // }
    vector<string> incoming_chunks; // to be hashed and added
    vector<string> outgoing_chunks; // to be hashed and subtracted

    // incoming_chunks.push_back(last_chunk);
    incoming_chunks = chunks;

    int ut = -1;
    if (n_outgoing_edges > 1) { // this is not the first edge
      if (last_chunk_length == 1) {
        outgoing_chunks=chunks;//自作
        outgoing_chunks.pop_back();//自作
        outgoing_chunks.pop_back();//自作
        outgoing_chunks.push_back(sec_last_chunk.substr(0,
                                                        sec_last_chunk.length() - 1));
      } else if (last_chunk_length == 2) {
        // do nothing, only incoming chunk is the last chunk
        outgoing_chunks=chunks;//自作
        outgoing_chunks.pop_back();//自作
      } else { // 2 < last_chunk_length <= chunk_length, last chunk had 2 chars added
        outgoing_chunks=chunks;//自作
        outgoing_chunks.pop_back();//自作
        outgoing_chunks.push_back(last_chunk.substr(0, last_chunk_length - 2));
      }
      //ここから自作
      //最後に来たエッジの情報
      auto& last_dst_id = get<0>(outgoing_edges[n_outgoing_edges-2]);
      auto& last_dst_type = get<1>(outgoing_edges[n_outgoing_edges-2]);
      auto& last_e_type = get<2>(outgoing_edges[n_outgoing_edges-2]);
      edge last_edge = make_tuple(src_id, src_type, last_dst_id, last_dst_type,last_e_type, gid);

      for(auto& E:cache){
        ut += 1;
        if(get<0>(E)!=get<0>(last_edge)){
          continue;
        }
        if(get<1>(E)!=get<1>(last_edge)){
          continue;
        }
        if(get<2>(E)!=get<2>(last_edge)){
          continue;
        }
        if(get<3>(E)!=get<3>(last_edge)){
          continue;
        }
        if(get<4>(E)!=get<4>(last_edge)){
          continue;
        }
        // cout << "あったぞ" << endl;
        // cout << ut << "番目"<< endl;
        break;
      }
    }

    // end = chrono::steady_clock::now(); // end shingle construction
    // shingle_construction_time =
      // chrono::duration_cast<chrono::microseconds>(end - start);

  #ifdef DEBUG
    cout << "Incoming chunks: ";
    for (auto& c : incoming_chunks) {
      cout << c << ",";
    }
    cout << endl;

    cout << "Outgoing chunks: ";
    for (auto& c : outgoing_chunks) {
      cout << c << ",";
    }
    cout << endl;
  #endif

    // record the change in the projection vector
    // this is used to update the centroid
    // vector<int> projection_delta(L, 0);
    // vector<double> projection_delta(L, 0);
    // double decayed_delta;

    // start = chrono::steady_clock::now(); // start sketch update

    // update the projection vectors
    // for (auto& chunk : incoming_chunks) {
    //   for (uint32_t i = 0; i < L; i++) {
    //     // decayed_delta = projection[i];
    //     // cout << decayed_delta << " ";
    //     int delta = hashmulti(chunk, H[i]);
    //     projection[i] *= DECAYED_RATE;  //減衰させる
    //     projection[i] += delta;
    //     // cout << projection[i] << " " ;
    //     // decayed_delta = projection[i] - decayed_delta;
    //     // cout << decayed_delta << endl;
    //     // projection_delta[i] += decayed_delta;
    //   }
    // }

    for (uint32_t i = 0; i < L; i++) {
      projection[i] *= DECAYED_RATE;
      for (auto& chunk : incoming_chunks) {
        int delta = hashmulti(chunk, H[i]);
        projection[i] += delta;
        // projection_delta[i] += delta;
      }
    }

    // for (auto& chunk : outgoing_chunks) {
    //   for (uint32_t i = 0; i < L; i++) {
    //     int delta = hashmulti(chunk, H[i]);
    //     projection[i] -= delta*pow(DECAYED_RATE, (cache.size()-1)-ut);
    //     // projection_delta[i] -= delta;
    //   }
    // }
    // if(outgoing_chunks.size()!=0){
    //   if(last_chunk_length==2){
    //     for (uint32_t i = 0; i < L; i++) {
    //       int delta = 0;
    //       for(int j=0;j < outgoing_chunks.size(); j++){
    //         delta += hashmulti(outgoing_chunks[j], H[i]);
    //       }
    //       projection[i] += (1-pow(DECAYED_RATE, (cache.size()-1)-ut))*delta;
    //     }
    //   }else{
    //     for (uint32_t i = 0; i < L; i++) {
    //       int delta = 0;
    //       for(int j=0;j < outgoing_chunks.size()-1; j++){
    //         delta += hashmulti(outgoing_chunks[j], H[i]);
    //       }
    //       projection[i] += (1-pow(DECAYED_RATE, (cache.size()-1)-ut))*delta;
    //       projection[i] -= hashmulti(outgoing_chunks[outgoing_chunks.size()-1], H[i])
    //       *pow(DECAYED_RATE, (cache.size()-1)-ut);
    //     }
    //   }
    //
    // }
    if(outgoing_chunks.size()!=0){
      for (uint32_t i = 0; i < L; i++) {
        // decayed_delta=projection[i];
        double delta = 0;
        for(int j=0;j < outgoing_chunks.size(); j++){
          delta += hashmulti(outgoing_chunks[j], H[i]);
        }
        // projection_delta[i] -= delta * pow(DECAYED_RATE, (cache.size()-1)-ut);
        projection[i] -= delta * pow(DECAYED_RATE, (cache.size()-1)-ut);
      }
    }


    // update sketch = sign(projection)
    // for (uint32_t i = 0; i < L; i++) {
    //   sketch[i] = projection[i] >= 0 ? 1 : 0;
    // }
  // }
    // end = chrono::steady_clock::now(); // end sketch update
    // sketch_update_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // return make_tuple(projection_delta, shingle_construction_time, sketch_update_time);
  }
}
