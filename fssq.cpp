#include "fssq.h"
#include "param.h"
#include <iostream>
#include <math.h>

namespace std{
  void delayedSorting(Counter quasi_heap[][m], StreamHeap streamheap[], int root, int gid){

    Counter c = quasi_heap[gid][root]; //サブツリーのルートカウンタ
    c.error *= pow(DECAYED_RATE, streamheap[gid].t - c.ut);
    // cout << c.cnt << " " << DECAYED_RATE << " " << streamheap[gid].t << " " << c.ut << " ";
    c.cnt *= pow(DECAYED_RATE, streamheap[gid].t - c.ut);
    // cout << c.cnt << endl;
    c.ut = streamheap[gid].t;
    quasi_heap[gid][root] = c;

    //葉ノードならreturnする
    if(2*root + 1 >= m){
      return;
    }
    if(quasi_heap[gid][root*2+1].item == "NULL"){
      return;
    }

    //flagが立ってなかったらsort終わり
    if(c.delay == false){
      return;
    }

    //子ノードに対してソート
    delayedSorting(quasi_heap, streamheap, root*2+1, gid);
    delayedSorting(quasi_heap, streamheap, root*2+2, gid);

    Counter sml; //子ノードの小さいほう
    Counter left = quasi_heap[gid][root*2+1];
    Counter right = quasi_heap[gid][root*2+2];

    //親と子を比較し交換する
    if(left.cnt > right.cnt){
      sml = right;
      if(sml.cnt < c.cnt){
        quasi_heap[gid][root*2+2] = c;
        quasi_heap[gid][root*2+2].delay = true;
        quasi_heap[gid][root] = sml;
      }
    }else{
      sml = left;
      if(sml.cnt < c.cnt){
        quasi_heap[gid][root*2+1] = c;
        quasi_heap[gid][root*2+1].delay = true;
        quasi_heap[gid][root] = sml;
      }
    }

    quasi_heap[gid][root].delay = false;

  }

  void ssqAlgorithm(Counter quasi_heap[][m],StreamHeap streamheap[], string outgoing_chunks,int gid){
    //現在時刻を更新
    streamheap[gid].t += 1;
    //outgoing_chunkがヒープ内にあるか探す
    int length;
    for(length = 0; length < m; length++){
      if(quasi_heap[gid][length].item == outgoing_chunks){
        // cout << "attazo---" << endl;
        break;
      }
    }

    Counter c; //新しいカウンタを作る
    if(length < m){//ヒープにある
      c = quasi_heap[gid][length];
      c.error *= pow(DECAYED_RATE, streamheap[gid].t - c.ut);
      c.cnt = c.cnt * pow(DECAYED_RATE, streamheap[gid].t - c.ut) + 1;
      c.delay = true;
      c.ut = streamheap[gid].t;
      //heap内のカウンタ更新
      quasi_heap[gid][length] = c;
    }else{//ヒープにない
      if(streamheap[gid].size == m){//heapがfullなら
        //rootからソート
        delayedSorting(quasi_heap, streamheap, 0, gid);
        // cout <<"length  " << length << endl;
        Counter r = quasi_heap[gid][0]; //rはソート後のroot
       //rootをcで置き換え
       c.item = outgoing_chunks;
       c.error = r.cnt+1;
       c.cnt = r.cnt+1;
       c.delay = true;
       c.ut = streamheap[gid].t;
       quasi_heap[gid][0] = c;

     }else{//heapがfullでないなら
       //heapのサイズを増やす
       streamheap[gid].size += 1;

       c.item = outgoing_chunks;
       c.error = 0;
       c.cnt = 1;
       c.delay = true;
       c.ut = streamheap[gid].t;

       //heapに挿入
       quasi_heap[gid][streamheap[gid].size-1] = c;
     }
   }
 }

 void ssqPointer(Counter *qh[][m], StreamHeap *sh[]){
   cout << qh[0][0]->cnt << endl;
  //  qh[0][0]->cnt += 1;
  //  cout << qh[0][0]->cnt << endl;
  cout << sh[0]->size << endl;
 }


 void CounterInit(Counter qh[][m], vector<uint32_t> test_gids){
   cout << "222" << endl;
   for(auto& gid : test_gids){
     for(int i=0; i<m ;i++){
       cout << gid << endl;
       cout << "あいうえお" << endl;
       qh[gid][i].item = "かきくけこ";
      //  cout << (qh[gid][i])->item <<  endl;
      //  qh[gid][i]->ut = 0;
      //  qh[gid][i]-> error = 0;
      //  qh[gid][i]->delay = false;
     }
   }

 }

 void StreamHeapInit(StreamHeap sh[]){
   cout << sh[0].t << endl;
   sh[0].t = 100;
   cout << sh[0].t << endl;
 }

}
