/*
 * Copyright 2016 Emaad Ahmed Manzoor
 * License: Apache License, Version 2.0
 */
 #include <math.h>

#ifndef STREAMSPOT_PARAM_H_
#define STREAMSPOT_PARAM_H_

#ifdef DEBUG
#define NDEBUG            0
#endif

#define K                 1
#define B                 100
#define R                 20
#define BUF_SIZE          50
#define DELIMITER         '\t'
#define L                 1000      // must be = B * R
#define SEED              23
#define CLUSTER_UPDATE_INTERVAL   10

#define PI                3.1415926535897

//時間減衰で使う
#define HEIGHT            7
#define m                 (int)pow(2, HEIGHT)-1  //63
#define DECAYED_RATE      0.95

#endif
