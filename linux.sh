#!/bin/bash

g++ -O2 -mfpu=neon -march=armv8-a -fno-tree-vectorize -o out main.cpp
./out