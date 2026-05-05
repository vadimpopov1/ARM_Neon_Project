#!/bin/bash

g++ -O2 -march=armv8-a -fno-tree-vectorize -o out main.cpp
./out