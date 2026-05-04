#!/bin/bash

g++ -O3 -march=armv8-a+simd -o out main.cpp
./out