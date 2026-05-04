#!/bin/bash

g++ -O3 -mfpu=neon -march=armv8-a -o out main.cpp
./out