#!/bin/bash

g++ -O3 -march=armv8-a -fno-tree-vectorize \
  -I imgui -I imgui/backends -I implot \
  -DGL_SILENCE_DEPRECATION \
  imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
  imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
  implot/implot.cpp implot/implot_items.cpp \
  main.cpp -o out \
  -lglfw -lGL

./out