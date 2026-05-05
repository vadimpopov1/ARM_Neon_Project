#!/bin/bash

g++ -O3 -march=armv8-a -mfpu=neon -fno-tree-vectorize \
  -I imgui -I imgui/backends \
  -DGL_SILENCE_DEPRECATION \
  imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
  imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
  main.cpp -o out \
  -lglfw -lGL

./out