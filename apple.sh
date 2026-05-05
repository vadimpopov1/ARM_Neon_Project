#!/bin/bash

GLFW=/opt/homebrew/opt/glfw

g++ -O2 -march=armv8-a+simd -fno-tree-vectorize \
  -I imgui -I imgui/backends \
  -I $GLFW/include \
  -DGL_SILENCE_DEPRECATION \
  imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
  imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp \
  main.cpp -o out \
  -L $GLFW/lib -lglfw -framework OpenGL

./out