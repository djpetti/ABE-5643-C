cmake_minimum_required(VERSION 2.8)
project(hw1)

add_library(numerical pollard_rho.cc poly.cc mod.cc euclid.cc)

add_executable(factor factor.cc)
target_link_libraries(factor numerical)