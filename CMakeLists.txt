cmake_minimum_required(VERSION 3.12)
project(AirwaySkeleton)

set(CMAKE_CXX_STANDARD 14)
set(CGAL_DIR "/home/ihealth/programs/CGAL/lib/cmake/CGAL")

find_package(CGAL REQUIRED COMPONENTS ImageIO)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(mesh_a_3d_gray_image mesh_a_3d_gray_image.cpp)
target_link_libraries(mesh_a_3d_gray_image CGAL::CGAL Boost::boost Eigen3::Eigen)

add_executable(simple_mcfskel_example simple_mcfskel_example.cpp)
target_link_libraries(simple_mcfskel_example CGAL::CGAL Boost::boost Eigen3::Eigen)

