cmake_minimum_required(VERSION 3.16)
project(radxa_camera_age LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Main app: C entry (main.c) + C++/OpenCV core (app.cpp).
add_executable(age_gender src/main.c src/app.cpp)
target_link_libraries(age_gender ${OpenCV_LIBS} Threads::Threads)

# --- diagnostic tools (optional) ---
add_executable(capture_test src/capture_test.cpp)
target_link_libraries(capture_test ${OpenCV_LIBS})
add_executable(decode_test src/decode_test.cpp)
target_link_libraries(decode_test ${OpenCV_LIBS})
add_executable(mjpg_test src/mjpg_test.cpp)
target_link_libraries(mjpg_test ${OpenCV_LIBS})
add_executable(image_test src/image_test.cpp)
target_link_libraries(image_test ${OpenCV_LIBS})
