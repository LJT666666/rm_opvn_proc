cmake_minimum_required(VERSION 3.0.2)
project(rm_opvn_proc)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        dynamic_reconfigure
        geometry_msgs
        message_generation
        rm_common
        rm_msgs
        roscpp
        std_msgs
        pluginlib
        nodelet
        rm_vision
        image_transport
        cv_bridge
)


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(InferenceEngine REQUIRED)
set(ngraph_DIR "/opt/intel/openvino_2021/deployment_tools/ngraph")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "/opt/intel/openvino_2021/deployment_tools/ngraph/lib/libngraph.so")
set(ngraph_INCLUDE_DIRS "/opt/intel/openvino_2021/deployment_tools/ngraph/include/")

generate_dynamic_reconfigure_options(
        cfg/Opvn.cfg
)

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        roscpp
        dynamic_reconfigure
        geometry_msgs
        message_generation
        rm_common
        rm_msgs
        rm_vision
        std_msgs
        pluginlib
        LIBRARIES ${PROJECT_NAME}
        image_transport
        cv_bridge
)


include_directories(
        include
        include/rm_opvn_proc
        ${catkin_INCLUDE_DIRS}
        /opt/intel/openvino_2021/deployment_tools/inference_engine/include/
        /opt/intel/openvino_2021/deployment_tools/ngraph/include/
)

add_library(opvn_plugins src/opvn_plugins.cpp)

add_dependencies(opvn_plugins
        ${PROJECT_NAME}_gencfg
        )

target_link_libraries(opvn_plugins
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
        )
