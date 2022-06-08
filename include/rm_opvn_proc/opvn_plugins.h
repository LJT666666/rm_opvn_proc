//
// Created by heyicheng on 10/18/21.
//

#ifndef SRC_OPVN_PLUGINS_H
#define SRC_OPVN_PLUGINS_H

#include <ros/ros.h>
#include "rm_vision/vision_base/processor_interface.h"
#include <inference_engine.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <pluginlib/class_loader.h>
#include <thread>
#include <rm_msgs/TargetDetectionArray.h>
#include <rm_msgs/TargetDetection.h>


using namespace InferenceEngine;
using namespace cv;
using namespace std;

//extern int number;
#define maxn 51
const double eps=1E-8;

namespace opvn_plugins {

    struct Target {
        cv::Rect_<float> rect;
        std::vector<float> points;
        int label;
        float prob;
    };

    struct GridAndStride {
        int grid0;
        int grid1;
        int stride;
    };

    struct  bbox_t
    {
        cv::Point2f pts[4];
    };


class OpvnProcessor : public rm_vision::ProcessorInterface , public nodelet::Nodelet{
    public:
        void onInit() override;

        void initialize(ros::NodeHandle &nh) override;

        void imageProcess(cv_bridge::CvImagePtr &cv_image) override;

        void findArmor() override;

        void parseModel();

        void paramReconfig() override;

        void draw() override;

        Object getObj() override;

        void putObj() override;


    private:
    rm_msgs::TargetDetectionArray target_array_;
    ros::Publisher target_pub_;
    ros::NodeHandle nh_;
    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Publisher debug_pub_;
    image_transport::CameraSubscriber cam_sub_;

    void callback(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info)
    {

        target_array_.header = info->header;
        target_array_.detections.clear();
        boost::shared_ptr<cv_bridge::CvImage> temp = boost::const_pointer_cast<cv_bridge::CvImage>(cv_bridge::toCvShare(img, "bgr8"));
        auto predict_start = std::chrono::high_resolution_clock::now();
        imageProcess(temp);
        findArmor();
        draw();
        auto predict_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> infer_time = predict_end - predict_start;
        ROS_INFO("infer_time: %f", infer_time.count());
        for (auto& target : target_array_.detections)
        {
            target.pose.position.x = info->roi.x_offset;
            target.pose.position.y = info->roi.y_offset;
        }
        if (!target_array_.detections.empty()) {
            ROS_INFO("find targets!");
            int32_t buffer[8];
            memcpy(buffer, &target_array_.detections[0].pose.orientation.x, sizeof(int32_t) * 2);
            memcpy(buffer+2, &target_array_.detections[0].pose.orientation.y, sizeof(int32_t) * 2);
            memcpy(buffer+4, &target_array_.detections[0].pose.orientation.z, sizeof(int32_t) * 2);
            memcpy(buffer+6, &target_array_.detections[0].pose.orientation.w, sizeof(int32_t) * 2);
            static const char *class_names[] = {
                    "red_2", "red_3", "red_4", "red_5"
            };
            for(int i = 0;i < objects_.size(); i++){
                line(image_raw_, Point(objects_[i].points[0]/r_, objects_[i].points[1]/r_), Point(objects_[i].points[2]/r_, objects_[i].points[3]/r_), Scalar(0, 0, 255), 5, 4);
                line(image_raw_, Point(objects_[i].points[2]/r_, objects_[i].points[3]/r_), Point(objects_[i].points[4]/r_, objects_[i].points[5]/r_), Scalar(0, 0, 255), 5, 4);
                line(image_raw_, Point(objects_[i].points[4]/r_, objects_[i].points[5]/r_), Point(objects_[i].points[6]/r_, objects_[i].points[7]/r_), Scalar(0, 0, 255), 5, 4);
                line(image_raw_, Point(objects_[i].points[6]/r_, objects_[i].points[7]/r_), Point(objects_[i].points[0]/r_, objects_[i].points[1]/r_), Scalar(0, 0, 255), 5, 4);
                cv::putText(image_raw_, to_string(objects_[i].label), cv::Point(objects_[i].points[0]/r_, objects_[i].points[3]/r_-40),cv::FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 255, 0), 3);
            }

            debug_pub_.publish(cv_bridge::CvImage(info->header, "bgr8", image_raw_).toImageMsg());
            target_pub_.publish(target_array_);
        }
    }


        int class_num_;  // label num
        std::string xml_path_;  // .xml file path
        std::string bin_path_;  // .bin file path
        std::string input_name_;  // input name defined by model
        std::string output_name_;  // input name defined by model
        DataPtr output_info_;  // information of model output
        ExecutableNetwork executable_network_;  // executable network
        InferRequest infer_request_;
        double cof_threshold_;  // confidence threshold of object class
        double nms_area_threshold_;  // non-maximum suppression
        int input_row_, input_col_;  // input shape of model
        Mat square_image_;  //  input image of model
        Mat image_raw_;  //predict image
        std::vector<Target> objects_; //
        float r_;


        Mat staticResize(cv::Mat &img);
        void blobFromImage(cv::Mat &img, Blob::Ptr &blob);
        void decodeOutputs(const float *net_pred, float scale, int img_w, int img_h);
        void generateGridsAndStride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
        void qsortDescentInplace(std::vector<Target> &faceobjects, int left, int right);
        void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float *net_pred, double cof_threshold_,  std::vector<Target> &proposals);
        void nmsSortedBoxes(std::vector<Target> &faceobjects, std::vector<int> &picked, double nms_threshold);

        std::thread my_thread_;


    };
    

    int sig(double d){
        return(d>eps)-(d<-eps);
    }

    struct Points{
        double x,y;
        Points(){}
        Points(double x,double y):x(x),y(y){}
        bool operator==(const Points&p)const{
            return sig(x-p.x)==0&&sig(y-p.y)==0;
        }
    };
    
    class PolygonIou{
    public:
        double cross(Points o,Points a,Points b);
        double area(Points* ps,int n);
        int lineCross(Points a,Points b,Points c,Points d,Points&p);
        void polygon_cut(Points*p,int&n,Points a,Points b, Points* pp);
        double intersectArea(Points a,Points b,Points c,Points d);
        double intersectArea(Points*ps1,int n1,Points*ps2,int n2);
        double iou_poly(std::vector<double> p, std::vector<double> q);

    private:


    };

}  // namespace opvn_plugins






#endif //SRC_OPVN_PLUGINS_H