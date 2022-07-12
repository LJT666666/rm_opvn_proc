//
// Created by heyicheng on 10/18/21.
//
#include "rm_opvn_proc/opvn_plugins.h"
#include <pluginlib/class_list_macros.h>
#include "rm_vision/vision_base/processor_interface.h"
#include <ros/package.h>
#include <ros/callback_queue.h>

using namespace InferenceEngine;
using namespace cv;

PLUGINLIB_EXPORT_CLASS(opvn_plugins::OpvnProcessor, nodelet::Nodelet)

namespace opvn_plugins {
    void OpvnProcessor::onInit() {
        ros::NodeHandle& nh = getPrivateNodeHandle();
        static ros::CallbackQueue my_queue;
        nh.setCallbackQueue(&my_queue);
        initialize(nh);
        my_thread_ = std::thread([](){
             ros::SingleThreadedSpinner spinner;
             spinner.spin(&my_queue);
        });
    }

    void OpvnProcessor::initialize(ros::NodeHandle &nh) {
        nh_ = ros::NodeHandle(nh, "opvn_proc");
        nh_.param<double>("cof_threshold", cof_threshold_, 0.5);
        nh_.param<std::string>("xml_path", xml_path_, "/model/polygon_yolox_nano_100_4000.xml");
        nh_.param<std::string>("bin_path", bin_path_, "/model/polygon_yolox_nano_100_4000.bin");
        nh_.param<double>("nms_area_threshold", nms_area_threshold_, 0.3);
        nh_.param<int>("class_num", class_num_, 36);
        nh_.param<int>("input_row", input_row_, 416);
        nh_.param<int>("input_col", input_col_, 416);
        xml_path_ = ros::package::getPath("rm_opvn_proc") + xml_path_;
        bin_path_ = ros::package::getPath("rm_opvn_proc") + bin_path_;
        ROS_INFO("model_path:%s", (xml_path_+bin_path_).c_str());
        Core ie;
        CNNNetwork network = ie.ReadNetwork(xml_path_, bin_path_);

        if (network.getOutputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 output only");
        if (network.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Configure input & output
        // ---------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        input_name_ = network.getInputsInfo().begin()->first;
        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an
         * infer request. Resize and layout conversions are executed automatically
         * during inference */
        //input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        //input_info->setLayout(Layout::NHWC);
        //input_info->setPrecision(Precision::FP32);
        // --------------------------- Prepare output blobs
        // ----------------------------------------------------

        output_info_ = network.getOutputsInfo().begin()->second;
        output_name_ = network.getOutputsInfo().begin()->first;
        output_info_->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 2. Loading a model to the device
        // ------------------------------------------
        executable_network_ = ie.LoadNetwork(network, "CPU");
        // --------------------------- Step 3. Create an infer request
        // -------------------------------------------------
        infer_request_ = executable_network_.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        it_ = make_shared<image_transport::ImageTransport>(nh_);
        debug_pub_ = it_->advertise("debug_image", 1);
//        cam_sub_ = it_->subscribeCamera("/galaxy_camera/image_raw", 1, &OpvnProcessor::callback, this);
        bag_sub_ = it_->subscribe("/galaxy_camera/image_raw", 1, &OpvnProcessor::callback, this);
        target_pub_ = nh.advertise<decltype(target_array_)>("/processor/result_msg", 1);
    }

    void OpvnProcessor::imageProcess(cv_bridge::CvImagePtr &cv_image) {
        if (cv_image->image.empty()) {
            ROS_ERROR("invalid image input");
            return;
        }
        cv_image->image.copyTo(image_raw_);
//        image_raw_ = Mat::zeros(cv_image->image.rows, cv_image->image.cols, CV_8UC3);
    }

    void OpvnProcessor::findArmor() {
        InferRequest::Ptr infer = executable_network_.CreateInferRequestPtr();
//        Blob::Ptr frame_blob = infer->GetBlob(input_name_);
//        LockedMemory<void> blob_mapped = as<MemoryBlob>(frame_blob)->wmap();
//        auto *blob_data = blob_mapped.as<float *>();
//        //NCHW
//        for (size_t row = 0; row < input_row_; row++) {
//            for (size_t col = 0; col < input_col_; col++) {
//                for (size_t ch = 0; ch < 3; ch++) {
//                    blob_data[input_row_ * input_col_ + row * input_col_ + col] =
//                            float(square_image_.at<Vec3b>(row, col)[ch]) / 255.0;
//                }
//            }
//        }

//        image_raw_ = imread("/home/ljt666666/catkin_ws/src/rm_visplugin/rm_opvn_proc/images/2337.jpg");
        square_image_ = staticResize(image_raw_);
        ROS_INFO("seeking armor!");
        Blob::Ptr img_blob = infer_request_.GetBlob(input_name_);     // just wrap Mat data by Blob::Ptr
        blobFromImage(square_image_, img_blob);

        //infer
        infer_request_.Infer();

        // get infer result
        parseModel();
    }

    void OpvnProcessor::parseModel() {
        const Blob::Ptr output_blob = infer_request_.GetBlob(output_name_);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer
        // happens
        auto moutput_holder = moutput->rmap();
        const float *net_pred = moutput_holder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

        decodeOutputs(net_pred);

//        auto start = chrono::high_resolution_clock::now();
//        draw(image_raw_, objects);
    }

    void OpvnProcessor::decodeOutputs(const float *net_pred) {
        std::vector<Target> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;

        generateGridsAndStride(input_col_, input_row_, strides, grid_strides);
        generateYoloxProposals(grid_strides, net_pred, cof_threshold_, proposals);
//        ROS_INFO("%d", proposals.size());

        qsortDescentInplace(proposals, proposals.size() - 1);

        std::vector<int> picked;
//        ROS_INFO("%d", proposalproposalss.size());

        nmsSortedBoxes(proposals, picked, nms_area_threshold_);

        objects_ = proposals;
        int count = proposals.size();

        for (int i = 0; i < count; i++) {
            rm_msgs::TargetDetection one_target;
            one_target.confidence = proposals[i].prob;
            one_target.id = proposals[i].label;
            int32_t temp[8];

            for (int k = 0; k < 4; k++)
            {
                temp[k * 2] = static_cast<int32_t>(roundf(proposals[i].points[k]));
                temp[k * 2 + 1] = static_cast<int32_t>(roundf(proposals[i].points[k + 1]));
            }
            memcpy(&one_target.pose.orientation.x, &temp[0], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.y, &temp[2], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.z, &temp[4], sizeof(int32_t) * 2);
            memcpy(&one_target.pose.orientation.w, &temp[6], sizeof(int32_t) * 2);
            target_array_.detections.emplace_back(one_target);

        }
    }

    Mat OpvnProcessor::staticResize(cv::Mat &img) {
        r_ = std::min(input_col_ / (img.cols * 1.0), input_row_ / (img.rows * 1.0));
        int unpad_w = r_ * img.cols;
        int unpad_h = r_ * img.rows;
        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());

        //cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat out(input_row_, input_col_, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    void OpvnProcessor::blobFromImage(cv::Mat &img, Blob::Ptr &blob) {
        int channels = 3;
        int img_h = img.rows;
        int img_w = img.cols;
        InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);

        // locked memory holder should be alive all time while access to its buffer happens
        auto mblob_holder = mblob->wmap();

        float *blob_data = mblob_holder.as<float *>();

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < img_h; h++) {
                for (size_t w = 0; w < img_w; w++) {
                    blob_data[c * img_w * img_h + h * img_w + w] =
                            (float) img.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    }

    void OpvnProcessor::generateGridsAndStride(const int target_w, const int target_h, std::vector<int> &strides,
                                          std::vector<GridAndStride> &grid_strides) {
        for (auto stride : strides) {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++) {
                for (int g0 = 0; g0 < num_grid_w; g0++) {
                    grid_strides.push_back((GridAndStride) {g0, g1, stride});
                }
            }
        }
    }

    void OpvnProcessor::qsortDescentInplace(std::vector<Target> &faceobjects, int right) {
        if (faceobjects.empty())
            return;
        int i = 0;
        int j = right;
        float p = faceobjects[(0 + right) / 2].prob;

        while (i <= j) {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j) {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }
    }

    void OpvnProcessor::generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float *net_pred, double cof_threshold_,  std::vector<Target>& proposals){

        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = anchor_idx * (class_num_ + 9);

            // yolox/models/yolo_head.py decode logic
            //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
            //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

            //normal yolox
            /*float x_center = (feat_ptr[basic_pos + 0] + grid0) * stride;
            float y_center = (feat_ptr[basic_pos + 1] + grid1) * stride;
            float w = exp(feat_ptr[basic_pos + 2]) * stride;
            float h = exp(feat_ptr[basic_pos + 3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;*/

            //polygon_yolox
            float x1 = (net_pred[basic_pos + 0] + grid0) * stride;
            float y1 = (net_pred[basic_pos + 1] + grid1) * stride;
            float x2 = (net_pred[basic_pos + 2] + grid0) * stride;
            float y2 = (net_pred[basic_pos + 3] + grid1) * stride;
            float x3 = (net_pred[basic_pos + 4] + grid0) * stride;
            float y3 = (net_pred[basic_pos + 5] + grid1) * stride;
            float x4 = (net_pred[basic_pos + 6] + grid0) * stride;
            float y4 = (net_pred[basic_pos + 7] + grid1) * stride;

            float box_objectness = net_pred[basic_pos + 8];

            for (int class_idx = 0; class_idx < class_num_; class_idx++) {
                int class_id = net_pred[basic_pos + 8 + class_idx + 1];
                float box_cls_score = net_pred[basic_pos + 9 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > cof_threshold_) {
                    Target obj;
                    obj.points.push_back(x1);
                    obj.points.push_back(y1);
                    obj.points.push_back(x2);
                    obj.points.push_back(y2);
                    obj.points.push_back(x3);
                    obj.points.push_back(y3);
                    obj.points.push_back(x4);
                    obj.points.push_back(y4);
                    obj.label = class_id;
                    obj.prob = box_prob;
                    proposals.push_back(obj);
                }
            } // class loop
        } // point anchor loop
    }

    void OpvnProcessor::nmsSortedBoxes(std::vector<Target> &faceobjects, std::vector<int> &picked, double nms_threshold)  {
        picked.clear();
        PolygonIou polygon_iou;
        vector<Target> results;
        while (!faceobjects.empty())
        {
            results.push_back(faceobjects[0]);
            int index = 1;
            while (index < faceobjects.size()) {
                Target a = faceobjects[0];
                Target b = faceobjects[index];
//                bbox_t box1;
//                bbox_t box2;
//                box1.pts[0].x = a.points[0], box1.pts[0].y = a.points[1];
//                box1.pts[3].x = a.points[2], box1.pts[3].y = a.points[3];
//                box1.pts[2].x = a.points[4], box1.pts[2].y = a.points[5];
//                box1.pts[1].x = a.points[6], box1.pts[1].y = a.points[7];

//                box2.pts[0].x = b.points[0], box2.pts[0].y = b.points[1];
//                box2.pts[3].x = b.points[2], box2.pts[3].y = b.points[3];
//                box2.pts[2].x = b.points[4], box2.pts[2].y = b.points[5];
//                box2.pts[1].x = b.points[6], box2.pts[1].y = b.points[7];
//                H.getBoxes(box1, box2);
                double p[8] = {a.points[0], a.points[1], a.points[6], a.points[7], a.points[4], a.points[5], a.points[2], a.points[3]};
                double q[8] = {b.points[0], b.points[1], b.points[6], b.points[7], b.points[4], b.points[5], b.points[2], b.points[3]};
                vector<double> pp, qq;
                for(int i = 0;i < 8; i++)
                {
                    pp.push_back(p[i]);
                    qq.push_back(q[i]);
                }

                float iou_value = polygon_iou.iouPoly(pp, qq);
                if (iou_value == 1)
                    iou_value = 0;
//                cout << "iou_value=" << iou_value << endl;
                if (iou_value > nms_threshold) {
                    faceobjects.erase(faceobjects.begin() + index);
                }
                else {
                    index++;
                }
            }
            faceobjects.erase(faceobjects.begin());
        }
        faceobjects = results;
    }

    double PolygonIou::cross(Points o,Points a,Points b){  //叉积
        return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
    }
    double PolygonIou::area(Points* ps,int n){
        ps[n]=ps[0];
        double res=0;
        for(int i=0;i<n;i++){
            res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
        }
        return res/2.0;
    }
    int PolygonIou::lineCross(Points a,Points b,Points c,Points d,Points&p){
        double s1,s2;
        s1=cross(c,a,b);
        s2=cross(a,b,d);
        if(sig(s1)==0&&sig(s2)==0) return 2;
        if(sig(s2-s1)==0) return 0;
        p.x=(c.x*s2-d.x*s1)/(s2-s1);
        p.y=(c.y*s2-d.y*s1)/(s2-s1);
        return 1;
    }
//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
//void polygonCut(Point*p,int&n,Point a,Point b){
//    static Point pp[maxn];
//    int m=0;p[n]=p[0];
//    for(int i=0;i<n;i++){
//        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
//        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
//            lineCross(a,b,p[i],p[i+1],pp[m++]);
//    }
//    n=0;
//    for(int i=0;i<m;i++)
//        if(!i||!(pp[i]==pp[i-1]))
//            p[n++]=pp[i];
//    while(n>1&&p[n-1]==p[0])n--;
//}
    void PolygonIou::polygonCut(Points*p,int&n,Points a,Points b, Points* pp){
//    static Points pp[maxn];
        int m=0;p[n]=p[0];
        for(int i=0;i<n;i++){
            if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
            if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
                lineCross(a,b,p[i],p[i+1],pp[m++]);
        }
        n=0;
        for(int i=0;i<m;i++)
            if(!i||!(pp[i]==pp[i-1]))
                p[n++]=pp[i];
        while(n>1&&p[n-1]==p[0])n--;
    }
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
    double PolygonIou::intersectArea(Points a,Points b,Points c,Points d){
        Points o(0,0);
        int s1=sig(cross(o,a,b));
        int s2=sig(cross(o,c,d));
        if(s1==0||s2==0)return 0.0;//退化，面积为0
        if(s1==-1) swap(a,b);
        if(s2==-1) swap(c,d);
        Points p[10]={o,a,b};
        int n=3;
        Points pp[maxn];
        polygonCut(p,n,o,c, pp);
        polygonCut(p,n,c,d, pp);
        polygonCut(p,n,d,o, pp);
        double res=fabs(area(p,n));
        if(s1*s2==-1) res=-res;return res;
    }
//求两多边形的交面积
    double PolygonIou::intersectArea(Points*ps1,int n1,Points*ps2,int n2){
        if(area(ps1,n1)<0) reverse(ps1,ps1+n1);
        if(area(ps2,n2)<0) reverse(ps2,ps2+n2);
        ps1[n1]=ps1[0];
        ps2[n2]=ps2[0];
        double res=0;
        for(int i=0;i<n1;i++){
            for(int j=0;j<n2;j++){
                res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
            }
        }
        return res;//assumeresispositive!
    }


    double PolygonIou::iouPoly(vector<double> p, vector<double> q) {
        Points ps1[maxn],ps2[maxn];
        int n1 = 4;
        int n2 = 4;
        for (int i = 0; i < 4; i++) {
            ps1[i].x = p[i * 2];
            ps1[i].y = p[i * 2 + 1];
            ps2[i].x = q[i * 2];
            ps2[i].y = q[i * 2 + 1];
        }
        double inter_area = intersectArea(ps1, n1, ps2, n2);
        double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
        double iou = inter_area / union_area;

//    cout << "inter_area:" << inter_area << endl;
//    cout << "union_area:" << union_area << endl;
//    cout << "iou:" << iou << endl;

        return iou;
    }




    void OpvnProcessor::paramReconfig() {

    }

    void OpvnProcessor::draw() {

    }

    void OpvnProcessor::putObj() {

    }

    OpvnProcessor::Object OpvnProcessor::getObj() {

    }

}  // namespace opvn_plugins
