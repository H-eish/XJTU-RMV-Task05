#include <chrono>
#include <cmath> // For std::fabs
#include <cstddef>
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include <map>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "armor_detector/MvCameraControl.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

// 将海康相机像素格式转换为ROS图像编码格式
std::string MvPixelTypeToString(MvGvspPixelType enType)
{
    switch (enType)
    {
    case PixelType_Gvsp_Mono8:
        return "mono8";
    case PixelType_Gvsp_BGR8_Packed:
        return "bgr8";
    case PixelType_Gvsp_RGB8_Packed:
        return "rgb8";
    case PixelType_Gvsp_BayerGR8:
        return "bayer_grbg8";
    case PixelType_Gvsp_BayerRG8:
        return "bayer_rggb8";
    case PixelType_Gvsp_BayerGB8:
        return "bayer_gbrg8";
    case PixelType_Gvsp_BayerBG8:
        return "bayer_bggr8";
    default:
        return "unknown";
    }
}

class ArmorDetectorNode : public rclcpp::Node
{
public:
    explicit ArmorDetectorNode(const rclcpp::NodeOptions &options)
        : Node("armor_detector_node", options),
          handle_(nullptr),
          is_connected_(false)
    {
        // ==================== 1. ROS2参数声明和初始化 ====================
        declare_parameters();
        this->declare_parameter<std::string>("onnx_model_path", "/home/heish/Documents/Projects/XJTU-RMV-Task05/armor_detector/src/armor_detector/models/model/armor_digit.onnx");
        onnx_path_ = this->get_parameter("onnx_model_path").as_string();

        try
        {
            net_ = cv::dnn::readNetFromONNX(onnx_path_);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            RCLCPP_INFO(this->get_logger(), "Successfully loaded ONNX model from: %s", onnx_path_.c_str());
        }
        catch (const cv::Exception &e)
        {
            RCLCPP_FATAL(this->get_logger(), "Failed to load ONNX model: %s", e.what());
            if (rclcpp::ok())
            {
                rclcpp::shutdown();
            }
            return;
        }

        // ==================== 2. 创建图像发布者 ====================
        std::string topic_name = this->get_parameter("topic_name").as_string();
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_name, 10);

        std::string processed_topic_name = this->get_parameter("processed_topic_name").as_string();
        processed_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(processed_topic_name, 10);

        input_source_ = this->get_parameter("input_source").as_string(); //获取输入源参数
        video_path_ = this->get_parameter("video_path").as_string();

        // 根据输入源参数初始化video文件或相机
        if (input_source_ == "video")
        {
            RCLCPP_INFO(this->get_logger(), "Input source is VIDEO from: %s", video_path_.c_str());
            cap_.open(video_path_);
            if (!cap_.isOpened())
            {
                RCLCPP_FATAL(this->get_logger(), "Failed to open video file!");
                if (rclcpp::ok())
                {
                    rclcpp::shutdown();
                }
                return;
            }
        }
        else if (input_source_ == "camera")
        {
            RCLCPP_INFO(this->get_logger(), "Input source is HIK CAMERA.");
            // ==================== 3. [修改] 初始化SDK，整个生命周期只执行一次 ====================
            int nRet = MV_CC_Initialize();
            if (MV_OK != nRet)
            {
                RCLCPP_FATAL(this->get_logger(), "Failed to initialize SDK! nRet [0x%x]", nRet);
                // 如果SDK初始化失败，节点无法工作，直接关闭
                if (rclcpp::ok())
                {
                    rclcpp::shutdown();
                }
                return;
            }
        }
        else
        {
            RCLCPP_FATAL(this->get_logger(), "Invalid input_source parameter: %s. Use 'camera' or 'video'.", input_source_.c_str());
            if (rclcpp::ok())
            {
                rclcpp::shutdown();
            }
            return;
        }

        // ==================== 4. 设置参数回调 ====================
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ArmorDetectorNode::on_parameter_set, this, std::placeholders::_1));

        // ==================== 5. 创建定时器，用于抓取、发布图像和处理重连 ====================
        grab_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), // [修改] 降低轮询频率，10Hz足够用于重连检测和图像抓取
            std::bind(&ArmorDetectorNode::process_and_publish, this));

        // ==================== 6. 创建定时器，用于从相机同步参数状态 ====================
        if (input_source_ == "camera")
        {
            param_sync_timer_ = this->create_wall_timer(
                std::chrono::seconds(1), // 每秒同步一次
                std::bind(&ArmorDetectorNode::sync_parameters_from_camera, this));
        }

        RCLCPP_INFO(this->get_logger(), "Armor detector node has been initialized. Attempting first connection...");

        // 使用您提供的相机内参矩阵 (K)
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 2556.7, 0, 1547.0,
                          0, 2576.7, 1102.9,
                          0, 0, 1.0);

        // 使用您提供的畸变系数 (D)
        dist_coeffs_ = (cv::Mat_<double>(1, 4) << -0.4109, 0.1718, 0, 0);

        // 定义真实世界中装甲板的3D坐标 (单位：米)
        // !! 警告: 请务必修改为你的实际尺寸 !!
        const double armor_width = 0.135;                                             // 例如: 13.5 cm
        const double armor_height = 0.056;                                            // 例如: 5.6 cm
        armor_points_.push_back(cv::Point3f(-armor_width / 2, -armor_height / 2, 0)); // 左下
        armor_points_.push_back(cv::Point3f(armor_width / 2, -armor_height / 2, 0));  // 右下
        armor_points_.push_back(cv::Point3f(armor_width / 2, armor_height / 2, 0));   // 右上
        armor_points_.push_back(cv::Point3f(-armor_width / 2, armor_height / 2, 0));  // 左上
    }

    ~ArmorDetectorNode()
    {
        if (handle_ != nullptr)
        {
            MV_CC_StopGrabbing(handle_);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
        }
        // [修改] SDK反初始化，同样只执行一次
        MV_CC_Finalize();
        RCLCPP_INFO(this->get_logger(), "Armor detector has been destroyed.");
    }

private:
    // ==================== ROS2参数相关函数 ====================
    void declare_parameters()
    {
        this->declare_parameter<std::string>("camera_sn", "");
        this->declare_parameter<double>("exposure_time", 5000.0);
        this->declare_parameter<double>("gain", 5.0);
        this->declare_parameter<double>("frame_rate", 30.0);
        this->declare_parameter<std::string>("pixel_format", "bgr8");
        this->declare_parameter<std::string>("camera_frame_id", "camera_frame");
        this->declare_parameter<std::string>("topic_name", "image_raw");
        this->declare_parameter<std::string>("input_source", "camera"); // "camera" or "video"
        this->declare_parameter<std::string>("video_path", "path/to/your/video.mp4");
        this->declare_parameter<std::string>("processed_topic_name", "image_processed");

        pixel_format_map_["mono8"] = PixelType_Gvsp_Mono8;
        pixel_format_map_["bgr8"] = PixelType_Gvsp_BGR8_Packed;
        pixel_format_map_["rgb8"] = PixelType_Gvsp_RGB8_Packed;
        pixel_format_map_["bayer_grbg8"] = PixelType_Gvsp_BayerGR8;
        pixel_format_map_["bayer_rggb8"] = PixelType_Gvsp_BayerRG8;
        pixel_format_map_["bayer_gbrg8"] = PixelType_Gvsp_BayerGB8;
        pixel_format_map_["bayer_bggr8"] = PixelType_Gvsp_BayerBG8;
    }

    // ==================== 修改点在这里 ====================
    rcl_interfaces::msg::SetParametersResult on_parameter_set(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        if (!is_connected_ || handle_ == nullptr)
        {
            result.successful = false;
            result.reason = "Camera is not connected.";
            return result;
        }

        for (const auto &param : parameters)
        {
            std::string name = param.get_name();
            if (name == "exposure_time")
            {
                if (MV_OK != MV_CC_SetFloatValue(handle_, "ExposureTime", param.as_double()))
                {
                    result.successful = false;
                    result.reason = "Failed to set exposure time.";
                }
            }
            else if (name == "gain")
            {
                if (MV_OK != MV_CC_SetFloatValue(handle_, "Gain", param.as_double()))
                {
                    result.successful = false;
                    result.reason = "Failed to set gain.";
                }
            }
            else if (name == "frame_rate")
            {
                if (MV_OK != MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", param.as_double()))
                {
                    result.successful = false;
                    result.reason = "Failed to set frame rate.";
                }
            }
            else if (name == "pixel_format")
            {
                const auto &format_str = param.as_string();
                if (pixel_format_map_.count(format_str))
                {
                    // 关键步骤：更改像素格式需要停止/启动流
                    grab_timer_->cancel(); // 暂停抓图定时器，防止冲突

                    int nRet = MV_CC_StopGrabbing(handle_);
                    if (MV_OK != nRet)
                    {
                        result.successful = false;
                        result.reason = "Failed to stop grabbing to change pixel format.";
                        RCLCPP_ERROR(this->get_logger(), "%s Error: [0x%x]", result.reason.c_str(), nRet);
                        // 即使失败，也要尝试恢复
                        grab_timer_->reset();
                        continue; // 处理下一个参数
                    }

                    nRet = MV_CC_SetEnumValue(handle_, "PixelFormat", pixel_format_map_.at(format_str));
                    if (MV_OK != nRet)
                    {
                        result.successful = false;
                        result.reason = "Failed to set PixelFormat value.";
                        RCLCPP_ERROR(this->get_logger(), "%s Error: [0x%x]", result.reason.c_str(), nRet);
                        // 失败后，尝试重启流以恢复相机
                        MV_CC_StartGrabbing(handle_);
                        grab_timer_->reset();
                        continue;
                    }

                    nRet = MV_CC_StartGrabbing(handle_);
                    if (MV_OK != nRet)
                    {
                        result.successful = false;
                        result.reason = "CRITICAL: Failed to restart grabbing after setting pixel format.";
                        is_connected_ = false; // 标记为断开，让主循环去处理重连
                        RCLCPP_FATAL(this->get_logger(), "%s Error: [0x%x]", result.reason.c_str(), nRet);
                    }
                    else
                    {
                        RCLCPP_INFO(this->get_logger(), "Successfully set PixelFormat to %s", format_str.c_str());
                    }

                    grab_timer_->reset(); // 恢复抓图定时器
                }
                else
                {
                    result.successful = false;
                    result.reason = "Unsupported pixel format: " + param.as_string();
                }
            }
            if (!result.successful)
            {
                // 如果不是pixel_format相关的错误，在这里打印
                if (name != "pixel_format")
                {
                    RCLCPP_WARN(this->get_logger(), "%s", result.reason.c_str());
                }
            }
        }
        return result;
    }
    // ====================================================

    void sync_parameters_from_camera()
    {
        if (!is_connected_ || handle_ == nullptr)
        {
            return;
        }

        if (MV_OK != MV_CC_InvalidateNodes(handle_))
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "Could not invalidate node cache. Parameter sync may be inaccurate.");
            return;
        }

        std::vector<rclcpp::Parameter> updated_params;
        MVCC_FLOATVALUE stFloatValue{};

        if (MV_OK == MV_CC_GetFloatValue(handle_, "ExposureTime", &stFloatValue))
        {
            if (std::fabs(this->get_parameter("exposure_time").as_double() - stFloatValue.fCurValue) > 1e-6)
            {
                updated_params.emplace_back("exposure_time", stFloatValue.fCurValue);
            }
        }

        if (MV_OK == MV_CC_GetFloatValue(handle_, "Gain", &stFloatValue))
        {
            if (std::fabs(this->get_parameter("gain").as_double() - stFloatValue.fCurValue) > 1e-6)
            {
                updated_params.emplace_back("gain", stFloatValue.fCurValue);
            }
        }

        if (MV_OK == MV_CC_GetFloatValue(handle_, "ResultingFrameRate", &stFloatValue))
        {
            if (std::fabs(this->get_parameter("frame_rate").as_double() - stFloatValue.fCurValue) > 0.1)
            {
                updated_params.emplace_back("frame_rate", stFloatValue.fCurValue);
            }
        }

        MVCC_ENUMVALUE stEnumValue{};
        if (MV_OK == MV_CC_GetEnumValue(handle_, "PixelFormat", &stEnumValue))
        {
            std::string hw_format_str = MvPixelTypeToString(static_cast<MvGvspPixelType>(stEnumValue.nCurValue));
            if (hw_format_str != "unknown" && this->get_parameter("pixel_format").as_string() != hw_format_str)
            {
                updated_params.emplace_back("pixel_format", hw_format_str);
            }
        }

        if (!updated_params.empty())
        {
            this->set_parameters(updated_params);
            RCLCPP_INFO(this->get_logger(), "Synchronized parameters from camera hardware.");
        }
    }

    // ==================== 相机控制函数 ====================
    // [修改] connect_camera不再负责SDK初始化
    void connect_camera()
    {
        int nRet = MV_OK;

        memset(&stDeviceList_, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList_);
        if (MV_OK != nRet)
        {
            // 枚举失败，简单返回，等待下次尝试
            return;
        }

        if (stDeviceList_.nDeviceNum == 0)
        {
            // [修改] 未找到设备是预期行为，安静地返回，等待下次尝试
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Found %d devices. Attempting to establish connection.", stDeviceList_.nDeviceNum);

        std::string camera_sn = this->get_parameter("camera_sn").as_string();
        int device_index = -1;
        if (!camera_sn.empty())
        {
            for (unsigned int i = 0; i < stDeviceList_.nDeviceNum; i++)
            {
                MV_CC_DEVICE_INFO *pDeviceInfo = stDeviceList_.pDeviceInfo[i];
                const char *sn = (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE) ? (const char *)pDeviceInfo->SpecialInfo.stGigEInfo.chSerialNumber : (const char *)pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber;
                if (camera_sn == std::string(sn))
                {
                    device_index = i;
                    RCLCPP_INFO(this->get_logger(), "Found matching camera with SN: %s", camera_sn.c_str());
                    break;
                }
            }
            if (device_index == -1)
            {
                RCLCPP_WARN(this->get_logger(), "Found devices, but camera with SN [%s] not among them. Will retry.", camera_sn.c_str());
                return;
            }
        }
        else
        {
            device_index = 0;
            RCLCPP_INFO(this->get_logger(), "No camera_sn specified, connecting to the first device found.");
        }

        nRet = MV_CC_CreateHandle(&handle_, stDeviceList_.pDeviceInfo[device_index]);
        if (MV_OK != nRet)
        {
            RCLCPP_ERROR(this->get_logger(), "CreateHandle fail! nRet [0x%x]", nRet);
            return;
        }

        nRet = MV_CC_OpenDevice(handle_);
        if (MV_OK != nRet)
        {
            RCLCPP_ERROR(this->get_logger(), "OpenDevice fail! nRet [0x%x]", nRet);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Device opened successfully.");

        set_initial_parameters();

        nRet = MV_CC_StartGrabbing(handle_);
        if (MV_OK != nRet)
        {
            RCLCPP_ERROR(this->get_logger(), "StartGrabbing fail! nRet [0x%x]", nRet);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
            return;
        }

        // [修改] 只有在所有步骤成功后，才将状态设置为已连接
        is_connected_ = true;
        RCLCPP_INFO(this->get_logger(), "Camera connected and started grabbing.");
    }

    void set_initial_parameters()
    {
        if (handle_ == nullptr)
            return;
        MV_CC_SetEnumValue(handle_, "TriggerMode", 0);
        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
        MV_CC_SetFloatValue(handle_, "ExposureTime", this->get_parameter("exposure_time").as_double());
        MV_CC_SetFloatValue(handle_, "Gain", this->get_parameter("gain").as_double());
        MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", this->get_parameter("frame_rate").as_double());

        std::string format_str = this->get_parameter("pixel_format").as_string();
        if (pixel_format_map_.count(format_str))
        {
            MV_CC_SetEnumValue(handle_, "PixelFormat", pixel_format_map_.at(format_str));
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Initial pixel format %s not supported, using camera default.", format_str.c_str());
        }
        RCLCPP_INFO(this->get_logger(), "Initial camera parameters set.");
    }

    // [修改] 重命名并重构了主要逻辑函数
    void process_and_publish()
    {
        cv::Mat frame;
        bool frame_acquired = false;

        // ==================== 1. 根据数据源获取图像 ====================
        if (input_source_ == "video")
        {
            // --- 视频文件模式 ---
            cap_ >> frame;
            if (!frame.empty())
            {
                frame_acquired = true;
            }
            else
            {
                // 视频播放结束，循环播放
                RCLCPP_INFO(this->get_logger(), "Video stream ended. Restarting from the beginning.");
                cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                return; // 本次调用结束，等待下一次定时器触发
            }
        }
        else if (input_source_ == "camera")
        {
            // --- 相机模式 ---
            if (is_connected_)
            {
                // a) 检查物理连接状态
                if (handle_ != nullptr && !MV_CC_IsDeviceConnected(handle_))
                {
                    RCLCPP_ERROR(this->get_logger(), "Camera has been disconnected! Cleaning up resources and attempting to reconnect.");
                    MV_CC_StopGrabbing(handle_);
                    MV_CC_DestroyHandle(handle_);
                    handle_ = nullptr;
                    is_connected_ = false;
                    return; // 本次调用结束，下一次将进入重连逻辑
                }

                // b) 如果连接正常，抓取图像
                MV_FRAME_OUT stImageInfo{};
                // 设置一个合理的超时时间，例如100ms
                int nRet = MV_CC_GetImageBuffer(handle_, &stImageInfo, 100);

                if (nRet == MV_OK)
                {
                    // 1. 创建一个临时的 Mat 视图
                    cv::Mat sdk_frame_view(stImageInfo.stFrameInfo.nHeight, stImageInfo.stFrameInfo.nWidth, CV_8UC3, stImageInfo.pBufAddr);

                    // 2. 立刻将数据克隆到我们自己的 Mat 对象中
                    frame = sdk_frame_view.clone();

                    // 3. 现在数据已经安全地在我们自己的 'frame' 中了，可以立即释放SDK的缓冲区
                    MV_CC_FreeImageBuffer(handle_, &stImageInfo);

                    // 4. 确认克隆成功
                    if (!frame.empty())
                    {
                        frame_acquired = true;
                    }
                }
                else
                {
                    // 获取图像失败，但不是超时错误，则打印警告
                    // 超时是正常现象（例如帧率较低或无触发时），无需打印日志
                    if (nRet != static_cast<int>(MV_E_GC_TIMEOUT))
                    {
                        RCLCPP_WARN(this->get_logger(), "GetImageBuffer failed! nRet [0x%x]", nRet);
                    }
                }
            }
            else
            {
                // c) 如果未连接，则尝试重连
                // 使用 THROTTLE 避免日志刷屏，每2秒打印一次重连信息
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Camera is not connected. Attempting to reconnect...");
                connect_camera(); // 此函数成功后会把 is_connected_ 设为 true
            }
        }

        // 如果没有成功获取到图像，则直接返回
        if (!frame_acquired || frame.empty())
        {
            return;
        }

        // ==================== 2. 创建消息头并发布原始图像 ====================
        // 使用 cv_bridge 将 cv::Mat 转换为 sensor_msgs::msg::Image
        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = this->get_parameter("camera_frame_id").as_string();

        // 发布原始图像
        sensor_msgs::msg::Image::SharedPtr raw_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        image_pub_->publish(*raw_msg);

        // ==================== 3. 处理图像并发布结果 ====================
        // 克隆一份图像用于绘制，以防破坏原始图像数据
        cv::Mat processed_frame = frame.clone();

        // 调用装甲板检测函数
        process_armor_detection(processed_frame);

        // 发布处理后的图像
        sensor_msgs::msg::Image::SharedPtr processed_msg = cv_bridge::CvImage(header, "bgr8", processed_frame).toImageMsg();
        processed_image_pub_->publish(*processed_msg);
    }

    // 定义一个结构体来更好地表示灯条
    struct LightBar
    {
        cv::RotatedRect rect;
        cv::Point2f center;
        double angle;
        double height;

        LightBar(const cv::RotatedRect &r) : rect(r)
        {
            center = r.center;
            height = (r.size.width > r.size.height) ? r.size.width : r.size.height;
            angle = (r.size.width > r.size.height) ? r.angle + 90 : r.angle;
        }
    };

    std::pair<std::string, double> run_inference(cv::Mat &roi, cv::dnn::Net &net)
    {
        // 在 run_inference 函数或类成员中定义这个映射
        const std::vector<std::string> class_names = {
            "1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"};

        if (roi.empty() || roi.cols <= 0 || roi.rows <= 0)
        {
            return {"", 0.0}; // 对于无效的ROI，返回空结果
        }

        // ==================== 关键修复 ====================
        // 创建ROI的副本，避免修改原始图像
        cv::Mat roi_copy = roi.clone();
        // ================================================

        // 1. 预处理图像
        cv::Mat gray, resized, input_blob;
        cv::cvtColor(roi_copy, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray, 50, 255, cv::THRESH_BINARY);

        // 调整大小以匹配模型输入 (例如 20x28)
        cv::resize(gray, resized, cv::Size(20, 28));
        cv::imshow("resized", resized);
        cv::waitKey(1);

        // 创建blob
        input_blob = cv::dnn::blobFromImage(resized, 1.0 / 127.5, cv::Size(20, 28), cv::Scalar(127.5), false, false);

        // 2. 运行推理
        net.setInput(input_blob);
        cv::Mat outputs = net.forward();

        // --- 手动应用Softmax将Logits转为概率 ---
        cv::Mat probabilities;
        cv::exp(outputs, probabilities);
        probabilities /= cv::sum(probabilities)[0];

        // 3. 处理输出
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(probabilities, nullptr, &confidence, nullptr, &class_id_point);

        // 如果置信度高于阈值
        if (confidence > 0.8)
        {
            int class_id = class_id_point.x;
            if (class_id >= 0 && static_cast<size_t>(class_id) < class_names.size())
            {
                // 4. 返回识别到的类别名称和置信度
                return {class_names[class_id], confidence};
            }
        }

        // 如果置信度低或未识别，返回空结果
        return {"", 0.0};
    }

    void process_armor_detection(cv::Mat &frame)
    {
        // --- 1. 预处理 ---
        cv::Mat gray, mask;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

        double scale = 0.4;
        cv::Mat show;
        cv::resize(mask, show, cv::Size(), scale, scale);
        cv::imshow("mask", show);
        cv::waitKey(1);

        // --- 2. 查找轮廓并筛选灯条 ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::RotatedRect> light_bars;
        for (const auto &contour : contours)
        {
            if (contour.size() < 5)
                continue;

            cv::RotatedRect fitted_ellipse = cv::fitEllipse(contour);
            double contour_area = cv::contourArea(contour);
            // RotatedRect的size成员包含椭圆外接矩形的长和宽，即椭圆的长轴和短轴
            double ellipse_area = CV_PI * (fitted_ellipse.size.width / 2.0) * (fitted_ellipse.size.height / 2.0);

            double area_ratio = contour_area / ellipse_area;

            // 如果面积比偏离1.0太多（比如超过20%），则认为不是一个好的椭圆
            if (std::abs(1.0 - area_ratio) > 0.2)
            {
                continue;
            }

            // --- 筛选标准 3: 长宽比 ---
            // 确保长轴是较长的一边
            double aspect_ratio = (fitted_ellipse.size.width > fitted_ellipse.size.height) ?
             (fitted_ellipse.size.width / fitted_ellipse.size.height) : (fitted_ellipse.size.height / fitted_ellipse.size.width);

            // 我们只想要比较“扁”的椭圆，排除掉近似圆形的物体
            if (aspect_ratio < 1.2 || aspect_ratio > 15.0)
            {
                continue;
            }
            light_bars.push_back(fitted_ellipse);
        }
        std::sort(light_bars.begin(), light_bars.end(), [](const cv::RotatedRect &a, const cv::RotatedRect &b)
                  { return a.center.x < b.center.x; });

        // ==================== 在这里添加绘制代码 ====================
        // 目的：可视化所有被识别为灯条的候选对象
        for (const auto &bar : light_bars)
        {
            cv::Point2f vertices[4];
            bar.points(vertices);
            for (int k = 0; k < 4; k++)
            {
                // 使用红色(BGR)来绘制灯条，以便和绿色的装甲板区分
                cv::line(frame, vertices[k], vertices[(k + 1) % 4], cv::Scalar(0, 0, 255), 2);
            }
        }
        // ==========================================================

        // --- 3. 灯条配对 ---
        for (size_t i = 0; i < light_bars.size(); ++i)
        {
            for (size_t j = i + 1; j < light_bars.size(); ++j)
            {

                const auto &bar1 = light_bars[i];
                const auto &bar2 = light_bars[j];

                // == == == == == == == == == == 优化后的角度筛选逻辑 == == == == == == == == == ==
                // 1. 分别获取两个灯条长边的“归一化”角度。
                //    无论OpenCV如何标记width/height，我们都确保得到的是长边与水平轴的夹角。
                float normalized_angle1 = bar1.size.width < bar1.size.height ? bar1.angle : bar1.angle + 90.0f;
                float normalized_angle2 = bar2.size.width < bar2.size.height ? bar2.angle : bar2.angle + 90.0f;

                // 2. 计算一个能处理“环绕”问题的真实角度差。
                //    这会计算两条线之间的最小夹角。
                float angle_diff = std::abs(normalized_angle1 - normalized_angle2);
                if (angle_diff > 90.0f)
                {
                    angle_diff = 180.0f - angle_diff;
                }

                // 3. 使用优化后的角度差进行判断
                if (angle_diff > 15.0) // 这里的 15.0 阈值现在变得更加可靠，你仍然可以微调它
                    continue;

                float height1 = std::max(bar1.size.width, bar1.size.height);
                float height2 = std::max(bar2.size.width, bar2.size.height);
                if (std::abs(height1 - height2) / std::max(height1, height2) > 0.4)
                    continue;

                float y_diff = std::abs(bar1.center.y - bar2.center.y);
                if (y_diff > std::min(height1, height2) * 1.5)
                    continue;

                float distance = cv::norm(bar1.center - bar2.center);
                float avg_height = (height1 + height2) / 2.0;
                float armor_aspect_ratio = distance / avg_height;
                if (armor_aspect_ratio < 1.5 || armor_aspect_ratio > 3.5)
                    continue;

                // 确定装甲板的旋转矩形
                cv::Point2f bar1_pts[4], bar2_pts[4];
                bar1.points(bar1_pts);
                bar2.points(bar2_pts);
                std::vector<cv::Point2f> armor_pts;
                for (int k = 0; k < 4; ++k)
                {
                    armor_pts.push_back(bar1_pts[k]);
                    armor_pts.push_back(bar2_pts[k]);
                }
                cv::RotatedRect armor_rotated_rect = cv::minAreaRect(armor_pts);

                if (armor_rotated_rect.size.width <= 0 || armor_rotated_rect.size.height <= 0)
                    break;

                // ==================== 在这里添加绘制代码 ====================
                // 目的：可视化所有已成功配对的装甲板候选框
                   /* cv::Point2f paired_vertices[4];
                    armor_rotated_rect.points(paired_vertices);
                    for (int k = 0; k < 4; k++)
                    {
                        // 使用明橙色(BGR: 0, 165, 255)来绘制已配对但尚未识别的装甲板
                        cv::line(frame, paired_vertices[k], paired_vertices[(k + 1) % 4], cv::Scalar(0, 165, 255), 2);
                    }*/
                // ==========================================================

                // --- 提取数字ROI ---
                // 1. 计算可靠的角度
                cv::Point2f delta = bar2.center - bar1.center;
                float reliable_angle = std::atan2(delta.y, delta.x) * 180.0 / CV_PI;

                // 2. 定义ROI尺寸
                const float digit_aspect_ratio = 20.0 / 28.0;
                float roi_height = avg_height * 2.5;
                float roi_width = roi_height * digit_aspect_ratio;
                cv::Size2f roi_size(roi_width, roi_height);

                // 3. 仿射变换以扶正ROI
                cv::Mat rotation_mat = cv::getRotationMatrix2D(armor_rotated_rect.center, reliable_angle, 1.0);
                cv::Mat warped_frame;
                cv::warpAffine(frame, warped_frame, rotation_mat, frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

                // 4. 提取ROI
                cv::Mat digit_roi;
                cv::getRectSubPix(warped_frame, roi_size, armor_rotated_rect.center, digit_roi);

                // --- 进行推理并根据结果决定是否绘制 ---
                if (!digit_roi.empty())
                {
                    // cv::imshow("Digit ROI", digit_roi);
                    // cv::waitKey(1);

                    // 1. 先调用推理函数获取结果
                    // (确保你已经更新了 run_inference 函数，使其返回 std::pair<std::string, double>)
                    auto result = run_inference(digit_roi, net_);
                    std::string class_name = result.first;
                    double confidence = result.second;

                    // 2. 检查识别结果是否有效且不是 "9neg"
                    if (!class_name.empty() && class_name != "9neg")
                    {
                        // ==================== PNP 解算代码 ====================
                        // 1. 获取图像中的2D角点
                        cv::Point2f image_points_raw[4];
                        armor_rotated_rect.points(image_points_raw);

                        // 关键: 对2D点进行排序，以确保与3D模型点的顺序(左下,右下,右上,左上)一致
                        std::vector<cv::Point2f> image_points = {image_points_raw[0], image_points_raw[1], image_points_raw[2], image_points_raw[3]};
                        // 按y坐标排序，y小的在上，y大的在下
                        std::sort(image_points.begin(), image_points.end(), [](const cv::Point2f &a, const cv::Point2f &b)
                                  { return a.y < b.y; });
                        // 对上下两对点分别按x坐标排序
                        if (image_points[0].x > image_points[1].x)
                            std::swap(image_points[0], image_points[1]); // 上方两个点 (左上, 右上)
                        if (image_points[2].x < image_points[3].x)
                            std::swap(image_points[2], image_points[3]); // 下方两个点 (右下, 左下)

                        // 最终顺序调整为：左下, 右下, 右上, 左上
                        std::vector<cv::Point2f> sorted_image_points = {image_points[3], image_points[2], image_points[1], image_points[0]};

                        // 2. 调用 solvePnP
                        cv::Mat rvec, tvec;
                        // 使用我们唯一的3D模型 armor_points_
                        cv::solvePnP(armor_points_, sorted_image_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);

                        // 3. 输出结果
                        // tvec 是平移向量，单位与你定义的3D模型单位一致 (米)
                        double distance = cv::norm(tvec);
                        RCLCPP_INFO(this->get_logger(), "Armor [%s] detected. Position (X,Y,Z): [%.3f, %.3f, %.3f] m. Distance: %.3f m",
                                    class_name.c_str(), tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2), distance);

                        // ==================== 绘制和显示代码 (保留) ====================
                        // 绘制矩形框
                        cv::Point2f vertices[4];
                        armor_rotated_rect.points(vertices);
                        for (int k = 0; k < 4; k++)
                            cv::line(frame, vertices[k], vertices[(k + 1) % 4], cv::Scalar(0, 255, 0), 2);

                        // 绘制标签
                        std::string label = cv::format("%s: %.2f", class_name.c_str(), confidence);
                        int baseLine;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        cv::putText(frame, label,
                                    cv::Point(armor_rotated_rect.boundingRect().x, armor_rotated_rect.boundingRect().y - labelSize.height),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                        // (可选) 在图像上绘制坐标轴以可视化位姿
                        cv::drawFrameAxes(frame, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1); // 绘制一个10cm长的坐标轴
                    }
                }
            }
        }
    }
    // ==================== 成员变量 ====================
    void *handle_;
    bool is_connected_;
    MV_CC_DEVICE_INFO_LIST stDeviceList_;
    cv::dnn::Net net_;
    std::string onnx_path_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr grab_timer_;
    rclcpp::TimerBase::SharedPtr param_sync_timer_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    std::map<std::string, MvGvspPixelType> pixel_format_map_;
    std::string input_source_;
    std::string video_path_;
    cv::VideoCapture cap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_image_pub_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> armor_points_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<ArmorDetectorNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}