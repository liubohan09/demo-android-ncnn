//
// Create by liubohan
// 2023 / 1 / 23
//

#ifndef AFtigernet_H
#define AFtigernet_H

#include "net.h"
#include "YoloV5.h"

typedef struct HeadInfo_
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct CenterPrior_
{
    int x;
    int y;
    int stride;
} CenterPrior;


class AFtigernet{
public:
    AFtigernet(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);

    ~AFtigernet();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float score_threshold, float nms_threshold);

//    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//                                    "hair drier", "toothbrush"};

    std::vector<std::string> labels{"tiger"};
private:
    void preprocess(JNIEnv *env, jobject image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results, float width_ratio, float height_ratio);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, float width_ratio, float height_ratio);

    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

    ncnn::Net *Net;
    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = {416, 416}; // input height and width
    int num_class = 1; // number of classes. 80 for COCO
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.


public:
    static AFtigernet *detector;
    static bool hasGPU;
//    static bool toUseGPU;
};


#endif //AFtigernet_H
