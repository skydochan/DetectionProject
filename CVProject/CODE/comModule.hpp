#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
//#include <onnxruntime_c_api.h>


using namespace cv;
using namespace std;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};
typedef struct PointInfo
{
	Point pt;
	float score;
} PointInfo;

typedef struct FaceBoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	PointInfo kpt1;
	PointInfo kpt2;
	PointInfo kpt3;
	PointInfo kpt4;
	PointInfo kpt5;
} FaceBoxInfo;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class NanoDet_Plus
{
public:
	NanoDet_Plus(Net_config config);
	int detect(Mat& srcimg);
private:
	float score_threshold = 0.5;
	float nms_threshold = 0.5;
	vector<string> class_names;
	int num_class;

	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	vector<float> input_image_;
	void normalize_(Mat img);
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(vector<BoxInfo>& generate_boxes, const float* preds);
	void nms(vector<BoxInfo>& input_boxes);
	const bool keep_ratio = false;
	int inpWidth;
	int inpHeight;
	int reg_max;
	const int num_stages = 4;
	const int stride[4] = { 8,16,32,64 };
	const float mean[3] = { 103.53, 116.28, 123.675 };
	const float stds[3] = { 57.375, 57.12, 58.395 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "nanodetplus");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

class YOLOV7_face
{
public:
	YOLOV7_face(Net_config config);
	int detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<FaceBoxInfo>& input_boxes);
	bool has_postprocess;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV7_face");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};