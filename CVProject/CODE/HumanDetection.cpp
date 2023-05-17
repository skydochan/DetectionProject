//#define _CRT_SECURE_NO_WARNINGS
#include"comModule.hpp"
string classnames[80] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", 
"firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
"skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle",
"wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
"broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed",
"diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven",
"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


NanoDet_Plus::NanoDet_Plus(Net_config config)
{
	//string classesFile = "class.names";
	//ifstream ifs(classesFile.c_str());
	//string line;
	//while (getline(ifs, line)) this->class_names.push_back(line);
	//this->num_class = class_names.size();
	this->num_class = 80;
	for (int i = 0; i < this->num_class; i++){
		string line = classnames[i];
		this->class_names.push_back(line);
	}
	this->nms_threshold = config.nmsThreshold;
	this->score_threshold = config.confThreshold;
	string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
		/*for (int j = 0; j < output_dims.size(); j++)
		{
			cout << output_dims[j] << ",";
		}
		cout << endl;*/
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->reg_max = (output_node_dims[0][output_node_dims[0].size() - 1] - this->num_class) / 4 - 1;
}

Mat NanoDet_Plus::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void NanoDet_Plus::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				//this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - mean[c] / 255.0) / (stds[c] / 255.0);
				this->input_image_[c * row * col + i * col + j] = (pix - mean[c]) / stds[c];
			}
		}
	}
}

void NanoDet_Plus::softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void NanoDet_Plus::generate_proposal(vector<BoxInfo>& generate_boxes, const float* preds)
{
	const int reg_1max = reg_max + 1;
	const int len = this->num_class + 4 * reg_1max;
	for (int n = 0; n < this->num_stages; n++)
	{
		const int stride_ = this->stride[n];
		const int num_grid_y = (int)ceil((float)this->inpHeight / stride_);
		const int num_grid_x = (int)ceil((float)this->inpWidth / stride_);
		////cout << "num_grid_x=" << num_grid_x << ",num_grid_y=" << num_grid_y << endl;

		for (int i = 0; i < num_grid_y; i++)
		{
			for (int j = 0; j < num_grid_x; j++)
			{
				int max_ind = 0;
				float max_score = 0;
				for (int k = 0; k < num_class; k++)
				{
					if (preds[k] > max_score)
					{
						max_score = preds[k];
						max_ind = k;
					}
				}
				if (max_score >= score_threshold)
				{
					const float* pbox = preds + this->num_class;
					float dis_pred[4];
					float* y = new float[reg_1max];
					for (int k = 0; k < 4; k++)
					{
						softmax_(pbox + k * reg_1max, y, reg_1max);
						float dis = 0.f;
						for (int l = 0; l < reg_1max; l++)
						{
							dis += l * y[l];
						}
						dis_pred[k] = dis * stride_;
					}
					delete[] y;
					/*float pb_cx = (j + 0.5f) * stride_ - 0.5;
					float pb_cy = (i + 0.5f) * stride_ - 0.5;*/
					float pb_cx = j * stride_;
					float pb_cy = i * stride_;
					float x0 = pb_cx - dis_pred[0];
					float y0 = pb_cy - dis_pred[1];
					float x1 = pb_cx + dis_pred[2];
					float y1 = pb_cy + dis_pred[3];
					generate_boxes.push_back(BoxInfo{ x0, y0, x1, y1, max_score, max_ind });
				}
				preds += len;
			}
		}
	}

}

void NanoDet_Plus::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nms_threshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

int NanoDet_Plus::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat cv_image = srcimg.clone();
	Mat dst = this->resize_image(cv_image, &newh, &neww, &top, &left);
	this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	vector<BoxInfo> generate_boxes;
	//模型推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ¿ªÊ¼ÍÆÀí
	/////generate proposals
	const float* preds = ort_outputs[0].GetTensorMutableData<float>();
	generate_proposal(generate_boxes, preds);

	//// Perform non maximum suppression to eliminate redundant overlapping boxes with
	//// lower confidences
	nms(generate_boxes);
	////画框，调试用
	//float ratioh = (float)cv_image.rows / newh;
	//float ratiow = (float)cv_image.cols / neww;
	//for (size_t i = 0; i < generate_boxes.size(); ++i)
	//{
	//	int xmin = (int)max((generate_boxes[i].x1 - left) * ratiow, 0.f);
	//	int ymin = (int)max((generate_boxes[i].y1 - top) * ratioh, 0.f);
	//	int xmax = (int)min((generate_boxes[i].x2 - left) * ratiow, (float)cv_image.cols);
	//	int ymax = (int)min((generate_boxes[i].y2 - top) * ratioh, (float)cv_image.rows);
	//	rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
	//	string label = format("%.2f", generate_boxes[i].score);
	//	label = this->class_names[generate_boxes[i].label] + ":" + label;
	//	putText(srcimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	//}
	//imshow("kWinName", srcimg);
	//waitKey(0);
	int num_human = 0;
	float ratioh = (float)cv_image.rows / newh;
	float ratiow = (float)cv_image.cols / neww;
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		if (0 == generate_boxes[i].label)
		{
			int xmin = (int)max((generate_boxes[i].x1 - left) * ratiow, 0.f);
			int ymin = (int)max((generate_boxes[i].y1 - top) * ratioh, 0.f);
			int xmax = (int)min((generate_boxes[i].x2 - left) * ratiow, (float)cv_image.cols);
			int ymax = (int)min((generate_boxes[i].y2 - top) * ratioh, (float)cv_image.rows);
			if ((xmax - xmin) * (ymax - ymin) > 600)
				num_human++;
		}
	}
	return num_human;
}
