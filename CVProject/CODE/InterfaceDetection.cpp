#include"InterfaceDetection.hpp"
#include <opencv2/core/utils/logger.hpp>
//获取版本号
void module_getversion(unsigned char* version)
{
	std::string VID = "V.2023A001";
	for (int i = 0; i < 10; i++)
	{
		version[i] = VID[i];
	}
}
//初始化检测器并加载参数
int module_init(const string humandetFilepath,const string facedetFilepath, int moduleLevel)
{
	//低配 人体检测
	if (moduleLevel == LOWCONFIGURATION) {
		g_ModueLevel = LOWCONFIGURATION;
		ifstream f(humandetFilepath.c_str());
		bool rtn1 = f.good();
		if (rtn1 == false)
		{
			return -5;
		}
		Net_config nano_nets = { 0.5, 0.5, humandetFilepath.c_str()};
		g_Nanodet_plus = new NanoDet_Plus(nano_nets);
		return NORMAL;
	}
	//中配 人体检测+人脸遮挡检测
	else if (moduleLevel == MEDIUMCONFIGURATION)
	{
		g_ModueLevel = MEDIUMCONFIGURATION;
		ifstream f1(humandetFilepath.c_str());
		ifstream f2(facedetFilepath.c_str());
		bool rtn1 = f1.good();
		bool rtn2 = f2.good();
		if ( rtn1 == false || rtn2 == false)
		{
			return MODULLOADEERROR;
		}
		Net_config nano_nets = { 0.5, 0.5, humandetFilepath.c_str() };
		g_Nanodet_plus = new NanoDet_Plus(nano_nets);
		Net_config face_nets = { 0.5, 0.5, facedetFilepath.c_str() };
		g_Yolov7_face = new YOLOV7_face(face_nets);
		return NORMAL;
	}
	else {
		return CONFIGURATIONERROR;
	}
	
}
//模型检测
int module_detection(const string filename)
{
	int iResult_human = 0;
	int iResult_face = 0;

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);	//只输出错误日志
	Mat srcimg = imread(filename);
	if (srcimg.data == NULL)
		return IMAGEERROR;													//图像数据读取出错

	if (g_ModueLevel == LOWCONFIGURATION)
	{
		if (g_Nanodet_plus == nullptr)
			return MODULDEFINEEERROR;												//模型定义出错
		iResult_human = g_Nanodet_plus->detect(srcimg);
		if (iResult_human == 0)
			return CAMERAOCCLUSION;												//摄像头被遮挡
		else
			return NORMAL;												//正常
	}
	else if (g_ModueLevel == MEDIUMCONFIGURATION)
	{
		if (g_Nanodet_plus == nullptr || g_Yolov7_face == nullptr)
		{
			return MODULDEFINEEERROR;												//模型定义出错
		}
		int iResult_face = g_Yolov7_face->detect(srcimg);
		if (iResult_face > 0)
		{
			return NORMAL;                                              //检测到人脸，有人脸就一定有人体
		}
		else
		{   //未检测到人脸 ，检人体
			iResult_human = g_Nanodet_plus->detect(srcimg);
			if (iResult_face > 0)  //有人体
			{
				return FACEOCCLUSION;         //人脸遮挡
			}
			else {
				return CAMERAOCCLUSION;        //摄像头遮挡
			}
		}
	}
	else
	{
		return CONFIGURATIONERROR;
	}
}

//int main()
//{
//	vector<BoxInfo> iResultbox;
//	const string filepath_human = "weights\\nanodet-plus-m_320.onnx";
//	const string filepath_face = "weights\\yolov7-lite-face.onnx";
//	bool rtn1 = module_init(filepath_human, filepath_face,MEDIUMCONFIGURATION);
//
//	int rtn2 = module_detection("images/zidane.jpg");
//
//	return rtn2;
//
//}