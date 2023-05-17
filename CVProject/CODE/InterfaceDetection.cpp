#include"InterfaceDetection.hpp"
#include <opencv2/core/utils/logger.hpp>
//��ȡ�汾��
void module_getversion(unsigned char* version)
{
	std::string VID = "V.2023A001";
	for (int i = 0; i < 10; i++)
	{
		version[i] = VID[i];
	}
}
//��ʼ������������ز���
int module_init(const string humandetFilepath,const string facedetFilepath, int moduleLevel)
{
	//���� ������
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
	//���� ������+�����ڵ����
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
//ģ�ͼ��
int module_detection(const string filename)
{
	int iResult_human = 0;
	int iResult_face = 0;

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);	//ֻ���������־
	Mat srcimg = imread(filename);
	if (srcimg.data == NULL)
		return IMAGEERROR;													//ͼ�����ݶ�ȡ����

	if (g_ModueLevel == LOWCONFIGURATION)
	{
		if (g_Nanodet_plus == nullptr)
			return MODULDEFINEEERROR;												//ģ�Ͷ������
		iResult_human = g_Nanodet_plus->detect(srcimg);
		if (iResult_human == 0)
			return CAMERAOCCLUSION;												//����ͷ���ڵ�
		else
			return NORMAL;												//����
	}
	else if (g_ModueLevel == MEDIUMCONFIGURATION)
	{
		if (g_Nanodet_plus == nullptr || g_Yolov7_face == nullptr)
		{
			return MODULDEFINEEERROR;												//ģ�Ͷ������
		}
		int iResult_face = g_Yolov7_face->detect(srcimg);
		if (iResult_face > 0)
		{
			return NORMAL;                                              //��⵽��������������һ��������
		}
		else
		{   //δ��⵽���� ��������
			iResult_human = g_Nanodet_plus->detect(srcimg);
			if (iResult_face > 0)  //������
			{
				return FACEOCCLUSION;         //�����ڵ�
			}
			else {
				return CAMERAOCCLUSION;        //����ͷ�ڵ�
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