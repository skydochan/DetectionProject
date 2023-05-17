#include "comModule.hpp"

//����
#define      LOWCONFIGURATION             1  //����   ����ͷ�ڵ���ҹ���������
#define      MEDIUMCONFIGURATION          2  //����   ����ͷ�ڵ� + �����ڵ���ҹ���������
#define      HIGHCONFIGURATION            3  //����   ˫Ŀ  ����ͷ�ڵ� + �����ڵ�

//������
#define      NORMAL                       1 //����
#define      CAMERAOCCLUSION             -1 //����ͷ���ڵ�
#define      FACEOCCLUSION               -2 //�������ڵ�
#define      IMAGEERROR                  -3 //ͼ�������ȡ����������·������ȷ
#define      MODULDEFINEEERROR           -4 //ģ�Ͷ������
#define      MODULLOADEERROR             -5 //ģ�ͼ��س���
#define      CONFIGURATIONERROR          -6 //���ô���Ŀǰֻ֧�ֵ��������


int g_ModueLevel = LOWCONFIGURATION;
NanoDet_Plus * g_Nanodet_plus = nullptr;
YOLOV7_face * g_Yolov7_face = nullptr;


extern "C" _declspec(dllexport) void module_getversion(unsigned char* version);
extern "C" _declspec(dllexport) int module_init(const string humandetFilepath, const string facedetFilepath, int moduleLevel);
extern "C" _declspec(dllexport) int module_detection(const string filename);
