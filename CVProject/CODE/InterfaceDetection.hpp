#include "comModule.hpp"

//配置
#define      LOWCONFIGURATION             1  //低配   摄像头遮挡（夜间检测能力差）
#define      MEDIUMCONFIGURATION          2  //中配   摄像头遮挡 + 人脸遮挡（夜间检测能力差）
#define      HIGHCONFIGURATION            3  //高配   双目  摄像头遮挡 + 人脸遮挡

//错误码
#define      NORMAL                       1 //正常
#define      CAMERAOCCLUSION             -1 //摄像头被遮挡
#define      FACEOCCLUSION               -2 //人脸被遮挡
#define      IMAGEERROR                  -3 //图像输出读取出错，可能是路径不正确
#define      MODULDEFINEEERROR           -4 //模型定义出错
#define      MODULLOADEERROR             -5 //模型加载出错
#define      CONFIGURATIONERROR          -6 //配置错误，目前只支持低配和中配


int g_ModueLevel = LOWCONFIGURATION;
NanoDet_Plus * g_Nanodet_plus = nullptr;
YOLOV7_face * g_Yolov7_face = nullptr;


extern "C" _declspec(dllexport) void module_getversion(unsigned char* version);
extern "C" _declspec(dllexport) int module_init(const string humandetFilepath, const string facedetFilepath, int moduleLevel);
extern "C" _declspec(dllexport) int module_detection(const string filename);
