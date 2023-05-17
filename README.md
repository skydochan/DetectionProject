# DetectionProject
这是一个生成人体检测与人脸检测dll的工程。其中在dll中定义了3个接口，分别是：

//获取版本号

extern "C" LIB_API void module_getversion(unsigned char* version);

//初始化检测器并加载参数

extern "C" LIB_API int module_init(const string humandetFilepath, const string facedetFilepath, int moduleLevel);

//检测函数

extern "C" LIB_API int module_detection(const string filename);
