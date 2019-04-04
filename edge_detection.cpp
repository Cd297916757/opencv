#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int Ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";

//Canny以1:3作为掩码输入
static void CannyThreshold(int, void*)
{
	//用3x3的核减少噪声
	blur(src_gray, detected_edges, Size(3, 3));
	//Canny检测器
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*Ratio, kernel_size);
	//内容全为0
	dst = Scalar::all(0);
	//将src以detected_edges为掩码复制到dst
	src.copyTo(dst, detected_edges);
	//显示掩码后的结果
	imshow(window_name, dst);
}

int main()
{
	src = imread("dog.bmp");
	//判断图像是否加载成功
	if (src.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		system("pause");
		return -1;
	}

	//建立一个和原图大小和形态一致的矩阵
	dst.create(src.size(), src.type());
	//转成灰度
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	namedWindow(window_name, WINDOW_AUTOSIZE);
	//建立滑杆用来作为掩码值得输入
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	//掩码后显示结果
	CannyThreshold(0,0);

	waitKey(0);
	return 0;
}