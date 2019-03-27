#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
using namespace cv;

int main(int argc, char** argv)
{
	Mat image = imread("1.jpeg");
	namedWindow("Display window", CV_WINDOW_AUTOSIZE);//自动适应图片的大小
	imshow("Display window", image);
	
	char* imageName = "1.jpeg";
	Mat gray_image;
	//从RGB变为灰度
	cvtColor(image, gray_image, CV_BGR2GRAY);
	//存储转换后的文件
	imwrite("Gray_1.jpg", gray_image);
	//namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
	imshow("Gray_1.jpg", gray_image);
	waitKey(0);

	return 0;
}