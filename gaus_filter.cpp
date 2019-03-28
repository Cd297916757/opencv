#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

#define size 3//卷积核大小
#define sigma 1.5//sigma越大，平滑效果越明显

double gaus[size][size];

void GetgaussianKernel()
{
	const double PI = 4.0*atan(1.0);//π
	int center = size / 2;
	double sum = 0;
	//高斯低通滤波
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//忽略公式中的(1 / (2 * PI*sigma*sigma))*，不影响结果
			gaus[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	cout << "低通滤波矩阵：" << endl << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] /= sum;
			cout << gaus[i][j] << "  ";
		}
		cout << endl << endl;
	}
}

int main()
{
	Mat src_image = imread("einstein.bmp");
	//判断图像是否加载成功
	if (src_image.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		system("pause");
		return -1;
	}
	namedWindow("src_image", WINDOW_AUTOSIZE);
	imshow("src_image", src_image);

	GetgaussianKernel();//生成高斯卷积核矩阵
	Mat kern = Mat(size, size, CV_64F, gaus);
	Mat dst_image_low,dst_image_high;
	filter2D(src_image, dst_image_low, src_image.depth(), kern);
	//addWeighted方法进行图像相减
	addWeighted(src_image, 1, dst_image_low, -1, 0, dst_image_high);

	namedWindow("low_pass", WINDOW_AUTOSIZE);
	namedWindow("high_pass", WINDOW_AUTOSIZE);
	imshow("low_pass", dst_image_low);
	imshow("high_pass", dst_image_high);

	waitKey(0);
	return 0;
}