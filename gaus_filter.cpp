#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

#define size 5//卷积核大小
#define sigma 1//卷积核的标准差大小

double gaus[size][size];

void GetGaussianKernel()
{
	const double PI = 4.0*atan(1.0);//π
	int center = size / 2;
	double sum = 0;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//(ceter,ceter)为中心点坐标
			gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	cout << "卷积核矩阵如下：" << endl;
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
	Mat srcImage = imread("marilyn.bmp");
	//判断图像是否加载成功
	if (srcImage.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		system("pause");
		return -1;
	}
	namedWindow("srcImage", WINDOW_AUTOSIZE);
	imshow("srcImage", srcImage);

	GetGaussianKernel();//生成高斯卷积核矩阵
	Mat kern = Mat(size, size, CV_64F, gaus);

	Mat dstImage;
	filter2D(srcImage, dstImage, srcImage.depth(), kern);
	namedWindow("dstImage", WINDOW_AUTOSIZE);
	imshow("dstImage", dstImage);

	waitKey(0);
	return 0;
}