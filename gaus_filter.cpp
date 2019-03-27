#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;


//******************高斯卷积核生成函数*************************
//第一个参数gaus是一个指向含有3个double类型数组的指针；
//第二个参数size是高斯卷积核的尺寸大小；
//第三个参数sigma是卷积核的标准差
void GetGaussianKernel(double **gaus, const int size, const double sigma)
{
	const double PI = 4.0*atan(1.0); //圆周率π赋值
	int center = size / 2;
	double sum = 0;
	for (int i = 0; i<size; i++)
	{
		for (int j = 0; j<size; j++)
		{
			gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	cout << "卷积核：" << endl;
	for (int i = 0; i<size; i++)
	{
		for (int j = 0; j<size; j++)
		{
			gaus[i][j] /= sum;
			cout << gaus[i][j] << "  ";
		}
		cout << endl << endl;
	}
	return;
}

int main()
{
	const int size = 5; //定义卷积核大小
	const double sigma = 1;//定义sigma大小，决定图片模糊程度
	double **gaus = new double *[size];
	for (int i = 0; i < size; i++)
		gaus[i] = new double[size];  //动态生成矩阵

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

	GetGaussianKernel(gaus, size, sigma); //生成高斯卷积核

	double temp[size][size];
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			temp[i][j] = gaus[i][j];
	Mat kern = Mat(3, 3, CV_64F, temp);

	Mat dstImage;
	filter2D(srcImage, dstImage, srcImage.depth(), kern);
	namedWindow("dstImage", WINDOW_AUTOSIZE);
	imshow("dstImage", dstImage);

	waitKey(0);
	return 0;
}