#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

#define size 3//卷积核大小
#define sigma 1//sigma越大，平滑效果越明显

double gaus[size][size];

void GetgaussianKernel()
{
	const double PI = 4.0*atan(1.0);//π
	int center = size / 2;
	double sum = 0;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			//忽略公式中的(1 / (2 * PI*sigma*sigma))*，不影响结果
			gaus[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	//cout << "低通滤波矩阵：" << endl << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] /= sum;
			//cout << gaus[i][j] << "  ";
		}
		//cout << endl << endl;
	}
}

int main()
{
	Mat src_image1 = imread("cat.bmp");
	Mat src_image2 = imread("dog.bmp");
	//判断图像是否加载成功
	if (src_image1.data && src_image2.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		system("pause");
		return -1;
	}
	namedWindow("src_image1", WINDOW_AUTOSIZE);
	namedWindow("src_image2", WINDOW_AUTOSIZE);
	imshow("src_image1", src_image1);
	imshow("src_image2", src_image2);

	GetgaussianKernel();//生成高斯卷积核矩阵
	Mat gaus_kern = Mat(size, size, CV_64F, gaus);
	Mat lap_kern = (Mat_<double>(3, 3) << 
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1);
	Mat dst1_low, dst2_high,dst;

	filter2D(src_image1, dst1_low, src_image1.depth(), gaus_kern);
	namedWindow("low_pass", WINDOW_AUTOSIZE);
	imshow("low_pass", dst1_low);

	//filter2D(src_image2, dst2_low, src_image2.depth(), kern);
	////addWeighted方法进行图像相减
	//addWeighted(src_image2, 1, dst2_low, -1, 0, dst2_high);
	filter2D(src_image2, dst2_high, src_image2.depth(), lap_kern);
	namedWindow("high_pass", WINDOW_AUTOSIZE);
	imshow("high_pass", dst2_high);

	namedWindow("hybrid_image", WINDOW_AUTOSIZE);
	addWeighted(dst1_low, 1, dst2_high, 1, 0, dst);
	imshow("hybrid_image", dst);

	waitKey(0);
	return 0;
}