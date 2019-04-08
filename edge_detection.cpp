#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

//做边缘检测不需要图像太过模糊
#define size 3//卷积核大小
#define sigma 1//sigma越大，平滑效果越明显
//双阈值检测
#define high_threshold 100
#define low_threshold 2

double gaus[size][size];

bool Max(uchar x, uchar a, uchar b)
{
	//考虑是否可以等
	if ((double)x >= (double)a && (double)x >= (double)b)
		return true;
	return false;
}

void GetgaussianKernel()
{
	const double PI = 4.0*atan(1.0);//π
	int center = size / 2;
	double sum = 0;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			gaus[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			gaus[i][j] /= sum;
}

Mat GaussianFilter(Mat src)
{
	Mat dst;
	GetgaussianKernel();//生成高斯卷积核矩阵
	Mat gaus_kern = Mat(size, size, CV_64F, gaus);
	filter2D(src, dst, src.depth(), gaus_kern);
	namedWindow("gaussian_filter", WINDOW_AUTOSIZE);
	imshow("gaussian_filter", dst);
	return dst;
}

//计算梯度值和梯度方向
void SobelFilter(Mat src, Mat val, Mat dir)
{
	Mat sobel_x = (Mat_<uchar>(3, 3) <<
		1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	Mat sobel_y = (Mat_<uchar>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		1, -2, -1);
	Mat gradient_x, gradient_y;
	filter2D(src, gradient_x, src.depth(), sobel_x);
	filter2D(src, gradient_y, src.depth(), sobel_y);
	//x,y平方和开方计算量过大，用两数绝对值来代替
	//精度可能不够会影响结果
	val = abs(gradient_x) + abs(gradient_y);

	int row = src.rows;
	int col = src.cols;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			dir.at<uchar>(i, j) = atan2(gradient_y.at<uchar>(i, j), gradient_x.at<uchar>(i, j));
			//dir.at<uchar>(i, j) = atan2(gradient_x.at<uchar>(i, j), gradient_y.at<uchar>(i, j));
}

//val不符合要求的值置为0
void NonMaxSuppression(Mat val, Mat dir)
{
	//原图片扩展0，否则后面循环矩阵越界
	copyMakeBorder(val, val, 1, 1, 1, 1, BorderTypes::BORDER_CONSTANT);
	copyMakeBorder(dir, dir, 1, 1, 1, 1, BorderTypes::BORDER_CONSTANT);
	int row = val.rows;
	int col = val.cols;

	for (int i = 1; i < row - 1; i++)
	{
		for (int j = 1; j < col - 1; j++)
		{
			if ((dir.at<uchar>(i, j) >= -22.5 && dir.at<uchar>(i, j) < 22.5)
				| (dir.at<uchar>(i, j) >= 157.5 && dir.at<uchar>(i, j) <= 180)
				| (dir.at<uchar>(i, j) < -157.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i + 1, j), val.at<uchar>(i - 1, j)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<uchar>(i, j) >= 22.5 && dir.at<uchar>(i, j) < 67.5)
				| (dir.at<uchar>(i, j) >= -157.5 && dir.at<uchar>(i, j) < -112.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i + 1, j + 1), val.at<uchar>(i - 1, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<uchar>(i, j) >= 67.5 && dir.at<uchar>(i, j) < 112.5)
				| (dir.at<uchar>(i, j) >= -112.5 && dir.at<uchar>(i, j) < -67.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i, j + 1), val.at<uchar>(i, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<uchar>(i, j) >= 112.5 && dir.at<uchar>(i, j) < 157.5)
				| (dir.at<uchar>(i, j) >= -67.5 && dir.at<uchar>(i, j) < -22.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i - 1, j + 1), val.at<uchar>(i + 1, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
		}
	}
}

void ThresholdDetection(Mat val, Mat edge)
{
	copyMakeBorder(edge, edge, 1, 1, 1, 1, BorderTypes::BORDER_CONSTANT);
	int row = val.rows;
	int col = val.cols;

	//确定一定是边的
	for (int i = 1; i < row - 1; i++)
	{
		for (int j = 1; j < col - 1; j++)
		{
			if (val.at<uchar>(i, j) >= high_threshold)
				edge.at<uchar>(i, j) = 2;//一定是边
			else if (val.at<uchar>(i, j) >= low_threshold)
				edge.at<uchar>(i, j) = 1;//可能是边
			else
				edge.at<uchar>(i, j) = 0;//一定不是边
		}
	}
	//二次遍历，确定所有边
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if ((edge.at<uchar>(i, j) == 1) &&
				((edge.at<uchar>(i + 1, j) == 2) | (edge.at<uchar>(i - 1, j) == 2)
					| (edge.at<uchar>(i, j + 1) == 2) | (edge.at<uchar>(i, j - 1) == 2)))
				edge.at<uchar>(i, j) = 1;
			else
				edge.at<uchar>(i, j) = 0;
		}
	}
}

int main()
{
	Mat src = imread("dog.bmp");
	//判断图像是否加载成功
	if (src.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		system("pause");
		return -1;
	}

	Mat src_gray, src_low, gradient_val, gradient_dir, edge;

	//灰度处理，高斯滤波
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_low = GaussianFilter(src_gray);

	//Sobel算子计算梯度
	gradient_val = src_gray.clone();
	gradient_dir = src_gray.clone();
	SobelFilter(src_gray, gradient_val, gradient_dir);

	//非极大值抑制
	NonMaxSuppression(gradient_val, gradient_dir);

	//双阈值检测
	edge = gradient_val.clone();
	ThresholdDetection(gradient_val, edge);

	//最后确定边缘
	int row = gradient_val.rows;
	int col = gradient_val.cols;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			if (edge.at<uchar>(i,j) == 0)
				gradient_val.at<uchar>(i,j) = 0;

	namedWindow("edge_detection", WINDOW_AUTOSIZE);
	imshow("edge_detection", gradient_val);

	waitKey(0);
	return 0;
}