#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

//做边缘检测不需要图像太过模糊
#define k_size 3//卷积核大小
#define sigma 1//sigma越大，平滑效果越明显
//双阈值检测
#define high_threshold 100
#define low_threshold 40

const double PI = 4.0*atan(1.0);//π
double gaus[k_size][k_size];

bool Max(uchar x, uchar a, uchar b)
{
	//考虑是否可以等
	if ((double)x >= (double)a && (double)x >= (double)b)
		return true;
	return false;
}

void GetgaussianKernel()
{
	int center = k_size / 2;
	double sum = 0;

	for (int i = 0; i < k_size; i++)
	{
		for (int j = 0; j < k_size; j++)
		{
			gaus[i][j] = exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gaus[i][j];
		}
	}
	for (int i = 0; i < k_size; i++)
		for (int j = 0; j < k_size; j++)
			gaus[i][j] /= sum;
}

Mat GaussianFilter(Mat src)
{
	Mat dst;
	GetgaussianKernel();//生成高斯卷积核矩阵
	Mat gaus_kern = Mat(k_size, k_size, CV_64F, gaus);
	filter2D(src, dst, src.depth(), gaus_kern);
	//namedWindow("gaussian_filter", WINDOW_AUTOSIZE);
	//imshow("gaussian_filter", dst);
	return dst;
}

//计算梯度值和梯度方向
void SobelFilter(Mat src, Mat &val, Mat &dir)
{
	//Mat sobel_x = (Mat_<uchar>(3, 3) <<
	//	1, 0, -1,
	//	2, 0, -2,
	//	1, 0, -1);
	//Mat sobel_y = (Mat_<uchar>(3, 3) <<
	//	1, 2, 1,
	//	0, 0, 0,
	//	1, -2, -1);
	//Mat gx, gy;
	//filter2D(src, gx, src.depth(), sobel_x);
	//filter2D(src, gy, src.depth(), sobel_y);

	Mat gx, gy;
	//转换成double
	src.convertTo(src, CV_64F);
	Sobel(src, gx, src.depth(), 1, 0, 3);
	Sobel(src, gy, src.depth(), 0, 1, 3);
	gx.convertTo(gx, CV_64F);
	gy.convertTo(gy, CV_64F);

	magnitude(gx, gy, val);
	val.convertTo(val, CV_8U);
	int row = src.rows;
	int col = src.cols;
	//int temp;
	//for (int i = 0; i < row; i++)
	//{
	//	for (int j = 0; j < col; j++)
	//	{
	//		temp = gx.at<double>(i, j) * gx.at<double>(i, j) + gy.at<double>(i, j) * gy.at<double>(i, j);
	//		val.at<uchar>(i, j) = sqrt(temp);
	//	}
	//}

	//默认返回的是弧度制，换算成角度制
	//uchar没有负数，错误,结果没有负数
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			dir.at<double>(i, j) = atan2(gy.at<double>(i, j), gx.at<double>(i, j)) * 180/PI;
			//dir.at<double>(i, j) = atan2(gx.at<uchar>(i, j), gy.at<uchar>(i, j))* 180/PI;

	//-------------debug-------------------
	//int r1 = gx.rows;
	//int c1 = gx.cols;
	//for (int i = 0; i < r1; i++)
	//{
	//	for (int j = 0; j < c1; j++)
	//		cout << dir.at<double>(i, j) << " ";
	//	cout << endl;
	//}
	//--------------------------------------
}

//val不符合要求的值置为0
void NonMaxSuppression(Mat &val, Mat dir)
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
			if ((dir.at<double>(i, j) >= -22.5 && dir.at<double>(i, j) < 22.5)
				| (dir.at<double>(i, j) >= 157.5 && dir.at<double>(i, j) <= 180)
				| (dir.at<double>(i, j) < -157.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i + 1, j), val.at<uchar>(i - 1, j)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<double>(i, j) >= 22.5 && dir.at<double>(i, j) < 67.5)
				| (dir.at<double>(i, j) >= -157.5 && dir.at<double>(i, j) < -112.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i + 1, j + 1), val.at<uchar>(i - 1, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<double>(i, j) >= 67.5 && dir.at<double>(i, j) < 112.5)
				| (dir.at<double>(i, j) >= -112.5 && dir.at<double>(i, j) < -67.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i, j + 1), val.at<uchar>(i, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
			else if ((dir.at<double>(i, j) >= 112.5 && dir.at<double>(i, j) < 157.5)
				| (dir.at<double>(i, j) >= -67.5 && dir.at<double>(i, j) < -22.5))
			{
				if (!Max(val.at<uchar>(i, j), val.at<uchar>(i - 1, j + 1), val.at<uchar>(i + 1, j - 1)))
					val.at<uchar>(i, j) = 0;
			}
		}
	}
	//==========debug==========
	//for (int i = 0; i < row; i++)
	//{
	//	for (int j = 0; j < col; j++)
	//		cout << (int)val.at<uchar>(i, j) << " ";
	//	cout << endl;
	//}
	//========================
}

void ThresholdDetection(Mat val, Mat &edge)
{
	int row = val.rows;
	int col = val.cols;

	//确定一定是边的
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (val.at<uchar>(i, j) >= high_threshold)
				edge.at<uchar>(i, j) = 2;//一定是边
			else if (val.at<uchar>(i, j) < low_threshold)
				edge.at<uchar>(i, j) = 0;//一定不是边
			else 
				edge.at<uchar>(i, j) = 1;//可能是边的
		}
	}	
	//深度优先遍历，确定所有边
	bool changed;
	do
	{
		changed = false;
		for (int i = 1; i < row - 1; i++)
		{
			for (int j = 1; j < col - 1; j++)
			{
				if (edge.at<uchar>(i, j) == 2)
				{
					if (edge.at<uchar>(i + 1, j) == 1)
					{
						edge.at<uchar>(i + 1, j) = 2;
						changed = true;
					}
					if (edge.at<uchar>(i - 1, j) == 1)
					{
						edge.at<uchar>(i - 1, j) = 2;
						changed = true;
					}
					if (edge.at<uchar>(i, j + 1) == 1)
					{
						edge.at<uchar>(i, j + 1) = 2;
						changed = true;
					}
					if (edge.at<uchar>(i, j - 1) == 1)
					{
						edge.at<uchar>(i, j - 1) = 2;
						changed = true;
					}
				}
			}
		}
	} while (changed);

	//-------------debug-------------------
	//for (int i = 0; i < row; i++)
	//{
	//	for (int j = 0; j < col; j++)
	//		cout << (int)edge.at<uchar>(i, j) << " ";
	//	cout << endl;
	//}
	//--------------------------------------
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

	Mat src_gray, src_low, gradient_val, edge;

	//灰度处理，高斯滤波
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_low = GaussianFilter(src_gray);

	//Sobel算子计算梯度
	gradient_val = src_low.clone();
	Mat gradient_dir(src_low.rows, src_low.cols, CV_64F);
	SobelFilter(src_low, gradient_val, gradient_dir);
	//namedWindow("sobel", WINDOW_AUTOSIZE);
	//imshow("sobel", gradient_val);

	//非极大值抑制
	NonMaxSuppression(gradient_val, gradient_dir);
	//namedWindow("NMS", WINDOW_AUTOSIZE);
	//imshow("NMS", gradient_val);

	//双阈值检测
	edge = gradient_val.clone();
	ThresholdDetection(gradient_val, edge);

	//最后确定边缘
	int row = gradient_val.rows;
	int col = gradient_val.cols;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (edge.at<uchar>(i, j) == 0 || edge.at<uchar>(i, j) == 1)
				gradient_val.at<uchar>(i, j) = 0;
			//else
			//	gradient_val.at<uchar>(i, j) = 255;
		}
	}
	namedWindow("edge_detection", WINDOW_AUTOSIZE);
	imshow("edge_detection", gradient_val);

	waitKey(0);
	return 0;
}