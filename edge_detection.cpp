#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

#define k_size 3//����˴�С
#define sigma 1//sigmaԽ��ƽ��Ч��Խ����
#define high_threshold 100
//#define low_threshold 40

int low_threshold = 0;
const double PI = 4.0*atan(1.0);//��
double gaus[k_size][k_size];

Mat val, edge, dst;

bool Max(uchar x, uchar a, uchar b)
{
	//�����Ƿ���Ե�
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
	GetgaussianKernel();//���ɸ�˹����˾���
	Mat gaus_kern = Mat(k_size, k_size, CV_64F, gaus);
	filter2D(src, dst, src.depth(), gaus_kern);
	//namedWindow("gaussian_filter", WINDOW_AUTOSIZE);
	//imshow("gaussian_filter", dst);
	return dst;
}

//�����ݶ�ֵ���ݶȷ���
void SobelFilter(Mat src, Mat &dir)
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
	//ת����double
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

	//Ĭ�Ϸ��ص��ǻ����ƣ�����ɽǶ���
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			dir.at<double>(i, j) = atan2(gy.at<double>(i, j), gx.at<double>(i, j)) * 180/PI;
}

//val������Ҫ���ֵ��Ϊ0
void NonMaxSuppression(Mat dir)
{
	//ԭͼƬ��չ0���������ѭ������Խ��
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
}

void ThresholdDetection(int ,void*)
{
	cout << low_threshold << endl;

	dst = val.clone();
	Mat temp = edge.clone();
	int row = val.rows;
	int col = val.cols;

	//ȷ��һ���Ǳߵ�
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (val.at<uchar>(i, j) >= high_threshold)
				temp.at<uchar>(i, j) = 2;//һ���Ǳ�
			else if (val.at<uchar>(i, j) < low_threshold)
				temp.at<uchar>(i, j) = 0;//һ�����Ǳ�
			else 
				temp.at<uchar>(i, j) = 1;//�����Ǳߵ�
		}
	}	
	//������ȱ�����ȷ�����б�
	bool changed;
	do
	{
		changed = false;
		for (int i = 1; i < row - 1; i++)
		{
			for (int j = 1; j < col - 1; j++)
			{
				if (temp.at<uchar>(i, j) == 2)
				{
					if (temp.at<uchar>(i + 1, j) == 1)
					{
						temp.at<uchar>(i + 1, j) = 2;
						changed = true;
					}
					if (temp.at<uchar>(i - 1, j) == 1)
					{
						temp.at<uchar>(i - 1, j) = 2;
						changed = true;
					}
					if (temp.at<uchar>(i, j + 1) == 1)
					{
						temp.at<uchar>(i, j + 1) = 2;
						changed = true;
					}
					if (temp.at<uchar>(i, j - 1) == 1)
					{
						temp.at<uchar>(i, j - 1) = 2;
						changed = true;
					}
				}
			}
		}
	} while (changed);

	//���ȷ����Ե
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (temp.at<uchar>(i, j) == 0 || temp.at<uchar>(i, j) == 1)
				dst.at<uchar>(i, j) = 0;
			//else
			//	val.at<uchar>(i, j) = 255;
		}
	}
	imshow("edge_detection", dst);
}

int main()
{
	Mat src = imread("dog.bmp");
	//�ж�ͼ���Ƿ���سɹ�
	if (src.data)
		cout << "ͼ����سɹ�!" << endl << endl;
	else
	{
		cout << "ͼ�����ʧ��!" << endl << endl;
		system("pause");
		return -1;
	}

	Mat src_gray, src_low;

	//�Ҷȴ�����˹�˲�
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_low = GaussianFilter(src_gray);

	//Sobel���Ӽ����ݶ�
	val = src_low.clone();
	Mat dir(src_low.rows, src_low.cols, CV_64F);
	SobelFilter(src_low, dir);
	//namedWindow("sobel", WINDOW_AUTOSIZE);
	//imshow("sobel", val);

	//�Ǽ���ֵ����
	NonMaxSuppression(dir);
	//namedWindow("NMS", WINDOW_AUTOSIZE);
	//imshow("NMS", val);

	//˫��ֵ���
	edge = val.clone();
	ThresholdDetection(0,0);

	namedWindow("edge_detection", WINDOW_AUTOSIZE);
	//�������˻�ȡ����ֵ����
	createTrackbar("Min Threshold:", "edge_detection", &low_threshold, high_threshold, ThresholdDetection);

	waitKey(0);
	return 0;
}