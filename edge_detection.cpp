#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

//����Ե��ⲻ��Ҫͼ��̫��ģ��
#define k_size 3//����˴�С
#define sigma 1//sigmaԽ��ƽ��Ч��Խ����
//˫��ֵ���
#define high_threshold 100
#define low_threshold 2

double gaus[k_size][k_size];

bool Max(uchar x, uchar a, uchar b)
{
	//�����Ƿ���Ե�
	if ((double)x >= (double)a && (double)x >= (double)b)
		return true;
	return false;
}

void GetgaussianKernel()
{
	const double PI = 4.0*atan(1.0);//��
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
	//namedWindow("gaussian_filter", WINDOW_AUTOk_size);
	//imshow("gaussian_filter", dst);
	return dst;
}

//�����ݶ�ֵ���ݶȷ���
void SobelFilter(Mat src, Mat val, Mat dir)
{
	Mat sobel_x = (Mat_<double>(3, 3) <<
		1, 0, -1,
		2, 0, -2,
		1, 0, -1);
	Mat sobel_y = (Mat_<double>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		1, -2, -1);
	Mat gradient_x, gradient_y;
	filter2D(src, gradient_x, src.depth(), sobel_x);
	filter2D(src, gradient_y, src.depth(), sobel_y);

	//�����þ���ֵ��ӽ��ƣ�y�᷽����ݶȺܶ�255
	//Mat temp = gradient_x.clone();
	//Mat temp(gradient_x.size(), gradient_x.type());
	//magnitude(gradient_x, gradient_y, temp);
	Mat gx, gy;
	Sobel(src, gx, src.depth(), 1, 0, 3);
	Sobel(src, gy, src.depth(), 0, 1, 3);

	Mat mag(gx.size(), gx.type());
	magnitude(gx, gy, mag);
	//-------------debug-------------------
	//int row1 = gradient_x.rows;
	//int col1 = gradient_x.cols;
	//for (int i = 0; i < row1; i++)
	//{
	//	for (int j = 0; j < col1; j++)
	//			cout << (int)gradient_y.at<uchar>(i, j) << " ";
	//	cout << endl;
	//}
	//--------------------------------------
	int row = src.rows;
	int col = src.cols;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			dir.at<uchar>(i, j) = atan2(gradient_y.at<uchar>(i, j), gradient_x.at<uchar>(i, j));
	//dir.at<uchar>(i, j) = atan2(gradient_x.at<uchar>(i, j), gradient_y.at<uchar>(i, j));
}

//val������Ҫ���ֵ��Ϊ0
void NonMaxSuppression(Mat val, Mat dir)
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

	//ȷ��һ���Ǳߵ�
	for (int i = 1; i < row - 1; i++)
	{
		for (int j = 1; j < col - 1; j++)
		{
			if (val.at<uchar>(i, j) >= high_threshold)
				edge.at<uchar>(i, j) = 2;//һ���Ǳ�
			else if (val.at<uchar>(i, j) >= low_threshold)
				edge.at<uchar>(i, j) = 1;//�����Ǳ�
			else
				edge.at<uchar>(i, j) = 0;//һ�����Ǳ�
		}
	}
	//���α�����ȷ�����б�
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
	//�ж�ͼ���Ƿ���سɹ�
	if (src.data)
		cout << "ͼ����سɹ�!" << endl << endl;
	else
	{
		cout << "ͼ�����ʧ��!" << endl << endl;
		system("pause");
		return -1;
	}

	Mat src_gray, src_low, gradient_val, gradient_dir, edge;

	//�Ҷȴ�����˹�˲�
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_low = GaussianFilter(src_gray);

	//Sobel���Ӽ����ݶ�
	gradient_val = src_gray.clone();
	gradient_dir = src_gray.clone();
	SobelFilter(src_gray, gradient_val, gradient_dir);

	////�Ǽ���ֵ����
	//NonMaxSuppression(gradient_val, gradient_dir);

	////˫��ֵ���
	//edge = gradient_val.clone();
	//ThresholdDetection(gradient_val, edge);

	////���ȷ����Ե
	//int row = gradient_val.rows;
	//int col = gradient_val.cols;
	//for (int i = 0; i < row; i++)
	//{
	//	for (int j = 0; j < col; j++)
	//	{
	//		if (edge.at<uchar>(i, j) == 0)
	//			gradient_val.at<uchar>(i, j) = 0;
	//	}
	//}
	//namedWindow("edge_detection", WINDOW_AUTOk_size);
	//imshow("edge_detection", gradient_val);

	waitKey(0);
	return 0;
}