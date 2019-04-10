#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"
#include <queue>

using namespace std;
using namespace cv;

#define k_size 3//����˴�С
#define sigma 1//sigmaԽ��ƽ��Ч��Խ����
//#define high_threshold 100
//#define low_threshold 40

class node {
public:
	int x;
	int y;
	node(int x, int y) {
		this->x = x;
		this->y = y;
	}
	node()
	{
		this->x = 0;
		this->y = 0;
	}
};

int high_threshold = 100;
int low_threshold = 0;
const double PI = 4.0*atan(1.0);//��
double gaus[k_size][k_size];

Mat val, edge, dst, area;

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
			dir.at<double>(i, j) = atan2(gy.at<double>(i, j), gx.at<double>(i, j)) * 180 / PI;
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

void bfs(Mat &in, node current)
{
	queue<node> q;
	int x, y;
	int row = in.rows;
	int col = in.cols;
	q.push(current);
	while (!q.empty())
	{
		current = (q.front());
		x = current.x;
		y = current.y;
		area.at<uchar>(x, y) = 2;
		if (x - 1 >= 0)
		{
			if (area.at<uchar>(x - 1, y) == 1 && in.at<uchar>(x - 1, y) == 0)
			{
				q.push(node(x - 1, y));
				in.at<uchar>(x - 1, y) = 1;
			}
		}
		if (x + 1 < row)
		{
			if (area.at<uchar>(x + 1, y) == 1 && in.at<uchar>(x + 1, y) == 0)
			{
				q.push(node(x + 1, y));
				in.at<uchar>(x + 1, y) = 1;
			}
		}
		if (y - 1 >= 0)
		{
			if (area.at<uchar>(x, y - 1) == 1 && in.at<uchar>(x, y - 1) == 0)
			{
				q.push(node(x, y - 1));
				in.at<uchar>(x, y - 1) = 1;
			}
		}
		if (y + 1 < col)
		{
			if (area.at<uchar>(x, y + 1) == 1 && in.at<uchar>(x, y + 1) == 0)
			{
				q.push(node(x, y + 1));
				in.at<uchar>(x, y + 1) = 1;
			}
		}
		if (x + 1 < row &&y + 1 < col)
		{
			if (area.at<uchar>(x + 1, y + 1) == 1 && in.at<uchar>(x + 1, y + 1) == 0)
			{
				q.push(node(x + 1, y + 1));
				in.at<uchar>(x + 1, y + 1) = 1;
			}
		}
		if (x - 1 >= 0 && y + 1 < col)
		{
			if (area.at<uchar>(x - 1, y + 1) == 1 && in.at<uchar>(x - 1, y + 1) == 0)
			{
				q.push(node(x - 1, y + 1));
				in.at<uchar>(x - 1, y + 1) = 1;
			}
		}
		if (x - 1 >= 0 && y - 1 >= 0)
		{
			if (area.at<uchar>(x - 1, y - 1) == 1 && in.at<uchar>(x - 1, y - 1) == 0)
			{
				q.push(node(x - 1, y - 1));
				in.at<uchar>(x - 1, y - 1) = 1;
			}
		}
		if (x + 1 < row && y + 1 < col)
		{
			if (area.at<uchar>(x + 1, y + 1) == 1 && in.at<uchar>(x + 1, y + 1) == 0)
			{
				q.push(node(x + 1, y + 1));
				in.at<uchar>(x - 1, y + 1) = 1;
			}
		}
		q.pop();
	}
}

void ThresholdDetection(int, void*)
{
	high_threshold = low_threshold * 3;
	dst = val.clone();
	area = edge.clone();
	int row = val.rows;
	int col = val.cols;

	//ȷ��һ���Ǳߵ�
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (val.at<uchar>(i, j) >= high_threshold)
				area.at<uchar>(i, j) = 2;//һ���Ǳ�
			else if (val.at<uchar>(i, j) < low_threshold)
				area.at<uchar>(i, j) = 0;//һ�����Ǳ�
			else
				area.at<uchar>(i, j) = 1;//�����Ǳߵ�
		}
	}

	////ȷ�����б�
	//bool changed;
	//do
	//{
	//	changed = false;
	//	for (int i = 1; i < row - 1; i++)
	//	{
	//		for (int j = 1; j < col - 1; j++)
	//		{
	//			if (area.at<uchar>(i, j) == 2)
	//			{
	//				if (area.at<uchar>(i + 1, j) == 1)
	//				{
	//					area.at<uchar>(i + 1, j) = 2;
	//					changed = true;
	//				}
	//				if (area.at<uchar>(i - 1, j) == 1)
	//				{
	//					area.at<uchar>(i - 1, j) = 2;
	//					changed = true;
	//				}
	//				if (area.at<uchar>(i, j + 1) == 1)
	//				{
	//					area.at<uchar>(i, j + 1) = 2;
	//					changed = true;
	//				}
	//				if (area.at<uchar>(i, j - 1) == 1)
	//				{
	//					area.at<uchar>(i, j - 1) = 2;
	//					changed = true;
	//				}
	//			}
	//		}
	//	}
	//} while (changed);

	//�����ظ��������,�����
	Mat in(row, col, CV_8U, Scalar::all(0));
	node current;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (area.at<uchar>(i, j) == 2 && in.at<uchar>(i, j) == 0)
			{
				current = node(i, j);
				bfs(in, current);
			}
		}
	}

	//���ȷ����Ե
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (area.at<uchar>(i, j) == 0 || area.at<uchar>(i, j) == 1)
				dst.at<uchar>(i, j) = 0;
		}
	}
	imshow("edge_detection", dst);
}

int main()
{
	Mat src = imread("dog.bmp");
	//�ж�ͼ���Ƿ���سɹ�
	if (src.data)
		cout << "ͼ����سɹ�!" << endl;
	else
	{
		cout << "ͼ�����ʧ��!" << endl;
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
	ThresholdDetection(0, 0);

	namedWindow("edge_detection", WINDOW_AUTOSIZE);
	//�������˻�ȡ����ֵ����
	createTrackbar("Min Threshold:", "edge_detection", &low_threshold, 100, ThresholdDetection);

	waitKey(0);
	return 0;
}