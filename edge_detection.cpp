#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "math.h"

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int Ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";

//Canny��1:3��Ϊ��������
static void CannyThreshold(int, void*)
{
	//��3x3�ĺ˼�������
	blur(src_gray, detected_edges, Size(3, 3));
	//Canny�����
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*Ratio, kernel_size);
	//����ȫΪ0
	dst = Scalar::all(0);
	//��src��detected_edgesΪ���븴�Ƶ�dst
	src.copyTo(dst, detected_edges);
	//��ʾ�����Ľ��
	imshow(window_name, dst);
}

int main()
{
	src = imread("dog.bmp");
	//�ж�ͼ���Ƿ���سɹ�
	if (src.data)
		cout << "ͼ����سɹ�!" << endl << endl;
	else
	{
		cout << "ͼ�����ʧ��!" << endl << endl;
		system("pause");
		return -1;
	}

	//����һ����ԭͼ��С����̬һ�µľ���
	dst.create(src.size(), src.type());
	//ת�ɻҶ�
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	namedWindow(window_name, WINDOW_AUTOSIZE);
	//��������������Ϊ����ֵ������
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	//�������ʾ���
	CannyThreshold(0,0);

	waitKey(0);
	return 0;
}