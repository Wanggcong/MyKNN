#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

int main()
{
	//*************************************************************************
	//载入数据
	Mat train_data(100, 2, CV_32FC1);
	Mat labels(100, 1, CV_32FC1);
	ifstream fin("E:\\test\\testSet.txt", ios::in);
	if (!fin)
	{
		cout << "can not open the file!" << endl;
		return -1;
	}
	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (j == 2)
			{
				fin >> labels.at<float>(i, 0);
				//cout << labels.at<float>(i, 0) << endl;
			}
			else
			{
				fin >> train_data.at<float>(i, j);
				//cout << train_data.at<float>(i, j)<< setw(15);
			}
		}
	}
	//*************************************************************************
	//训练数据
	//Mat sample
	//CvKNearest knn(train_data, labels,0, false, 10);
	CvKNearest knn;
	const Mat& sampleIdx = Mat();
	knn.train(train_data, labels, sampleIdx, false, 32, false);
	//*************************************************************************
	//测试数据

	//*************************************************************************
	//显示结果
	Mat I = Mat::zeros(512, 512, CV_8UC3);
	//Mat I(515, 512, CV_8UC3);
	Vec3b green(0, 100, 0), blue(100, 0, 0);
	float xmin = train_data.at<float>(0, 0);
	float xmax = train_data.at<float>(0, 0);

	float ymin = train_data.at<float>(0, 1);
	float ymax = train_data.at<float>(0, 1);

	for (int i = 0; i < 100; i++)                                 //要找出最大最小值，比例地显示在画面上
	{
		if (train_data.at<float>(i, 0)>xmax)
		{
			xmax = train_data.at<float>(i, 0);
		}
		if (train_data.at<float>(i, 0)<xmin)
		{
			xmin = train_data.at<float>(i, 0);
		}


		if (train_data.at<float>(i, 1)>ymax)
		{
			ymax = train_data.at<float>(i, 1);
		}
		if (train_data.at<float>(i, 1)<ymin)
		{
			ymin = train_data.at<float>(i, 1);
		}
	}
#if 1
	for (int i = 0; i < I.rows; ++i)
	for (int j = 0; j < I.cols; ++j)
	{
		float k1 = i*1.0 * (xmax - xmin) / 512 + xmin;                             //转成原值，预测
		float k2 = j*1.0 * (ymax - ymin) / 512 + ymin;
		Mat sampleMat = (Mat_<float>(1, 2) << k1, k2);
		Mat result,neighborResponse,dist;
		float response = knn.find_nearest(sampleMat, 10, result, neighborResponse, dist);                                                      //显示决策区

		if (response == 1)
			I.at<Vec3b>(j, i) = green;
		else
			I.at<Vec3b>(j, i) = blue;                                                                 //
	}
	int thick = -1;
	int lineType = 8;
	float px, py;

	for (int i = 0; i < 100; ++i)
	{
		px = train_data.at<float>(i, 0);
		py = train_data.at<float>(i, 1);

		//px = px*1.0 * (xmax - xmin)/512 + xmin;
		//py = py*1.0 * (ymax - ymin)/512 + ymin;
		px = (px - xmin) / (xmax - xmin) * 512;
		py = (py - ymin) / (ymax - ymin) * 512;

		cout << "py:" << py << setw(10) << "px:" << px << endl;

		if (labels.at<float>(i, 0) == 1)
			circle(I, Point((int)px, (int)py), 3, Scalar(0, 255, 0), thick, lineType);
		else
			circle(I, Point((int)px, (int)py), 3, Scalar(255, 0, 0), thick, lineType);
	}
#endif

	imwrite("result.png", I);	                   // save the Image
	imshow("KNN Training Data", I); // show it to the user
	waitKey(0);
	return 0;
}