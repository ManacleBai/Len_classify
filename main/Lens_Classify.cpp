// Lens_Classify.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")
#include <iostream>
#include <math.h>
#include<string.h>
#include "Header.h"
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int main() {
	stLenAlgorithm LenAlgorithm;
	Mat img = imread("C:\\Users\\Alfred\\Desktop\\LensPicture\\1112.jpg", -1);
	//Mat img = imread("C:\\Users\\Alfred\\Desktop\\LensPicture\\456.jpg", -1);
	Mat grey, dst, dst1, Himg, smallimg;
	if (img.empty()) return -1;
	resize(img, smallimg, Size(up_width, up_height), INTER_AREA);
	SHOW("Orig", smallimg);
	//==========de_Reflective============
	cvtColor(smallimg, grey, COLOR_BGR2GRAY);
	LenAlgorithm.IMGReflective.CompenSize = 10;
	MEMReflective(grey, &LenAlgorithm);
	cvtColor(smallimg, Himg, COLOR_BGR2HSV);
	//cvtColor(Himg, grey, COLOR_BGR2GRAY);
	//===================================
	// 
	//==========blurry===================
	blur(grey, dst, Size(1, 1), Point(-1, -1));
	GaussianBlur(dst, dst1, Size(11, 11), 11, 11);
	SHOW("GaussianBlur", dst1);
	//===================================
	// 
	//===========Sobel===================
	Mat gradX, gradY, dst12, dilation;
	Sobel(dst1, gradX, CV_16S, 1, 0, 3);
	Sobel(dst1, gradY, CV_16S, 0, 1, 3);
	convertScaleAbs(gradX, gradX);
	convertScaleAbs(gradY, gradY);
	addWeighted(gradX, 0.5, gradY, 0.5, 0, dst12);
	//threshold(dst12, dst12, 70, 255, THRESH_OTSU);
	threshold(dst12, dst12, 20, 255, THRESH_BINARY); //8
	ProcessReflective(dst12, &LenAlgorithm);
	morphologyEx(dst12, dilation, MORPH_CLOSE, noArray(), Point(-1, -1), 1); //10
	SHOW("dilation", dilation);
	//====================================
	// 
	//=========Sign=======================
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dilation, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	SignContours(contours, &LenAlgorithm);
	SmoothGraph(&LenAlgorithm, Kpoint);
	OneD_Hough(&LenAlgorithm, 1600, 90, 18, 80, 20);
	ClassifyGraphy(&LenAlgorithm);
	FindBoundary(&LenAlgorithm);
	//Classify(&LenAlgorithm);
	//=====================================
	//
	//========= result & picture ==========
	Draw(smallimg, &LenAlgorithm);
	//imwrite("C:\\Users\\Alfred\\Desktop\\LensPicture\\1111.jpg", smallimg);
	SHOW("smallimg", smallimg);
	waitKey(0);
	//=====================================
}

