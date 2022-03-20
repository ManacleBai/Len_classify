#pragma once
#include "Variable.h"
#include <core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace std;
using namespace cv;

//
//extern stLenAlgorithm;
//
//stLenAlgorithm LenAlgorithm;

void ProcessReflective(Mat img, stLenAlgorithm* LenAlgorithm);
void MEMReflective(Mat img, stLenAlgorithm *LenAlgorithm);
void OneD_Hough(stLenAlgorithm* LAM, uint16_t r, uchar theta, uchar vote,uint16_t minLength, uchar jumpNumber);
//void PixelCompensate(Mat img, stLenAlgorithm *LenAlgorithm, size_t direction);
void DeExtraPixel(Mat img, stLenAlgorithm* LenAlgorithm);
void SignContours(vector<vector<Point> > contours, stLenAlgorithm* LenAlgorithm);
void SmoothGraph(stLenAlgorithm *LenAlgorithm, uchar KpointSize);
void ClassifyGraphy(stLenAlgorithm* LAM);
// romove
void Classify(stLenAlgorithm* LenAlgorithm);
void AddFeature(stLenAlgorithm* LAM, vector<Point> Graphy);
//
void LS_Fit(stLenAlgorithm* LAM, vector<Point> axis, GraphPixel Type);
void FindBoundary(stLenAlgorithm* LAM);
void Draw(Mat img, stLenAlgorithm* LAM);

//Matrix
void transposeMatrix(vector<vector<float> >& Output, vector<vector<float> > Input); //A^T
void multipleMatrix(vector<vector<float> >& Output, vector<vector<float> > IutputA, vector<vector<float> > IutputB); //A X B
void inverseMatrix(vector<vector<float > > &Output, vector<vector<float> > Input);
float det(vector<vector<float> > Input);
//void 