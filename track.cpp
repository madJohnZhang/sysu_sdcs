#include "track.h"
/*
*	需要测试：repeat函数用法， = 赋值时mat中data是否会在函数返回时有效。
*/
// 显示图像文件
using namespace std;
using namespace cv;
#pragma comment(linker, "/subsystem:\"console\" /entry:\"mainCRTStartup\"")  


float GlobalPara::threshold = UPDATE_JUDGER;

Point origin;
Rect choose;
bool select_flag = false, standby =true;

GlobalPara::GlobalPara()
{
	
}

int main()
{	
	/*
	读入图片之后把存储从BGR转到RGB
	*/
	//从文件中读取图像  
	
	//出现遮挡势必prob下降
	//cameraTest();
	VideoCapture cap(0);
	Mat image, tmp;
	cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, NORMAL_W);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, NORMAL_H);
	if (!cap.isOpened())
	{
		cerr << "Devices can't be opened" << endl;
		exit(-1);
	}
	
	namedWindow("Camera");
	setMouseCallback("Camera", mouseCallback, 0);

	while (standby)
	{
		cap >> image;
		tmp = image.clone();
		if (select_flag)
		{
			rectangle(tmp, choose, Scalar(0, 0, 255));
		}
		imshow("Camera", tmp);
		waitKey(1);
	}

	HogRawPixelNormExtractor extractor;
	LogisticRegression lr;
	SlidingWindowSampler sampler;
	ParticleFilterMotionModel motion;
	ClassificationScoreJudger judger;

	threadpool worker(5);

	Mat posSample, negSample;
	Mat posFeature, negFeature, feat;
	Mat probs(1, 1, CV_32FC1);
	probs.at<float>(0) = 1.0;
	int maxIdx[2] = { 0 };
	double maxProb= 1.0;
	
	char filename[100];
	Mat rects(1, 4, CV_32FC1);
	//Mat rects = (Mat_<float>(1, 4) << 152.149f, 66.72f, 17.833f, 84.208f);
	rects.at<float>(0) = choose.x;
	rects.at<float>(1) = choose.y;
	rects.at<float>(2) = choose.width;
	rects.at<float>(3) = choose.height;
	cout << rects << endl;
	cout << rects.type() << CV_32FC1 << endl;
	//cout << norm(rects);
	clock_t start, finish, start_par, finish_par;

	int divide_n = 2;
	future<void> *featF = new future<void>[divide_n];
	future<void> trainF;
	for (int i = 1; i <= 1500; i++)
	{
		//sprintf_s(filename, "./girl/%08d.jpg", i);
		//image = imread(filename);
		if (image.empty())
		{
			break;
		}
		start = clock();
		//resize(image, image, Size(NORMAL_W, NORMAL_H));
		if (i == 1)
		{
			posSample = sampler.quickPos(rects);
			negSample = sampler.quickNeg(rects);
			/*posSample = sampler.doPosSample(rects);
			
			negSample = sampler.doNegSample(rects);*/
			//cout << negSample;
			//cout <<"posSample"<< posSample << endl;
			posFeature = extractor.extract(image, posSample);
			negFeature = extractor.extract(image, negSample);
			lr.trainLR(posFeature, negFeature);
			feat = Mat(posFeature.rows, PARTICLE_N, posFeature.type());
		}
		else
		{
			motion.genParticle(rects, probs);
			
			start_par = clock();
			feat = extractor.extract(image, rects);
			/*for (int i = 0; i < divide_n; i++)
			{
				Mat tmpRect = rects.rowRange(rects.rows*i / divide_n, rects.rows*(i+1) / divide_n);
				featF[i] = worker.commit(multiThr_extractCopy, extractor, image, tmpRect, feat.colRange(i*50,(i+1)*50));
			}
			featF[0].get();
			featF[1].get();*/
			
			
			//cout << rects << endl;
			//feat = extractor.extract(image, rects);
			finish_par = clock();
			cout << "particle time:: " << (double)(finish_par - start_par) / CLOCKS_PER_SEC;
			probs = lr.LR(feat.colRange(0,rects.rows));
			//cout << probs << endl;

			minMaxIdx(probs, NULL, &maxProb, NULL, maxIdx);
			//cout << "i:= " << i << "maxProb:= " << maxProb << endl;
			if (judger.doUpdate(maxProb))
			{
				clock_t start_sam = clock();
				negSample = sampler.quickNeg(rects.row(maxIdx[1]));
				posSample = sampler.quickPos(rects.row(maxIdx[1]));
				
				//posSample = sampler.doPosSample(rects.row(maxIdx[1]));
				////cout << "posSample" <<posSample << endl;
				//negSample = sampler.doNegSample(rects.row(maxIdx[1]));
				/*future<Mat> pos = worker.commit(multiThr_extract, extractor,image,posSample);
				future<Mat> neg = worker.commit(multiThr_extract, extractor, image, negSample);
				posFeature = pos.get();
				negFeature = neg.get();*/
				//cout << "negSample" << negSample << endl;
				posFeature = extractor.extract(image, posSample);
				negFeature = extractor.extract(image, negSample);
				
				trainF= worker.commit(trainLR_Thr, &lr, posFeature, negFeature);
				cout << "sample time" << clock() - start_sam;
				/*clock_t start = clock(), finish;
				lr.trainLR(posFeature, negFeature);
				finish = clock();
				cout <<endl<< "train time"<< (double)(finish-start)/CLOCKS_PER_SEC << endl;*/
			}
		}
		 
		/*if (i >= 120)
		{
			for (int i = 0; i < probs.cols; i++)
			{
				if (i != maxIdx[1] && probs.at<float>(i) > 0.3f && probs.at<float>(i) < 0.5f)
				{
					Rect contrast(rects.at<float>(i, 0), rects.at<float>(i, 1), rects.at<float>(i, 2), rects.at<float>(i, 3));
					rectangle(image, contrast, Scalar(0, 0, 255));
				}
			}
		}*/
		Rect object(rects.at<float>(maxIdx[1], 0), rects.at<float>(maxIdx[1], 1), rects.at<float>(maxIdx[1], 2), rects.at<float>(maxIdx[1], 3));
		cout << "maxIdx" << maxIdx[1] << "maxprob" << maxProb <<"i = "<< i << endl;
		cout << rects.row(maxIdx[1]) << endl;
		cout << "one iteration cost:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<"seconds" << endl;
		rectangle(image, object, Scalar(255));
		imshow("a girl", image);
		waitKey(1);
		cap >> image;
	}
	return 0;
}

void mouseCallback(int event, int x, int y, int flags, void * userdata)
{
	if (!standby)
	{
		return;
	}
	if (select_flag)
	{
		choose.x = MIN(origin.x, x);
		choose.y = MIN(origin.y, y);
		choose.width = abs(origin.x - x);
		choose.height = abs(origin.y - y);
		choose &= Rect(0, 0, NORMAL_W, NORMAL_H);
	}
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		select_flag = true;
		origin = Point(x, y);
		cout << x << y << endl;
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		standby = false;
		select_flag = false;
	}
}
