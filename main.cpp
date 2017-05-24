#include <vector>
#include <stdio.h>

#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>

#define DELAY 1 
#define ESC_KEY 27       

using namespace cv;
using namespace std;
int main(int argc, char *argv[])
{
	/*
	Mat src = imread("cup.bmp", 1);
	if (src.data == 0) {
		printf("Incorrect image data or format\n");
		return 1;
	}
	return 0;*/

	{
		Mat image;
		image = imread("cup.bmp", 1);

		if (!image.data)
		{
			printf("No image data \n");
			return 0;
		}
		namedWindow("Display Image", WINDOW_AUTOSIZE);
		imshow("Display Image", image);
		namedWindow("Copy Image", WINDOW_AUTOSIZE);
		Mat copy_ccomp, copy_external, copy_list, copy_tree;
		image.copyTo(copy_ccomp);
		image.copyTo(copy_external);
		image.copyTo(copy_list);
		image.copyTo(copy_tree);
		imshow("Copy Image", copy_ccomp);
		Mat convertedImage;
		//cvtColor(image, convertedImage, CV_RGB2GRAY, 0);
		cvtColor(image, convertedImage, COLOR_RGB2GRAY, 0);
		namedWindow("Converted to gray scaled Image", WINDOW_AUTOSIZE);
		imshow("Converted to gray scaled Image", convertedImage);
		Mat resizedImage;
		Size size(100, 100);
		resize(image, resizedImage, size, 0, 0, INTER_LINEAR);
		namedWindow("Resized Image", WINDOW_AUTOSIZE);
		imshow("Resized Image", resizedImage);

		namedWindow("Image ROI [100,100]", WINDOW_AUTOSIZE);
		imshow("Image ROI [100,100]", image(Rect(100, 100, 100, 100)));

		Mat binImage;
		//threshold(convertedImage, binImage, 175, 255, CV_THRESH_BINARY);
		threshold(convertedImage, binImage, 200, 255, THRESH_BINARY);
		namedWindow("Bin image", WINDOW_AUTOSIZE);
		imshow("Bin image", binImage);

		vector<vector<Point>> contours_ccomp, contours_external, contours_list, contours_tree;
		//findContours(binImage, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		//findContours(binImage, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		findContours(binImage, contours_ccomp, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		findContours(binImage, contours_external, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		findContours(binImage, contours_list, RETR_LIST, CHAIN_APPROX_SIMPLE);
		findContours(binImage, contours_tree, RETR_TREE, CHAIN_APPROX_SIMPLE);

		Scalar color(0, 255, 0);
		namedWindow("Contour", WINDOW_AUTOSIZE);
		drawContours(copy_ccomp, contours_ccomp, -1, color, 2);
		drawContours(copy_external, contours_external, -1, color, 2);
		drawContours(copy_list, contours_list, -1, color, 2);
		drawContours(copy_tree, contours_tree, -1, color, 2);
		imshow("Contour", copy_ccomp);

		imwrite("cup_ccomp.jpg", copy_ccomp);
		imwrite("cup_external.jpg", copy_external);
		imwrite("cup_list.jpg", copy_list);
		imwrite("cup_tree.jpg", copy_tree);

		copy_ccomp.release();
		convertedImage.release();
		resizedImage.release();
		image.release();
		binImage.release();
		waitKey(0);
	}
	//===================

	CascadeClassifier cascade_front, cascade_profile, cascade_eye, cascade_smile;
	//cascade.load("E:\\programming\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml");
	cascade_front.load("E:\\programming\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	cascade_profile.load("E:\\programming\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_profileface.xml");
	cascade_eye.load("E:\\programming\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml");
	cascade_smile.load("E:\\programming\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_smile.xml");


	//VideoCapture vCap1, vCap;// = VideoCapture();
	VideoCapture vCap;
	vCap.open(0);
	//vCap.open("dance.mp4");
	VideoWriter vWriter;
	//double fourcc = vCap1.get(CAP_PROP_FOURCC);
	//vCap.release();
	//vCap.open(0);
	vWriter.open("webCamCapture.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
		Size(vCap.get(CAP_PROP_FRAME_WIDTH), vCap.get(CAP_PROP_FRAME_HEIGHT)), true);
	/*vWriter.open("webCamCapture.avi", vCap.get(CAP_PROP_FOURCC),vCap.get(CAP_PROP_FPS),
		Size(vCap.get(CAP_PROP_FRAME_WIDTH), vCap.get(CAP_PROP_FRAME_HEIGHT)), true);
	/*vWriter.open("webCamCapture.avi", fourcc, vCap.get(CAP_PROP_FPS),
		Size(vCap.get(CAP_PROP_FRAME_WIDTH), vCap.get(CAP_PROP_FRAME_HEIGHT)), true);
	if (!vWriter.isOpened())
	{
		printf("Could not open the output video for write");
		return 0;
	}*/

	const char* webCamName = "webcam";
	namedWindow(webCamName);
	char key = -1;
	Mat image, gray;
	vector<Rect> objects_front, objects_profile, objects_smile, objects_eye;
	vCap >> image;
	while (image.data && key != ESC_KEY) {
		cvtColor(image, gray, COLOR_BGR2GRAY);
		cascade_front.detectMultiScale(gray, objects_front);
		for (int i = 0; i < objects_front.size(); i++)
		{
			rectangle(image,
				Point(objects_front[i].x, objects_front[i].y),
				Point(objects_front[i].x + objects_front[i].width,
					objects_front[i].y + objects_front[i].height),
				Scalar(255, 0, 0), 2);
		}
		cascade_profile.detectMultiScale(gray, objects_profile);
		for (int i = 0; i < objects_profile.size(); i++)
		{
			rectangle(image,
				Point(objects_profile[i].x, objects_profile[i].y),
				Point(objects_profile[i].x + objects_profile[i].width,
					objects_profile[i].y + objects_profile[i].height),
				Scalar(0, 0, 255), 2);
		}/*
		cascade_smile.detectMultiScale(gray, objects_smile);
		for (int i = 0; i < objects_smile.size(); i++)
		{
			rectangle(image,
				Point(objects_smile[i].x, objects_smile[i].y),
				Point(objects_smile[i].x + objects_smile[i].width,
					objects_smile[i].y + objects_smile[i].height),
				Scalar(0, 255, 0), 2);
		}
		cascade_eye.detectMultiScale(gray, objects_eye);
		for (int i = 0; i < objects_eye.size(); i++)
		{
			rectangle(image,
				Point(objects_eye[i].x, objects_eye[i].y),
				Point(objects_eye[i].x + objects_eye[i].width,
					objects_eye[i].y + objects_eye[i].height),
				Scalar(50, 255, 200), 2);
		}*/
		imshow(webCamName, image);
		vWriter << image;
		key = waitKey(DELAY);
		vCap >> image;
		gray.release();
		objects_front.clear();
	}
	//waitKey(0);
	vCap.release();
	vWriter.release();

	return 0;

}