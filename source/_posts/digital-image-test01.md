---
title: 数字图像处理实验-图像灰度变化
categories: 算法实践
date: 2022-09-20 18:32:15
tags: [数字图像, OpenCV]
cover:
top_img:
---
# 图像灰度变换

### 1、利用Opencv读取图像

完成程度：使用opencv中的imread（）函数完成了对图片文件“lena.bmp”的读取，并将读取的内容存储至cv中的Mat矩阵中，最后使用imshow（）函数将该图片在窗口中显示出来。

```C++
//  1、利用OpenCV读取图像
Mat read_image(String path) {
	Mat image = imread(path, IMREAD_GRAYSCALE);
	Mat tmp = image.clone();
	if (image.empty()) {
		cout << "Can't find the path.Please input the true path." << endl;
		return image;
	}
	imshow("lena.bmp", tmp);
	return tmp;
}

```



### 2、灰度图像二值化处理

完成程度：遍历内容1中读取的矩阵，设置阈值为128，将矩阵中大于128的像素设置为255，将矩阵中小于等于128的像素点设置为0，使用imshow（）函数输出该图片。

```C++
// 2、灰度图像二值化处理
void binary_image(Mat tmp) {
	// 设置阈值为128

	Mat image = Mat(tmp);
	for (int i = 0; i < image.rows; i++) {
		uchar* p = image.ptr(i);
		for (int j = 0; j < image.cols; j++) {
			if (p[j] > 128) {
				p[j] = 255;
			}
			else {
				p[j] = 0;
			}
		}
	}
	imshow("binary_image", image);
}

```

![image-20240302183456769](digital-image-test01/image-20240302183456769.png)



### 3、灰度图像的对数变换

完成程度：利用对数变化公式s=c*log(1+r)，对内容1的矩阵进行对数变化。调整不同的c的值[0.5,1,2,4,8]，对比不同的c值对图像变化的影响。

```c++
// 3、灰度图像的对数变换
void log_reserve(Mat image) {
	// 对数变化的函数s = c*log(1+r)

	Mat scrimage(image);
	Mat new_image(scrimage.size(), scrimage.type());
	// 对原图像进行加1操作
	add(scrimage, Scalar(1.0), scrimage);
	// 数据类型转化
	scrimage.convertTo(scrimage, CV_64F);
	float c[] = { 0.5, 1, 2, 4, 8 };

	for (int s = 0; s < 5; s++) {
		log(scrimage, new_image);
		new_image = c[s] * new_image;

		//对图像进行归一化处理调整阈值至0-255
		normalize(new_image, new_image, 0, 255, NORM_MINMAX);
		convertScaleAbs(new_image, new_image);
		String str = "c=" + to_string(c[s]);
		imshow(str, new_image);
	}
}
```



![image-20240302183504049](digital-image-test01/image-20240302183504049.png)



![image-20240302183514481](digital-image-test01/image-20240302183514481.png)



### 4、灰度图像的伽马变换

完成程度：利用伽马变化公式s=c*（r**γ）完成了对图像的转化，其中令c默认为1，调整γ值分别为[0.1,0.4,1,2.5,10]，对比不同伽马值对图像变换的影响。

```c++
// 4、灰度图像的伽马变换
void gama_reserve(Mat image) {
	// gama变化的函数s = c*(r**gama)，取c=1
	image.convertTo(image, CV_64F, 1.0 / 255, 0);
	Mat new_image(image.size(),image.type());

	float gama[] = { 0.10, 0.40, 1, 2.5, 10 };

	for (int i = 0; i < 5; i++) {
		pow(image, gama[i], new_image);
		new_image.convertTo(new_image, CV_8U, 255, 0);
		String str = "gama=" + to_string(gama[i]);
		imshow(str, new_image);
	}
}
```



![image-20240302183527447](digital-image-test01/image-20240302183527447.png)

![image-20240302183542527](digital-image-test01/image-20240302183542527.png)



### 5、彩色图像的补色变换

完成程度：通过imread（）函数读取“lenaRGB.bmp”彩色图像，并输出到窗口。使用容器存储彩色图像的三个通道的值，对不同的通道进行求补操作，具体操作为，用255减去原始值。将求补后的三个通道进行合并处理，并显示合并后的图像。

```c++
// 5、彩色图像的补色变换
void color_reverse(){
	// 读入彩色图像
	String path = "photo/lenaRGB.bmp";
	Mat image = imread(path);
	if (image.empty()) {
		cout << "Can't find the path.Please input the true path." << endl;
		return;
	}
	imshow("scr_image", image);
	// 分离三通道，并保存
	vector<Mat> channels;
	split(image, channels);
	// 求补操作
	for (int i = 0; i < 3; i++) {
		channels[i] = 255 - channels[i];
	}
	// 合并通道
	Mat new_image(image.size(), image.type());
	merge(channels, new_image);
	imshow("dir_image", new_image);
}
```



![image-20240302183548286](digital-image-test01/image-20240302183548286.png)

