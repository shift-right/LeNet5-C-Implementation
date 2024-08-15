#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include<inttypes.h>
using namespace std;
using namespace cv;

uint32_t swap_endian(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void readAndSave(const string& mnist_img_path, const string& mnist_label_path)
{
	//以二進位制格式讀取mnist資料庫中的影象檔案和標籤檔案  
	ifstream mnist_image(mnist_img_path, ios::in | ios::binary);
	ifstream mnist_label(mnist_label_path, ios::in | ios::binary);
	if (mnist_image.is_open() == false)
	{
		cout << "open mnist image file error!" << endl;
		return;
	}
	if (mnist_label.is_open() == false)
	{
		cout << "open mnist label file error!" << endl;
		return;
	}

	uint32_t magic;//檔案中的魔術數(magic number)  
	uint32_t num_items;//mnist影象集檔案中的影象數目  
	uint32_t num_label;//mnist標籤集檔案中的標籤數目  
	uint32_t rows;//影象的行數  
	uint32_t cols;//影象的列數  
	//讀魔術數  
	mnist_image.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != 2051)
	{
		cout << "this is not the mnist image file" << endl;
		return;
	}
	mnist_label.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	if (magic != 2049)
	{
		cout << "this is not the mnist label file" << endl;
		return;
	}
	//讀影象/標籤數  
	mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
	num_label = swap_endian(num_label);
	//判斷兩種標籤數是否相等  
	if (num_items != num_label)
	{
		cout << "the image file and label file are not a pair" << endl;
	}
	//讀影象行數、列數  
	mnist_image.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	mnist_image.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);
	//讀取影象  
	for (int i = 0; i != num_items; i++)
	{
		char* pixels = new char[rows * cols];
		mnist_image.read(pixels, rows * cols);
		char label;
		mnist_label.read(&label, 1);
		Mat image(rows, cols, CV_8UC1);
		for (int m = 0; m != rows; m++)
		{
			uchar* ptr = image.ptr<uchar>(m);
			for (int n = 0; n != cols; n++)
			{
				if (pixels[m * cols + n] == 0)
					ptr[n] = 0;
				else
					ptr[n] = 255;
			}
		}
		// string saveFile = "C:\\Users\\Yu\\Desktop\\mnist\\mnist_test\\" +  to_string(i) + "_" + to_string((unsigned int)label) + ".jpg";
		string saveFile = "C:\\Users\\Yu\\Desktop\\mnist\\mnist_train\\" +  to_string(i) + "_" + to_string((unsigned int)label) + ".jpg";
		imwrite(saveFile, image);
	}
}

int main()
{
	// readAndSave("C:\\Users\\Yu\\Desktop\\MNIST\\t10k-images.idx3-ubyte", "C:\\Users\\Yu\\Desktop\\MNIST\\t10k-labels.idx1-ubyte");
	readAndSave("C:\\Users\\Yu\\Desktop\\MNIST\\train-images.idx3-ubyte", "C:\\Users\\Yu\\Desktop\\MNIST\\train-labels.idx1-ubyte");
	return 0;
}