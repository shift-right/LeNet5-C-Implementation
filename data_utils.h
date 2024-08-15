#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "matrix.h"

namespace fs = std::filesystem;
using namespace matrix;

void load_path_from_txt(const std::string &home_path, const std::string &txt_path, std::vector<std::string> &return_file_paths)
{
    std::ifstream input_file(txt_path);
    if (!input_file.is_open()) {
        std::cout << "Failed to open file.\n";
        return;
    }
    std::vector<std::string> lines;
	std::string line;

    while (getline(input_file, line)){
        lines.push_back(line);
    }

    std::string new_line;

    for(int i=0; i<lines.size();i++){
		new_line = home_path;
        new_line += lines[i];
        return_file_paths.push_back(new_line);
	}
}

void load_path_from_folder(const std::string &home_path, const std::string &folder_path, std::vector<std::string> &return_file_paths)
{
    std::string file;
    int file_num = 0;

    for (const auto & entry_file : fs::directory_iterator(home_path+folder_path)){
        file = entry_file.path().string();
        return_file_paths.push_back(file);
        file_num++;
    }
    std::cout<<file_num<<" files in the folder"<<std::endl;
}

Matrix3d load_img(const std::string &img_path, const int & img_resize)
{
    const char *file_path = img_path.c_str();
    int chs, rows, cols;
    unsigned char *img = stbi_load(file_path, &cols, &rows, &chs, 0);
	if (img == NULL) {
		std::cout << "ERROE_FILE_NOT_LOAD" << std::endl;
	}
    size_t img_size = cols * rows * chs;

    
    Matrix3d return_img = Matrix3d(1,rows,cols);

    if(chs == 1){

        return_img = Matrix3d(1,rows,cols);
		std::vector<double> pixeldata;
		pixeldata.reserve(cols * rows);
		unsigned i = 0;
		for(unsigned char *p = img; p != img + img_size; p += chs) {
            pixeldata[i] = *p;
            i++;
        }
		for(unsigned row=0;row<rows;row++)
			for(unsigned col=0;col<cols;col++){
				return_img.data_[0][row][col] = pixeldata[(row*cols) + col];
			}
	}
    else{
        std::vector<double> pixeldata[3];
        for(unsigned i =0;i<chs;i++){
            pixeldata[i].reserve(cols * rows);
        }
        unsigned k = 0;
        for(unsigned char *p = img; p != img + img_size; p += chs) {
            for(unsigned j=0;j<chs;j++) 
                pixeldata[j][k] = *(p+j);
            k++;
        }
        for(unsigned row=0;row<rows;row++)
			for(unsigned col=0;col<cols;col++)
                return_img.data_[0][row][col] = pixeldata[0][row*cols + col]*0.299 + pixeldata[0][row*cols + col]*0.587 + pixeldata[0][row*cols + col]*0.114;
    }

    


    if(img_resize>0){
        
        unsigned r_rows = img_resize;
        unsigned r_cols = img_resize;

        if (img_resize > rows || img_resize > cols){
            Matrix3d resize_img = Matrix3d(chs,r_rows,r_cols,0);
            unsigned start_row = (r_rows-rows)/2;
            unsigned start_col = (r_cols-cols)/2;
            unsigned end_row = img_resize-start_row;
            unsigned end_col = img_resize-start_col;
            for(unsigned row=0;row<r_rows;row++){
                if(row>=start_row && row<end_row){
                    for(unsigned col=0;col<r_cols;col++){
                        if(col>=start_col && col<end_col){
                            resize_img.data_[0][row][col] = return_img.data_[0][row-start_row][col-start_col];
                        }
                    }
                }
            }
            stbi_image_free(img);
            return resize_img;  
        }

        
        if(img_resize<rows && img_resize<cols){
            Matrix3d resize_img = Matrix3d(chs,r_rows,r_cols);
            unsigned mid_row = rows/2;
            unsigned start_row = mid_row - img_resize/2;
            unsigned end_row = start_row + img_resize -1;
            unsigned mid_col = cols/2;
            unsigned start_col = mid_col - img_resize/2;
            unsigned end_col = start_col + img_resize -1;

            for(unsigned row=0;row<rows;row++){
                if(row>=start_row && row<= end_row){
                    for(unsigned col=0;col<cols;col++){
                        if(col>=start_col && col<= end_col){
                            resize_img.data_[0][row-start_row][col-start_col] = return_img.data_[0][row][col];
                        }
                    }
                }
            }
            stbi_image_free(img);
            return resize_img;   
        } 
        else if(img_resize<rows && img_resize>= cols){
            Matrix3d resize_img = Matrix3d(1,r_rows,cols);
            unsigned mid_row = rows/2;
            unsigned start_row = mid_row - img_resize/2;
            unsigned end_row = start_row + img_resize -1;
            for(unsigned row=0;row<rows;row++)
                if(row>=start_row && row<= end_row){
                    for(unsigned col=0;col<cols;col++){
                        resize_img.data_[0][row-start_row][col] = return_img.data_[0][row][col];
                    }
                }
            stbi_image_free(img);
            return resize_img; 
        } 
        else if (img_resize<cols && img_resize>= rows){
            Matrix3d resize_img = Matrix3d(1,rows,r_cols);
            unsigned mid_col = cols/2;
            unsigned start_col = mid_col - img_resize/2;
            unsigned end_col = start_col + img_resize -1;
            for(unsigned row=0;row<rows;row++)
                for(unsigned col=0;col<cols;col++){
                    if(col>=start_col && col<= end_col){
                        resize_img.data_[0][row][col-start_col] = return_img.data_[0][row][col];
                    }
                }
            stbi_image_free(img);
            return resize_img; 
        }

        
        else{
            Matrix3d resize_img = Matrix3d(1,r_rows,r_cols,0);

            for(unsigned row=0;row<rows;row++){
                for(unsigned col=0;col<cols;col++){
                    resize_img.data_[0][row][col] = return_img.data_[0][row][col];
                }
            }
            stbi_image_free(img);
            return resize_img;   
        } 

    }
    stbi_image_free(img);
    return return_img; 
}

void dataset(const std::string &home_path, std::string &object_path, std::vector<std::string> &file_paths, const std::string load_mode)
{
    Matrix3d img;
    if(load_mode == "txt"){
        load_path_from_txt(home_path, object_path, file_paths);
    }
    else if(load_mode == "folder"){
        load_path_from_folder(home_path, object_path, file_paths);
    }


}

void dataloader(std::vector<std::string> & file_paths, std::vector<std::vector<std::string>> & return_file_paths_batch, std::vector<std::vector<int>> & return_labels_batch, const int & batch_size, int &num_labels, const bool if_shuffle)//shuffle 寫在這裡 固定size
{
    std::string file_path;
    std::vector<std::string> new_file_paths;
    std::vector<std::string> new_labels;

    std::vector<std::string> file_paths_batch(batch_size);
    std::vector<int> label_batch(batch_size);

    auto rng = std::default_random_engine{};
    if(if_shuffle){
        // std::cout<<"data shuffle"<<std::endl;
		shuffle(begin(file_paths), end(file_paths), rng);
    }
    else{
        std::reverse(file_paths.begin(), file_paths.end());
    }


    std::string line;
    std::string lab;
    std::string new_line;
    std::vector<std::string> new_lines;
    for(int i=0; i<file_paths.size();i++){
        line = file_paths[i];
        std::stringstream ss(line);
        while (getline(ss, new_line)) {
            new_lines.push_back(new_line);
        }
    }

    while(new_lines.size()!=0){
        line = new_lines.back();
        line.pop_back();line.pop_back();line.pop_back();line.pop_back();
        lab = line.back();
		new_labels.push_back(lab);
		new_file_paths.push_back(new_lines.back());
		new_lines.pop_back();
		
	}

    int count = 0;
    int total_files = new_labels.size();
    
    std::vector<int> labels_category;


    while(total_files>=batch_size){
        if(file_paths.size()/batch_size==0 && file_paths.size()<batch_size){
            break;
        }

        for(int i=0;i<batch_size;i++){
            label_batch[i]=stoi(new_labels[count]);
            file_paths_batch[i] = new_file_paths[count];
            count++;
            total_files--;

            if(std::count(labels_category.begin(), labels_category.end(), label_batch[i])){}
            else{
                labels_category.push_back(label_batch[i]);
            }
        }
        return_file_paths_batch.push_back(file_paths_batch);
        return_labels_batch.push_back(label_batch);
    }
    
    num_labels = labels_category.size();


    // std::cout<<"labels:"<<labels_category.size()<<std::endl;
    // std::cout<<"count:"<<count<<std::endl;
}

Matrix3d label_to_mat(const int & label, const int & num_labels)
{
    Matrix3d return_mat = Matrix3d(1,1,num_labels,0);
    return_mat.data_[0][0][label] = 1;

    return return_mat;
}