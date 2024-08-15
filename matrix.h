#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <functional>
#include <random>
#include <cmath>

namespace matrix
{
    class Matrix3d
    {
        public:
            Matrix3d(unsigned chs=0, unsigned rows=0, unsigned cols=0, double init_val=0);
            unsigned chs_ , rows_, cols_;
            unsigned label;
            std::vector<unsigned> shape{0, 0, 0};
            std::vector<std::vector<std::vector<double>>> data_;

            double at(unsigned ch=0, unsigned row=0, unsigned col=0);

            Matrix3d operator+ (const Matrix3d & target);
            Matrix3d operator- (const Matrix3d & target);
            Matrix3d operator* (const Matrix3d & target);
            Matrix3d dot(const Matrix3d & target);
            Matrix3d operator/ (const Matrix3d & target);
            Matrix3d sqrt();
            Matrix3d square();
            
            Matrix3d add_scalar(double scaler = 0);
            Matrix3d multiply_scalar(double scaler = 1);
            Matrix3d divide_scalar(double scaler = 1);
            Matrix3d reshape(unsigned r_chs, unsigned r_rows, unsigned r_cols);
            Matrix3d copy();
            Matrix3d apply_function(std::function<double(double)> func);
            Matrix3d transpose();
            Matrix3d flatten();
            Matrix3d normalize(const double ori_min, const double ori_max, const double new_min, const double new_max);



            double sum();

            void random_initialize();
            std::vector<int> argmax(int axis);
            std::vector<int> argmin(int axis);      

               

            // random initialize
            // sum;
            // abs;
            // maximum index;
    };

    Matrix3d::Matrix3d(unsigned chs, unsigned rows, unsigned cols, double init_val)
    : chs_ (chs), rows_ (rows), cols_ (cols)
    {
        // if (chs == 0 || rows == 0 || cols == 0)
        // std::cout<<"Matrix constructor has 0 size"<<std::endl; //throw statement
        data_ = std::vector<std::vector<std::vector<double>>> (chs_,std::vector<std::vector<double>>(rows_,std::vector<double>(cols_,init_val)));
        shape[0] = chs;
        shape[1] = rows;
        shape[2] = cols;
    }

    double Matrix3d::at(unsigned ch, unsigned row, unsigned col)
    {
        return data_[ch][row][col];
    }

    Matrix3d Matrix3d::operator+(const Matrix3d & target)
    {
        assert(shape[0]==target.shape[0] && shape[1]==target.shape[1] && shape[2]==target.shape[2]);
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] + target.data_[ch][row][col];
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::operator-(const Matrix3d & target)
    {
        assert(shape[0]==target.shape[0] && shape[1]==target.shape[1] && shape[2]==target.shape[2]);
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] - target.data_[ch][row][col];
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::operator*(const Matrix3d & target)
    {
        assert(shape[0]==target.shape[0] && shape[2]==target.shape[1]);
        Matrix3d result = Matrix3d(shape[0], shape[1], target.shape[2],0);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned t_col=0;t_col<target.cols_;t_col++)
                {
                    for (unsigned col=0;col<cols_;col++)
                        result.data_[ch][row][t_col] += data_[ch][row][col] * target.data_[ch][col][t_col];
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::square()
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] * data_[ch][row][col];
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::sqrt()
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = std::sqrt(data_[ch][row][col]);
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::dot(const Matrix3d & target)
    {
        assert(shape[0]==target.shape[0] && shape[1]==target.shape[1] && shape[2]==target.shape[2]);
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] * target.data_[ch][row][col];
                }
            }
        } 
        return result;
    } 

    Matrix3d Matrix3d::normalize(const double ori_min, const double ori_max, const double new_min, const double new_max)
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        double scale = (new_max-new_min)/(ori_max-ori_min);

        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = (data_[ch][row][col]-ori_min)*scale+new_min;
                }
            }
        } 
        return result;
    }


    Matrix3d Matrix3d::operator/(const Matrix3d & target)
    {
        assert(shape[0]==target.shape[0] && shape[1]==target.shape[1] && shape[2]==target.shape[2]);
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    if(target.data_[ch][row][col]!=0){
                        result.data_[ch][row][col] = data_[ch][row][col] / target.data_[ch][row][col];
                    }
                    else{
                        result.data_[ch][row][col] = data_[ch][row][col] / 1e-8;
                    }
                    
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::add_scalar(double scaler)
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] + scaler;
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::multiply_scalar(double scaler)
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col] * scaler;
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::divide_scalar(double scaler)
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    if (data_[ch][row][col]!=0){
                        result.data_[ch][row][col] = data_[ch][row][col] / scaler;
                    }
                    else{
                        result.data_[ch][row][col] = 0.000001;
                    }
                }
            }
        } 
        return result;
    }


    Matrix3d Matrix3d::reshape(unsigned r_chs, unsigned r_rows, unsigned r_cols)
    {
        assert(chs_*rows_*cols_== r_chs*r_rows*r_cols);
        std::vector<double> temp(chs_*rows_*cols_); 
        Matrix3d result = Matrix3d(r_chs, r_rows, r_cols);

        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    temp[ch*rows_*cols_ + row*cols_ + col] = data_[ch][row][col];
                }
            }
        } 

        for(unsigned r_ch=0;r_ch<r_chs;r_ch++)
        {
            for(unsigned r_row=0;r_row<r_rows;r_row++)
            {
                for (unsigned r_col=0;r_col<r_cols;r_col++)
                {
                    result.data_[r_ch][r_row][r_col] = temp[r_ch*r_rows*r_cols + r_row*r_cols + r_col];
                }
            }
        }  
        return result;
    }

    Matrix3d Matrix3d::copy()
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = data_[ch][row][col];
                }
            }
        } 
        return result;
    }
    
    Matrix3d Matrix3d::apply_function(std::function<double(double)> func)
    {
        Matrix3d result = Matrix3d(shape[0], shape[1], shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][row][col] = func(data_[ch][row][col]);
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::transpose()
    {
        Matrix3d result = Matrix3d(shape[0], shape[2], shape[1]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[ch][col][row] = data_[ch][row][col];
                }
            }
        } 
        return result;
    }

    Matrix3d Matrix3d::flatten()
    {
        Matrix3d result = Matrix3d(1,1,shape[0]*shape[1]*shape[2]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    result.data_[0][0][ch*rows_*cols_ + row*cols_ +col] = data_[ch][row][col];
                }
            }
        } 
        return result;
    }


    void Matrix3d::random_initialize()
    {
        // std::default_random_engine generator;

        std::random_device rd;
        std::mt19937 generator( rd() );
        double num = 0;
        
        // std::normal_distribution<double> distribution(0.0,0.01);
        std::normal_distribution<double> distribution(0.0,0.1);

        Matrix3d result = Matrix3d(shape[0], shape[2], shape[1]);
        for(unsigned ch=0;ch<chs_;ch++)
        {
            for(unsigned row=0;row<rows_;row++)
            {
                for (unsigned col=0;col<cols_;col++)
                {
                    // while(num<=0){
                    //     num = distribution(generator);
                    // }
                    // data_[ch][row][col] = (double) rand() / (RAND_MAX + 0.0 );
                    data_[ch][row][col] = distribution(generator);
                    // data_[ch][row][col] = num;
                    // std::cout<<"random initialize:"<<data_[ch][row][col]<<std::endl;
                }
            }
        } 
    }

    std::vector<int> Matrix3d::argmax(int axis)
    {
        assert(shape[0]==1);
        if(axis==1){
            std::vector<double> maximum_val(shape[1],2.22507e-308);
            std::vector<int> maximum_index(shape[1],0);
            for(unsigned ch=0;ch<chs_;ch++){
                for(unsigned row=0;row<rows_;row++){
                    for (unsigned col=0;col<cols_;col++){
                        if (data_[ch][row][col]>maximum_val[row]){
                            maximum_val[row] = data_[ch][row][col];
                            maximum_index[row] = col;
                        }
                    }
                }
            } 
            return maximum_index;
        }
        else if(axis==2){
            std::vector<double> maximum_val(shape[2],2.22507e-308);
            std::vector<int> maximum_index(shape[2],0);
            for(unsigned ch=0;ch<chs_;ch++){
                for(unsigned col=0;col<cols_;col++){
                    for (unsigned row=0;row<rows_;row++){
                        if (data_[ch][row][col]>maximum_val[col]){
                            maximum_val[col] = data_[ch][row][col];
                            maximum_index[col] = row;
                        }
                    }
                }
            } 
            return maximum_index;
        }
        else{
            std::vector<int> maximum_index;
            return maximum_index;
        }
    }

    
    std::vector<int> Matrix3d::argmin(int axis)
    {
        assert(shape[0]==1);
        if(axis==1){
            std::vector<double> minimum_val(shape[1],1.79769e+308);
            std::vector<int> minimum_index(shape[1],0);
            for(unsigned ch=0;ch<chs_;ch++){
                for(unsigned row=0;row<rows_;row++){
                    for (unsigned col=0;col<cols_;col++){
                        if (data_[ch][row][col]<minimum_val[row]){
                            minimum_val[row] = data_[ch][row][col];
                            minimum_index[row] = col;
                        }
                    }
                }
            } 
            return minimum_index;
        }
        else if(axis==2){
            std::vector<double> minimum_val(shape[2],1.79769e+308);
            std::vector<int> minimum_index(shape[2],0);
            for(unsigned ch=0;ch<chs_;ch++){
                for(unsigned col=0;col<cols_;col++){
                    for (unsigned row=0;row<rows_;row++){
                        if (data_[ch][row][col]<minimum_val[col]){
                            minimum_val[col] = data_[ch][row][col];
                            minimum_index[col] = row;
                        }
                    }
                }
            } 
            return minimum_index;
        }
        else{
            std::vector<int> minimum_index;
            return minimum_index;
        }
    }

    double Matrix3d::sum()
    {
        double summation = 0;
        for(unsigned ch=0;ch<chs_;ch++){
            for(unsigned row=0;row<rows_;row++){
                for(unsigned col=0;col<cols_;col++){
                        summation += data_[ch][row][col];
                }
            }
        } 
        return summation;
    }



    class Matrix3d_batch
    {
        public:
            Matrix3d_batch(unsigned bats=0, unsigned chs=0, unsigned rows=0, unsigned cols=0, double init_val=0);
            unsigned bats_, chs_ , rows_, cols_;
            std::vector<unsigned> shape{0, 0, 0, 0};
            std::vector<Matrix3d> batch_data_;

            double at(unsigned bat, unsigned ch, unsigned row, unsigned col);

            Matrix3d_batch operator+ (const Matrix3d_batch & target);
            Matrix3d_batch operator- (const Matrix3d_batch & target);
            Matrix3d_batch operator* (const Matrix3d_batch & target);
            Matrix3d_batch dot (const Matrix3d_batch & target);
            Matrix3d_batch square();
            Matrix3d_batch sqrt();
            Matrix3d_batch operator/ (const Matrix3d_batch & target);
            
            Matrix3d_batch add_scalar (double scaler = 0);
            Matrix3d_batch multiply_scalar (double scaler = 1);
            Matrix3d_batch divide_scalar (double scaler = 1);
            Matrix3d_batch reshape (unsigned r_chs, unsigned r_rows, unsigned r_cols);
            Matrix3d_batch copy ();
            Matrix3d_batch apply_function(std::function<double(double)> func);
            Matrix3d_batch transpose();

            double sum();
            
            void random_initialize();
            std::vector<std::vector<int>> argmax(int axis);
            std::vector<std::vector<int>> argmin(int axis);
            // batch sum
    };

    Matrix3d_batch::Matrix3d_batch(unsigned bats, unsigned chs, unsigned rows, unsigned cols, double init_val)
    : bats_ (bats), chs_ (chs), rows_ (rows), cols_ (cols)
    {
        // if (chs == 0 || rows == 0 || cols == 0)
        // std::cout<<"Matrix constructor has 0 size"<<std::endl; //throw statement
        for(int i=0;i<bats;i++)
            batch_data_.push_back(Matrix3d(chs,rows,cols,init_val));
        shape[0] = bats;
        shape[1] = chs;
        shape[2] = rows;
        shape[3] = cols;
    }

    double Matrix3d_batch::at(unsigned bat, unsigned ch, unsigned row, unsigned col)
    {
        return  batch_data_[bat].at(ch, row, col);
    }

    Matrix3d_batch Matrix3d_batch::operator+(const Matrix3d_batch & target)
    {
        assert(shape[0]==target.shape[0]);
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat] + target.batch_data_[bat];
        return result;
    }

    Matrix3d_batch Matrix3d_batch::operator-(const Matrix3d_batch & target)
    {
        assert(shape[0]==target.shape[0]);
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat] - target.batch_data_[bat];
        return result;
    }

    Matrix3d_batch Matrix3d_batch::operator*(const Matrix3d_batch & target)
    {
        assert(shape[0]==target.shape[0]);
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],target.shape[3],0);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat] * target.batch_data_[bat];
        return result;
    }

    Matrix3d_batch Matrix3d_batch::operator/(const Matrix3d_batch & target)
    {
        assert(shape[0]==target.shape[0]);
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],target.shape[3],0);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat] / target.batch_data_[bat];
        return result;
    }

    Matrix3d_batch Matrix3d_batch::dot(const Matrix3d_batch & target)
    {
        assert(shape[0]==target.shape[0]);
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3],0);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].dot(target.batch_data_[bat]);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::square()
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3],0);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].square();
        return result;
    }

    Matrix3d_batch Matrix3d_batch::sqrt()
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3],0);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].sqrt();
        return result;
    }

    Matrix3d_batch Matrix3d_batch::add_scalar(double scaler)
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].add_scalar(scaler);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::multiply_scalar(double scaler)
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].multiply_scalar(scaler);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::divide_scalar(double scaler)
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].divide_scalar(scaler);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::reshape(unsigned r_chs, unsigned r_rows, unsigned r_cols)
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].reshape(r_chs, r_rows, r_cols);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::copy()
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].copy();
        return result;
    }

    Matrix3d_batch Matrix3d_batch::apply_function(std::function<double(double)> func)
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[2],shape[3]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].apply_function(func);
        return result;
    }

    Matrix3d_batch Matrix3d_batch::transpose()
    {
        Matrix3d_batch result = Matrix3d_batch(shape[0],shape[1],shape[3],shape[2]);
        for(unsigned bat=0;bat<bats_;bat++)
            result.batch_data_[bat] = batch_data_[bat].transpose();
        return result;
    }

    void Matrix3d_batch::random_initialize()
    {
        for(unsigned bat=0;bat<bats_;bat++)
            batch_data_[bat].random_initialize();
    }

    std::vector<std::vector<int>> Matrix3d_batch::argmax(int axis)
    {
        std::vector<std::vector<int>> maximum_index;
        for(unsigned bat=0;bat<bats_;bat++)
            maximum_index.push_back(batch_data_[bat].argmax((axis-1)));
        return maximum_index;
    }

    std::vector<std::vector<int>> Matrix3d_batch::argmin(int axis)
    {
        std::vector<std::vector<int>> minimum_index;
        for(unsigned bat=0;bat<bats_;bat++)
            minimum_index.push_back(batch_data_[bat].argmin((axis-1)));
        return minimum_index;
    }

    double Matrix3d_batch::sum()
    {
        double summation = 0;
        for(unsigned bat=0;bat<bats_;bat++)
            summation += batch_data_[bat].sum();
        return summation;
    }
}