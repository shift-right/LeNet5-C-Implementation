#pragma once

#include "matrix.h"
#include "nn.h"

using namespace matrix;
using namespace nn;

class Model
{
    public:
        Model(){};
        Matrix3d forward(Matrix3d &inputs);
        void backward(Matrix3d &grads);
        Conv_Layer conv_0{1,6,5,1,0,true,1e-03,"ReLU","None"};
        Max_Pool max_0{2,2,0,true};
        Conv_Layer conv_1{6,16,5,1,0,true,1e-03,"ReLU","None"};
        Max_Pool max_1{2,2,0,true};
        FC_Layer layer_0{(16*5*5),120,true,1e-03,"ReLU","None",false};
        FC_Layer layer_1{120,84,true,1e-03,"ReLU","None",false};
        FC_Layer layer_2{84,10,true,1e-03,"softmax","None",false,true}; //last_layer = true

};

Matrix3d Model::forward(Matrix3d &inputs)
{
    Matrix3d outputs_0 = conv_0.forward(inputs);
    Matrix3d outputs_1 = max_0.forward(outputs_0);
    Matrix3d outputs_2 = conv_1.forward(outputs_1);
    Matrix3d outputs_3 = max_1.forward(outputs_2);
  
    Matrix3d outputs_3_fla = outputs_3.flatten();
    Matrix3d outputs_4 = layer_0.forward(outputs_3_fla);
    Matrix3d outputs_5 = layer_1.forward(outputs_4);
    Matrix3d outputs_6 = layer_2.forward(outputs_5);

    return outputs_6;
}

void Model::backward(Matrix3d &grads)
{
    

    Matrix3d grads_0 = layer_2.backward(grads);
    Matrix3d grads_1 = layer_1.backward(grads_0);
    Matrix3d grads_2 = layer_0.backward(grads_1);
    Matrix3d grads_2_res = grads_2.reshape(max_1.output_channels,max_1.output_shape,max_1.output_shape);
    Matrix3d grads_3 = max_1.backward(grads_2_res);
    Matrix3d grads_4 = conv_1.backward(grads_3);
    Matrix3d grads_5 = max_0.backward(grads_4);
    Matrix3d grads_6 = conv_0.backward(grads_5);
}

