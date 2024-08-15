#pragma once
# include <vector>
#include "matrix.h"

using namespace matrix;

namespace nn
{
    class act_func
    {
        public:
            act_func(){};
            static double act_ReLU(double input);
            static double de_ReLU(double input);
            static double act_sigmoid(double act_input);
            static double de_sigmoid(double act_input);
            static Matrix3d act_softmax(Matrix3d & inputs);
            // static Matrix3d de_softmax(Matrix3d & inputs);

    };

    double act_func::act_ReLU(double input)
    {
        if (input>0)
            return input;
        else
            return 0;
    }

    double act_func::de_ReLU(double input)
    {
        if (input>0)
            return 1;
        else
            return 0;
    }

    double act_func::act_sigmoid(double input)
    {
        return 1/(1+exp(-1*input));
    }

    double act_func::de_sigmoid(double act_input)
    {
        return act_input*(1-act_input);
    }

    Matrix3d act_func::act_softmax(Matrix3d & inputs)
    {
        Matrix3d results = inputs.copy();
        double exp_sum = 0;
        for(int i =0;i<inputs.shape[0];i++){
            for(int j=0;j<inputs.shape[1];j++){
                for(int k=0;k<inputs.shape[2];k++){
                    results.data_[i][j][k] = exp(inputs.data_[i][j][k]);
                    exp_sum+= results.data_[i][j][k];
                }
            }
        }
        
        results = results.divide_scalar(exp_sum);
        return results;
    }

    class loss_func
    {
        public:
            loss_func(){};
            static double MSE_loss(Matrix3d &y_pred, Matrix3d &y);
            static Matrix3d MSE_loss_derivate(Matrix3d &y_pred, Matrix3d &y, std::string act_func_name);
            static double CE_loss(Matrix3d &y_pred, Matrix3d &y);
            static Matrix3d CE_loss_derivate(Matrix3d &y_pred, Matrix3d &y, std::string act_func_name);

    };

    double loss_func::MSE_loss(Matrix3d &y_pred, Matrix3d &y)
    {
        assert(y_pred.shape==y.shape);

        Matrix3d result = (y_pred-y).dot(y_pred-y);
        return result.sum();
    }

    Matrix3d loss_func::MSE_loss_derivate(Matrix3d &y_pred, Matrix3d &y, std::string act_func_name)
    {
        assert(y_pred.shape==y.shape);

        Matrix3d grads = (y_pred-y).multiply_scalar(2);
        
        Matrix3d results;

        int label_index_row = 0;
        int label_index_col = 0;

        if(act_func_name == "softmax"){
            double label_val = 0;
            for(int j=0;j<y.shape[1];j++){
                for(int k=0;k<y.shape[2];k++){
                    if(y.data_[0][j][k]==1){
                        label_val = y_pred.data_[0][j][k];
                        label_index_row = j;
                        label_index_col = k;
                    }
                }
            }
            results = Matrix3d(1,y.shape[1],y.shape[2],(-1*label_val));
            results = results - y_pred;
            results.data_[0][label_index_row][label_index_col] = label_val*(1-label_val);
            results = grads.dot(results);
        }
        else if (act_func_name == "sigmoid")
        {
            results = grads.dot(y_pred.dot((y_pred.multiply_scalar(-1)).add_scalar(1)));
        }
        else{
            results = grads;
        }

        return results;
    }

    double loss_func::CE_loss(Matrix3d &y_pred, Matrix3d &y)
    {
        assert(y_pred.shape==y.shape);
        double loss = 0;
       
        for(int j=0;j<y.shape[1];j++){
            for(int k=0;k<y.shape[2];k++){
                if(y.data_[0][j][k]==1){
                    loss -= log(y_pred.data_[0][j][k]+1e-8);
                }
            }
        }

       return loss;
    }

    Matrix3d loss_func::CE_loss_derivate(Matrix3d &y_pred, Matrix3d &y, std::string act_func_name)
    {
        assert(y_pred.shape==y.shape);
        Matrix3d results(1,y.shape[1],y.shape[2],0);
        if(act_func_name == "softmax"){
            for(int j=0;j<y.shape[1];j++){
                for(int k=0;k<y.shape[2];k++){
                   if(y.data_[0][j][k]==1){
                        results.data_[0][j][k] = y_pred.data_[0][j][k]-1;
                    }
                    else{
                        results.data_[0][j][k] = y_pred.data_[0][j][k];
                    }
                }
            }
        }
        else if (act_func_name == "sigmoid")
        {
            double target_val = 0;
            for(int j=0;j<y.shape[1];j++){
                for(int k=0;k<y.shape[2];k++){
                   if(y.data_[0][j][k]==1){
                        target_val -= 1/(y_pred.data_[0][j][k]+1e-8);
                    }
                }
            }
            results = results.add_scalar(target_val);
            results = results.dot(y_pred.apply_function(act_func::de_sigmoid));
        }
        else{
            double target_val = 0;
            for(int j=0;j<y.shape[1];j++){
                for(int k=0;k<y.shape[2];k++){
                   if(y.data_[0][j][k]==1){
                        target_val -= 1/(y_pred.data_[0][j][k]+1e-8);
                    }
                }
            }
            results = results.add_scalar(target_val);
        }
        return results;
    }


    class optimizer
    {
        public:
            double beta1_ = 0.9;
            double beta2_ = 0.999;
            double e_ = 1e-8;
            double t = 1;

            Matrix3d mt_;
            Matrix3d vt_;

            Matrix3d_batch mt_batch_;
            Matrix3d_batch vt_batch_;
            
            optimizer(double beta1 = 0.9, double beta2 = 0.999);
            Matrix3d enhance_grad(Matrix3d & grads);
            Matrix3d_batch enhance_grad_batch(Matrix3d_batch & grads);
            Matrix3d Adam(Matrix3d & grads);
            Matrix3d_batch Adam_batch(Matrix3d_batch & grads);
    };
    optimizer::optimizer(double beta1, double beta2)
    : beta1_ (beta1), beta2_ (beta2){}

    Matrix3d optimizer::enhance_grad(Matrix3d & grads)
    {
        Matrix3d result(grads.chs_,grads.rows_,grads.cols_,0);
        for(unsigned ch=0;ch<grads.chs_;ch++)
        {
            for(unsigned row=0;row<grads.rows_;row++)
            {
                for (unsigned col=0;col<grads.cols_;col++)
                {
                    if(grads.data_[ch][row][col]>0){
                        result.data_[ch][row][col] = grads.data_[ch][row][col] + e_;
                    }
                    else{
                        result.data_[ch][row][col] = grads.data_[ch][row][col] -e_;
                    }
                    
                }
            }
        } 
        return result;
    }


    Matrix3d_batch optimizer::enhance_grad_batch(Matrix3d_batch & grads)
    {
        Matrix3d_batch result(grads.bats_,grads.chs_,grads.rows_,grads.cols_,0);
        for(unsigned bat=0;bat<grads.bats_;bat++)
            result.batch_data_[bat] = enhance_grad(grads.batch_data_[bat]);
        return result;
    }

    Matrix3d optimizer::Adam(Matrix3d & grads)
    {
        Matrix3d local_grads = grads.copy();
        local_grads = enhance_grad(local_grads);
        
        mt_ = mt_.multiply_scalar(beta1_) + local_grads.multiply_scalar((1-beta1_));

        Matrix3d local_grads_square = local_grads.square();
        vt_ = vt_ .multiply_scalar(beta2_) + local_grads_square.multiply_scalar((1-beta2_));

        Matrix3d mt_hat = mt_.divide_scalar((1- pow(beta1_,t)));
        Matrix3d vt_hat = vt_.divide_scalar((1- pow(beta2_,t)));
        Matrix3d results = (mt_hat / (vt_hat.sqrt()));

        t+=1;

        return results;
    }

    Matrix3d_batch optimizer::Adam_batch(Matrix3d_batch & grads)
    {
        Matrix3d_batch local_grads = grads.copy();
        local_grads = enhance_grad_batch(local_grads);

        mt_batch_ = mt_batch_.multiply_scalar(beta1_) + local_grads.multiply_scalar((1-beta1_));

        Matrix3d_batch local_grads_square = local_grads.square();
        vt_batch_ = vt_batch_.multiply_scalar(beta2_) + local_grads_square.multiply_scalar((1-beta2_));

        Matrix3d_batch mt_hat_batch = mt_batch_.divide_scalar((1- pow(beta1_,t)));
        Matrix3d_batch vt_hat_batch = vt_batch_.divide_scalar((1- pow(beta2_,t)));
        Matrix3d_batch results = mt_hat_batch / (vt_hat_batch.sqrt());

        t+=1;
        
        return results;
    }

    class NN
    {
        public:
            bool if_backward_ = true;
            bool if_optimizer_ = false;
            bool if_activate_ = false;
            bool if_last_layer_ = false;
            bool if_bias_ = false;
            bool if_optimizer_init_ = false;
            double lr_ = 0.005;
            Matrix3d save_inputs;
            Matrix3d save_outputs;

            NN(bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_bias = false, bool if_last_layer= false);
            Matrix3d activate(Matrix3d &outputs);
            Matrix3d derivate();
            std::string act_func_name_ = "None";
            std::string optimizer_name_= "None"; 
            
    };

    NN::NN(bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_bias, bool if_last_layer)
    : if_backward_ (if_backward), lr_ (lr), act_func_name_ (act_func_name), optimizer_name_ (optimizer_name), if_bias_ (if_bias) ,if_last_layer_ (if_last_layer)
    {
        if (act_func_name_ != "None")
            if_activate_ = true;
        if (optimizer_name_ != "None")
            if_optimizer_ = true;
    }

    Matrix3d NN::activate(Matrix3d &outputs)
    {
        Matrix3d act_outputs;
        if (act_func_name_ == "ReLU"){
            act_outputs = outputs.apply_function(act_func::act_ReLU);
            return act_outputs;
        }
        else if (act_func_name_ == "sigmoid"){
            act_outputs = outputs.apply_function(act_func::act_sigmoid);
            return act_outputs;
        }   
        else if (act_func_name_ == "softmax"){
            act_outputs = act_func::act_softmax(outputs);
            return act_outputs;
        }
        else
            return outputs;
    }

    Matrix3d NN::derivate()
    {
        Matrix3d de_outputs;
        
        if (act_func_name_ == "ReLU"){
            de_outputs = save_outputs.apply_function(act_func::de_ReLU);
            return de_outputs;
        }
        else if (act_func_name_ == "sigmoid"){
            de_outputs = save_outputs.apply_function(act_func::de_sigmoid);
            return de_outputs;
        } 
        else
            return Matrix3d(save_outputs.shape[0],save_outputs.shape[1],save_outputs.shape[2],1);
    }

    class FC_Layer: public NN
    {
        public:
            
            Matrix3d weights;
            Matrix3d backward_grads;
            Matrix3d weights_grad;
            Matrix3d bias;
            
            FC_Layer(int D_in, int D_out, bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_bias = false ,bool if_last_layer= false);
            Matrix3d forward(Matrix3d & inputs);
            Matrix3d backward(Matrix3d & grads);
            optimizer OP_FC{};
    };

    FC_Layer::FC_Layer(int D_in, int D_out, bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_bias, bool if_last_layer) //initialize
    : NN(if_backward, lr, act_func_name, optimizer_name, if_bias, if_last_layer)
    {
        weights = Matrix3d(1, D_in, D_out);
        weights.random_initialize();
        weights = weights.divide_scalar(sqrt(D_in));

        if(if_bias){
            bias = Matrix3d(1, 1, D_out);
            bias.random_initialize();
        }
        
        
    }

    Matrix3d FC_Layer::forward(Matrix3d & inputs)
    {
        save_inputs = inputs.copy();
        Matrix3d outputs = inputs * weights;

        if(if_bias_){
            outputs = outputs + bias;
        }

        if(if_activate_){
            if (act_func_name_!="ReLU"){
                outputs = activate(outputs); //save output after activate
                save_outputs = outputs.copy();
            }
            else{ ////save output before activate
                save_outputs = outputs.copy();
                outputs = activate(outputs);
            }
        }
        else{
            save_outputs = outputs.copy();
        }
        
        return outputs;
    }

    Matrix3d FC_Layer::backward(Matrix3d & grads)
    {
        Matrix3d local_grads = grads.copy();
        
        if (if_backward_){
            if(if_activate_){
                if(!if_last_layer_){
                    local_grads = local_grads.dot(derivate());
                }
            }
            
            weights_grad = save_inputs.transpose() * local_grads;

            
            
            backward_grads = local_grads * weights.transpose();
           

            if (if_optimizer_){
                if (optimizer_name_ =="Adam"){
                    if (if_optimizer_init_){
                        Matrix3d coe = OP_FC.Adam(weights_grad);
                        weights = weights - weights_grad.dot(coe).multiply_scalar(lr_);
                    }
                    else{
                        OP_FC.mt_ = Matrix3d(weights.shape[0],weights.shape[1],weights.shape[2],0);
                        OP_FC.vt_ = Matrix3d(weights.shape[0],weights.shape[1],weights.shape[2],0);
                        Matrix3d coe = OP_FC.Adam(weights_grad);
                        weights = weights - weights_grad.dot(coe).multiply_scalar(lr_);
                        if_optimizer_init_ = true;
                    }
                    
                }

                // if(if_bias_){
                //     bias = bias + local_grads.multiply_scalar(lr_*coe);
                // }
            }
            else{
                weights = weights - weights_grad.multiply_scalar(lr_);
                if(if_bias_){
                    bias = bias + local_grads.multiply_scalar(lr_);
                }
            }
            return backward_grads;
        }
        else
            return 0;  
    }

    class Conv_Layer: public NN
    {
        public:
            Matrix3d_batch conv_kernel;
            Matrix3d backward_grads;
            Matrix3d_batch kernel_grads;
            int stride_;
            int padding_;
            int kernel_size_;
            int output_shape;

            Conv_Layer(int C_in, int C_out, int kernel_size, int stride, int padding, bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_last_layer= false);
            Matrix3d convolution(Matrix3d &input, Matrix3d &kernel);
            void convolution_derivate(Matrix3d &grad, Matrix3d &kernel, Matrix3d &kernel_grad);

            Matrix3d padding(Matrix3d & input);
            Matrix3d unpadding(Matrix3d & input);
            Matrix3d forward(Matrix3d & inputs);
            Matrix3d backward(Matrix3d & grads);
            optimizer OP_Conv{};
    };

    Conv_Layer::Conv_Layer(int C_in, int C_out, int kernel_size, int stride, int padding, bool if_backward, double lr, std::string act_func_name, std::string optimizer_name, bool if_last_layer) //initialize
    : NN(if_backward, lr, act_func_name, optimizer_name, if_last_layer)
    {
        kernel_size_ = kernel_size;
        stride_ = stride;
        padding_ = padding;

        conv_kernel = Matrix3d_batch(C_out, C_in, kernel_size, kernel_size);
        kernel_grads = Matrix3d_batch(C_out, C_in, kernel_size, kernel_size);

        conv_kernel.random_initialize();
        conv_kernel.divide_scalar(sqrt(C_in));
    }
    Matrix3d Conv_Layer::padding(Matrix3d &inputs)
    {
        int pad_row = inputs.shape[1] + padding_*2; 
        int pad_col = inputs.shape[2] + padding_*2; 
        Matrix3d padding_inputs(inputs.shape[0], pad_row, pad_col, 0);

        for(int i=0;i<inputs.shape[0];i++){
            for(int j=0;j<inputs.shape[1];j++){
                for(int k=0;k<inputs.shape[2];k++){
                    padding_inputs.data_[i][j+padding_][k+padding_] = inputs.data_[i][j][k];
                }
            }
        }
        return padding_inputs;
    }

    Matrix3d Conv_Layer::unpadding(Matrix3d &inputs)
    {
        int unpad_row = inputs.shape[1] - padding_*2; 
        int unpad_col = inputs.shape[2] - padding_*2; 


        Matrix3d unpadding_inputs(inputs.shape[0], unpad_row, unpad_col, 0);


        for(int i=0;i<unpadding_inputs.shape[0];i++){
            for(int j=0;j<unpadding_inputs.shape[1];j++){
                for(int k=0;k<unpadding_inputs.shape[2];k++){
                    unpadding_inputs.data_[i][j][k] = inputs.data_[i][j+padding_][k+padding_];
                }
            }
        }
        return unpadding_inputs;
    }

    Matrix3d Conv_Layer::convolution(Matrix3d &input, Matrix3d &kernel)
    {
        assert(input.shape[0]==kernel.shape[0]);
        
        Matrix3d output(1, output_shape, output_shape);

        double channel_sum = 0;
        double total_channel_sum = 0;

        int output_row = 0;
        int output_col = 0;
        
        for(int i=0;i<input.shape[1];i+=stride_){
            if(i+kernel_size_<input.shape[1]+1){
                output_col = 0;
                for(int j=0;j<input.shape[2];j+=stride_){
                    if(j+kernel_size_<input.shape[2]+1){
                        total_channel_sum = 0;
                        for(int k=0;k<kernel.shape[0];k++){ //number of channel
                            channel_sum = 0;
                            for(int l=0;l<kernel.shape[1];l++){ //number of kernel row
                                for(int m=0;m<kernel.shape[2];m++){ //number of kernel col 
                                    channel_sum += input.data_[k][i+l][j+m] * kernel.data_[k][l][m];
                                }
                            }
                            total_channel_sum += channel_sum;
                        }
                        output.data_[0][output_row][output_col] = total_channel_sum;
                        output_col++;
                    }
                    else{
                        continue;
                    }
                }
                output_row++;
            }
            else{
                break;
            }
        }
          
        return output;
    }

    void Conv_Layer::convolution_derivate(Matrix3d &grad, Matrix3d &kernel, Matrix3d &kernel_grad)
    {
        double pixel_grad;

        int output_row = 0;
        int output_col = 0;

        for(int i=0;i<save_inputs.shape[1];i+=stride_){
            if(i+kernel_size_<save_inputs.shape[1]+1){
                output_col = 0;
                for(int j=0;j<save_inputs.shape[2];j+=stride_){
                    if(j+kernel_size_<save_inputs.shape[2]+1){
                        pixel_grad = grad.data_[0][output_row][output_col];
                        for(int k=0;k<kernel.shape[0];k++){ //number of channel
                            for(int l=0;l<kernel.shape[1];l++){ //number of kernel row
                                for(int m=0;m<kernel.shape[2];m++){ //number of kernel col 
                                    kernel_grad.data_[k][l][m] += save_inputs.data_[k][i+l][j+m] * pixel_grad;
                                    backward_grads.data_[k][i+l][j+m] += kernel.data_[k][l][m] * pixel_grad;
                                }
                            }
                        }
                        output_col++;
                    }
                    else{
                        continue;
                    }
                }
                output_row++;
            }
            else{
                break;
            } 
        }
    }

    Matrix3d Conv_Layer::forward(Matrix3d &inputs)
    {
        output_shape = (inputs.shape[1] + 2*padding_ - (kernel_size_-1) -1)/stride_ + 1;

        
        if(padding_ > 0) //if_padding
            inputs = padding(inputs);
        
        save_inputs = inputs.copy();

        
        
        Matrix3d output;
        Matrix3d outputs(conv_kernel.shape[0], output_shape, output_shape);

        for(int i=0;i<conv_kernel.shape[0];i++){
            output = convolution(inputs, conv_kernel.batch_data_[i]); // dead here
            outputs.data_[i] = output.data_[0];
        }

        

        if(if_activate_){
            if (act_func_name_!="ReLU"){
                outputs = activate(outputs); //save output after activate
                save_outputs = outputs;
            }
            else{ ////save output before activate
                save_outputs = outputs;
                outputs = activate(outputs);
            }
        }
        else{
            save_outputs = outputs;
        }

        return outputs;
    }

    Matrix3d Conv_Layer::backward(Matrix3d & grads)
    {
        Matrix3d local_grads = grads.copy();
        if (if_backward_){
            
            backward_grads = Matrix3d(save_inputs.shape[0],save_inputs.shape[1],save_inputs.shape[2],0);

            if(if_activate_){
                if(!if_last_layer_){
                    local_grads = local_grads.dot(derivate());
                }
            }

            Matrix3d grad(1,local_grads.shape[1],local_grads.shape[2]);
            for(int i=0;i<local_grads.shape[0];i++){
                grad.data_[0] = local_grads.data_[i];
                convolution_derivate(grad, conv_kernel.batch_data_[i], kernel_grads.batch_data_[i]);
            }

            if (if_optimizer_){
                if (optimizer_name_ =="Adam"){
                    if (if_optimizer_init_){
                        Matrix3d_batch coe = OP_Conv.Adam_batch(kernel_grads);
                        conv_kernel = conv_kernel - kernel_grads.dot(coe).multiply_scalar(lr_);
                    }
                    else{
                        OP_Conv.mt_batch_ = Matrix3d_batch(conv_kernel.shape[0],conv_kernel.shape[1],conv_kernel.shape[2],conv_kernel.shape[3],0);
                        OP_Conv.vt_batch_ = Matrix3d_batch(conv_kernel.shape[0],conv_kernel.shape[1],conv_kernel.shape[2],conv_kernel.shape[3],0);
                        Matrix3d_batch coe = OP_Conv.Adam_batch(kernel_grads);
                        conv_kernel = conv_kernel - kernel_grads.dot(coe).multiply_scalar(lr_);
                        if_optimizer_init_ = true;
                    }
                    
                }
            }
            else{
                conv_kernel = conv_kernel - kernel_grads.multiply_scalar(lr_);
            }

            if(padding_ > 0){
                backward_grads = unpadding(backward_grads);
            }
            

            return backward_grads;
        }

        else
            return 0;
    }


    class Max_Pool: public NN
    {
        public:
            Matrix3d backward_grads;
            int kernel_size_;
            int stride_;
            int padding_;
            int output_channels;
            int output_shape;

            Matrix3d max_index_rows;
            Matrix3d max_index_cols;

            Max_Pool(int kernel_size, int stride, int padding, bool if_backward, bool if_last_layer= false);
            Matrix3d max_pool(Matrix3d &inputs);
            void max_pool_derivate(Matrix3d &grads);

            Matrix3d padding(Matrix3d & inputs);
            Matrix3d unpadding(Matrix3d & inputs);
            Matrix3d forward(Matrix3d & inputs);
            Matrix3d backward(Matrix3d & grads);

    };

    Max_Pool::Max_Pool(int kernel_size, int stride, int padding, bool if_backward, bool if_last_layer)
    :NN(if_backward, 0, "None", "None", if_last_layer)
    {
        kernel_size_ = kernel_size;
        stride_ = stride;
        padding_ = padding;
    }

    Matrix3d Max_Pool::padding(Matrix3d &inputs)
    {
        int pad_row = inputs.shape[1] + padding_*2; 
        int pad_col = inputs.shape[2] + padding_*2; 
        Matrix3d padding_input(inputs.shape[0], pad_row, pad_col, 0);

        for(int i=0;i<inputs.shape[0];i++){
            for(int j=0;j<inputs.shape[1];j++){
                for(int k=0;k<inputs.shape[2];k++){
                    padding_input.data_[i][j+padding_][k+padding_] = inputs.data_[i][j][k];
                }
            }
        }
        return padding_input;
    }

    Matrix3d Max_Pool::unpadding(Matrix3d &inputs)
    {
        int unpad_row = inputs.shape[1] - padding_*2; 
        int unpad_col = inputs.shape[2] - padding_*2; 
        Matrix3d unpadding_inputs(inputs.shape[0], unpad_row, unpad_col, 0);

        for(int i=0;i<unpadding_inputs.shape[0];i++){
            for(int j=0;j<unpadding_inputs.shape[1];j++){
                for(int k=0;k<unpadding_inputs.shape[2];k++){
                    unpadding_inputs.data_[i][j][k] = inputs.data_[i][j+padding_][k+padding_];
                }
            }
        }
        return unpadding_inputs;
    }

    Matrix3d Max_Pool::max_pool(Matrix3d &inputs)
    {
        int max_index_row = 0;
        int max_index_col = 0;
        double max_val = 0;

        Matrix3d outputs(inputs.shape[0], output_shape, output_shape);
        max_index_rows = Matrix3d (inputs.shape[0], output_shape, output_shape);
        max_index_cols = Matrix3d (inputs.shape[0], output_shape, output_shape);
        
        int output_row = 0;
        int output_col = 0;

        for(int m=0;m<inputs.shape[0];m++){
            output_row = 0;
            for(int i=0;i<inputs.shape[1];i+=stride_){
                if(i+kernel_size_<inputs.shape[1]+1){
                    output_col = 0;
                    for(int j=0;j<inputs.shape[2];j+=stride_){
                        if(j+kernel_size_<inputs.shape[2]+1){
                            max_val = 0;
                            max_index_row = 0;
                            max_index_col = 0;
                            for(int k=0;k<kernel_size_;k++){ //number of kernel row
                                for(int l=0;l<kernel_size_;l++){ //number of kernel col
                                    if(inputs.data_[m][i+k][j+l] > max_val){
                                        max_val = inputs.data_[m][i+k][j+l];
                                        max_index_row = i+k;
                                        max_index_col = j+l;
                                    }
                                }
                            }
                            outputs.data_[m][output_row][output_col] = max_val;
                            max_index_rows.data_[m][output_row][output_col] = max_index_row;
                            max_index_cols.data_[m][output_row][output_col] = max_index_col;
                            output_col++;
                        }
                        else{
                            continue;
                        }
                    }
                    output_row++;
                }
                else{
                    continue;
                }

               
            }
        }
        return outputs;
    }

    void Max_Pool::max_pool_derivate(Matrix3d &grads)
    {
        Matrix3d local_grads = grads.copy();
        double pixel_grad;
        for(int i=0;i<local_grads.shape[0];i++){
            for(int j=0;j<local_grads.shape[1];j++){
                for(int k=0;k<local_grads.shape[2];k++){
                    backward_grads.data_[i][(max_index_rows.data_[i][j][k])][(max_index_cols.data_[i][j][k])] += local_grads.data_[i][j][k];
                }
            }
        }
    }


    Matrix3d Max_Pool::forward(Matrix3d &inputs)
    {
        output_shape = (inputs.shape[1] + 2*padding_ - (kernel_size_-1) -1)/stride_ + 1;
        if(padding_ > 0)
            inputs = padding(inputs);

        save_inputs = inputs;

        output_channels = inputs.shape[0];

        
        Matrix3d outputs;

        outputs = max_pool(inputs);

        return outputs;
    }

    Matrix3d Max_Pool::backward(Matrix3d & grads)
    {
        Matrix3d local_grads = grads.copy();
        if (if_backward_){
        
            backward_grads = Matrix3d(save_inputs.shape[0],save_inputs.shape[1],save_inputs.shape[2],0);

            max_pool_derivate(local_grads);

            if(padding_ > 0){
                    backward_grads = unpadding(backward_grads);
            }

            return backward_grads;
        }
        else
            return 0;
    }
}
