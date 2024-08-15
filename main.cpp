#include "model.h"
#include "data_utils.h"

int main()
{
    Model model;

    int img_resize = 32;

    std::string home_path = "C:\\Users\\Yu\\Desktop\\mnist\\";
    std::string train_folder_path = "mnist_train\\";
    std::string test_folder_path = "mnist_test\\";
    

    std::string file_path;
    std::vector<std::string> file_paths;
    std::vector<std::vector<std::string>> return_file_paths_batch;
    std::vector<std::vector<int>> return_labels_batch;

    int batch_size = 25;
    

	int num_labels;
    // dataset(home_path, train_txt_path, file_paths,"txt");
    // dataset(home_path, test_txt_path, file_paths,"txt");
    
    dataset(home_path, train_folder_path, file_paths,"folder");
    dataloader(file_paths, return_file_paths_batch, return_labels_batch, batch_size, num_labels, true);

   

    Matrix3d_batch data_batch(batch_size,3,img_resize,img_resize);
	Matrix3d_batch label_batch(batch_size,1,1,num_labels,0);
	Matrix3d_batch output_batch(batch_size,1,1,num_labels);
    std::vector<std::vector<int>> pred_label;
    double correct = 0;
    double total = 0;
    double accuracy = 0;
    double total_loss = 0;
    double loss;
    Matrix3d grads(1,1,num_labels,0);
    // Matrix3d feature_extract;
    
    int epochs = 25;

    std::cout<<"------------------"<<std::endl;
    std::cout<<"training start"<<std::endl;

    for(int epoch = 0;epoch<epochs;epoch++){
        std::cout<<"epoch:"<<epoch<<std::endl;
        std::vector<std::vector<std::string>> return_file_paths_batch;
        std::vector<std::vector<int>> return_labels_batch;
        dataloader(file_paths, return_file_paths_batch, return_labels_batch, batch_size, num_labels, true);
        correct = 0;
        total = 0;
        accuracy = 0;
        total_loss = 0;
        for(int i=0;i<return_file_paths_batch.size();i++){
            std::cout<<"batch:"<<i<<std::endl;
            for(int j=0;j<return_file_paths_batch[i].size();j++){
                label_batch.batch_data_[j] = label_to_mat(return_labels_batch[i][j], num_labels);
                data_batch.batch_data_[j] = load_img(return_file_paths_batch[i][j], img_resize).normalize(0,255,0,1);

                output_batch.batch_data_[j] = model.forward(data_batch.batch_data_[j]);
                total++;
            }
            
            
            loss = 0;
            grads = grads.multiply_scalar(0);
            pred_label = output_batch.argmax(2);
            for(int k=0;k<pred_label.size();k++){

                if (pred_label[k][0] == return_labels_batch[i][k]){
                    correct++;
                }

                // loss = loss + loss_func::MSE_loss(output_batch.batch_data_[k],label_batch.batch_data_[k]);
                loss = loss + loss_func::CE_loss(output_batch.batch_data_[k],label_batch.batch_data_[k]);
                // grads = grads + loss_func::MSE_loss_derivate(output_batch.batch_data_[k],label_batch.batch_data_[k],"sigmoid");
                grads = grads + loss_func::CE_loss_derivate(output_batch.batch_data_[k],label_batch.batch_data_[k],"softmax");

                
            }

            if(i%100==0){
                std::cout<<"accuracy:"<<correct/total<<std::endl;
            }

            
            loss = loss/batch_size;
            // grads = grads.divide_scalar(batch_size);
            std::cout<<"batch training loss:"<<loss<<std::endl;
            model.backward(grads);
            std::cout<<"-----"<<std::endl;
            total_loss += loss;
           
        }
        std::cout<<"------------------"<<std::endl;
        // std::cout<<"correct:"<<correct<<std::endl;
        // std::cout<<"total:"<<total<<std::endl;
        accuracy = correct/total;
        std::cout<<"accuracy:"<<accuracy<<std::endl;
        std::cout<<"total_loss:"<<total_loss<<std::endl;
        std::cout<<"------------------"<<std::endl;
    }
    
    std::string file_path_test;
    std::vector<std::string> file_paths_test;
    std::vector<std::vector<std::string>> return_file_paths_batch_test;
    std::vector<std::vector<int>> return_labels_batch_test;

    // Matrix3d feature_extract_test;
    int batch_size_test = 25;
    int num_labels_test;

    dataset(home_path, test_folder_path, file_paths_test,"folder");
    dataloader(file_paths_test, return_file_paths_batch_test, return_labels_batch_test, batch_size_test, num_labels_test, false);

    Matrix3d_batch data_batch_test(batch_size_test,3,img_resize,img_resize);
	Matrix3d_batch label_batch_test(batch_size_test,1,1,num_labels_test,0);
	Matrix3d_batch output_batch_test(batch_size_test,1,1,num_labels_test);

    double correct_test = 0;
    double total_test = 0;
    double accuracy_test = 0;

    std::vector<std::vector<int>> pred_label_test;

    std::cout<<"------------------"<<std::endl;
    std::cout<<"testing start"<<std::endl;

    
    

    for(int l=0;l<return_file_paths_batch_test.size();l++){
        for(int m=0;m<return_file_paths_batch_test[l].size();m++){
            
            label_batch_test.batch_data_[m] = label_to_mat(return_labels_batch_test[l][m], num_labels_test);
            data_batch_test.batch_data_[m] = load_img(return_file_paths_batch_test[l][m], img_resize).normalize(0,255,0,1);
            // feature_extract = Color_Histogram(data_batch.batch_data_[j]);
            // output_batch.batch_data_[j] = model.foward(feature_extract);
            output_batch_test.batch_data_[m] = model.forward(data_batch_test.batch_data_[m]);
            total_test++;
        }
        
        pred_label_test = output_batch_test.argmax(2);
        for(int n=0;n<pred_label_test.size();n++){
            if (pred_label_test[n][0] == return_labels_batch_test[l][n]){
                correct_test++;
            }
        }
    }
    std::cout<<"------------------"<<std::endl;
    std::cout<<"testing result"<<std::endl;
    // std::cout<<"correct:"<<correct_test<<std::endl;
    // std::cout<<"total:"<<total_test<<std::endl;
    accuracy_test = correct_test/total_test;
    std::cout<<"accuracy:"<<accuracy_test<<std::endl;
    std::cout<<"------------------"<<std::endl;

    system("pause");
}