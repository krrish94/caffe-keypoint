/*
 Copyright 2015 Tomas Pfister
 Adapted by J. Krishna Murthy for caffe-keypoint
*/

// Include this layer only if Caffe has been compiled with OpenCV

#ifdef USE_OPENCV

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/heatmap_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

#include <stdint.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


namespace caffe
{

// Destructor
template <typename Dtype>
HeatmapDataLayer<Dtype>::~HeatmapDataLayer<Dtype>() {
    this->StopInternalThread();
}


// Performs layer-specific setup (reshape input/output blobs, prefetch data, etc.)
template<typename Dtype>
void HeatmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    // Get heatmap data parameters (parameters passed to this layer in the prototxt)
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Seed the pseudo rng
    const unsigned int rng_seed = caffe_rng_rand();
    srand(rng_seed);

    // Path to file containing ground-truth annotations
    LOG(INFO) << "Loading annotations from " << heatmap_data_param.source();

    // Open the input file stream
    std::ifstream infile(heatmap_data_param.source().c_str());
    // Strings to store name of the image, labels, crop information, and cluster information
    string img_name, labels;

    // Read input file
    while(infile >> img_name >> labels){

    	// Read comma-separated list of regression labels

    	// Vector to store all labels
    	std::vector<float> label;
    	
    	std::istringstream ss(labels);
    	// Read the labels (they are delimited by commas)
    	int labelCounter = 1;
    	while(ss){
    		std::string s;
    		if(!std::getline(ss, s, ',')){
    			break;
    		}
    		label.push_back(atof(s.c_str()));
    	}
    	labelCounter++;

    	// Push relevant data to the (img, label) list
		img_label_list_.push_back(std::make_pair(img_name, label));
        // LOG(INFO) << img_name;

    }

    // Number of channels (assumed 3 by default)
    this->datum_channels_ = 3;

    // Initialize image counter to 0
    cur_img_ = 0;


    // TODO: Implement mean value subtraction as well
    // Whether or not the mean file is provided in transform_param
    has_mean_file_ = this->transform_param_.has_mean_file();

    // If mean file is present, load it
    if(has_mean_file_){
    	
    	const string& mean_file = this->transform_param_.mean_file();
    	LOG(INFO) << "Loading mean file from " << mean_file;
    	BlobProto blob_proto;
    	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    	Blob<Dtype> data_mean;
    	data_mean.FromProto(blob_proto);
    	LOG(INFO) << "Mean file loaded";

    	// Read config (number of channels in the mean image)
    	this->datum_channels_ = data_mean.channels();

    	// Save mean as OpenCV compatible Mat struct
    	const Dtype* mean_buf = data_mean.cpu_data();
    	mean_img_.create(data_mean.height(), data_mean.width(), CV_32FC3);
    	for(int i = 0; i < data_mean.height(); ++i){
    		for(int j = 0; j < data_mean.width(); ++j){
    			for(int c = 0; c < this->datum_channels_; ++c){
    				mean_img_.at<cv::Vec3f>(i,j)[c] = mean_buf[2 + ((this->datum_channels_+c)*data_mean.height()+i)*data_mean.width()+j];
    			}
    		}
    	}
    	LOG(INFO) << "Mean file converted to OpenCV Mat struct";

    }


    // Reshape data to the desired output size (data_output_size is specified in heatmap_params)
    this->transformed_data_.Reshape(heatmap_data_param.batch_size(), this->datum_channels_, 
    	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    // Reshape the first top blob (data) according to the dimensions 
    // (batch_size x num_channels x data_output_size x data_output_size)
    top[0]->Reshape(heatmap_data_param.batch_size(), this->datum_channels_, 
    	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    // Reshape accordingly the prefetched data
    for (int i = 0; i < this->PREFETCH_COUNT; ++i){
        this->prefetch_[i].data_.Reshape(heatmap_data_param.batch_size(), this->datum_channels_, 
        	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    }
    // Compute the size of the datum (for a single image) output by this layer
    this->datum_size_ = this->datum_channels_ * heatmap_data_param.data_output_size() * heatmap_data_param.data_output_size();


    // Initialize the label

    // Number of channels in the label (number of labels per image)
    // Get the size by examining the first label struct stored in img_list_
    // i.e., the second entity in the first pair in img_list_
    int label_num_channels = img_label_list_[0].second.size() / 2;
    
    // Reshape the second top blob (corresponding to the label)
    top[1]->Reshape(heatmap_data_param.batch_size(), label_num_channels, 
    	heatmap_data_param.label_height(), heatmap_data_param.label_width());
    // Reshape the prefetched label according to the size of the top blob
    for (int i = 0; i < this->PREFETCH_COUNT; ++i){
        this->prefetch_[i].label_.Reshape(heatmap_data_param.batch_size(), label_num_channels, 
        	heatmap_data_param.label_height(), heatmap_data_param.label_width());
    }

    LOG(INFO) << "Output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
    LOG(INFO) << "Output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
    LOG(INFO) << "Number of label channels: " << label_num_channels;
    LOG(INFO) << "Datum channels: " << this->datum_channels_;

}


// Load a batch of data
template<typename Dtype>
void HeatmapDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

    // Start a timer
    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    
    // Get the heatmap data parameters for this layer
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Pointers to blobs' float data
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();

    // Create OpenCV Mat objects for storing various visualizations
    cv::Mat img, img_res, img_mean_vis, img_vis, img_res_vis, mean_img_this;

    
    // If data is to be visualized, create a window
    if (heatmap_data_param.visualize()) {
        cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("mean image", cv::WINDOW_AUTOSIZE);
    }


    // Collect "batch_size" images

    // Label for the current instance
    std::vector<float> cur_label;
    // Image name
    std::string img_name;

    // Loop over images
    for (int idx_img = 0; idx_img < heatmap_data_param.batch_size(); idx_img++)
    {
        // Get image name and class from img_label_list_
        this->GetCurImg(img_name, cur_label);

        // Get number of channels (keypoints) for image label
        int label_num_channels = cur_label.size() / 2;

        // Read in the current image
        std::string img_path = heatmap_data_param.root_img_dir() + img_name;
        DLOG(INFO) << "img: " << img_path;
        img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        // Show visualization of original image, overlaying annotations joined appropriately by lines
        if (heatmap_data_param.visualize())
        {
        	// Clone the original image
            cv::Mat img_annotation_vis = img.clone();
            // Overlay keypoint annotation(s) on the cloned image
            this->VisualizeAnnotations(img_annotation_vis, cur_label.size()/2, cur_label);
            // Display the image with the keypoints overlaid
            cv::imshow("input image", img_annotation_vis);
            // Clone the original image again, this time for visualizing the mean subtracted image
            img_vis = img.clone();
        }

        // Convert from BGR (OpenCV) to RGB (Caffe)
        cv::cvtColor(img, img, CV_BGR2RGB);
        // Convert image to float-32
        img.convertTo(img, CV_32FC3);

        // Subtract mean image, if mean file is specified
        if(this->has_mean_file_){

        	img -= mean_img_;

        	if(heatmap_data_param.visualize()){
        		// img_vis -= this->mean_img_;
        		// img_mean_vis = mean_img_.clone();
        		// cv::cvtColor(img_mean_vis, img_mean_vis, CV_RGB2BGR);
        		// cv::imshow("mean image", img_mean_vis);
        	}

        }

        // Resize input image to output image size
        cv::Size s(heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
        cv::resize(img, img, s);

        // Resulting image dims
        const int channel_size = heatmap_data_param.data_output_size() * heatmap_data_param.data_output_size();
        const int img_size = channel_size * this->datum_channels_;

        // Store image data
        DLOG(INFO) << "storing image";
        for(int c = 0; c < this->datum_channels_; ++c){
        	for(int i = 0; i < heatmap_data_param.data_output_size(); ++i){
        		for(int j = 0; j < heatmap_data_param.data_output_size(); ++j){
        			top_data[idx_img*img_size + c*channel_size + i*heatmap_data_param.data_output_size() + j] = img.at<cv::Vec3f>(i,j)[c];
        		}
        	}
        }

        // Store label (after spreading it using a gaussian)
        DLOG(INFO) << "storing labels";

        // Size of each channel of the label
        const int label_channel_size = heatmap_data_param.label_height() * heatmap_data_param.label_width();
        // Size of each label image (consists of as many channels as there are keypoints)
        const int label_img_size = label_channel_size * label_num_channels;

        // Create a data matrix (cv Mat) to store the label image
        cv::Mat dataMatrix = cv::Mat::zeros(heatmap_data_param.label_height(), heatmap_data_param.label_width(), CV_32FC1);
        // Resize factor for the label
        float label_resize_fact = (float) heatmap_data_param.label_height() / (float) heatmap_data_param.data_output_size();

        // Write the label data to the image (after applying gaussian smoothing to the ground-truth 
        // keypoint location)

        // For each label (keypoint annotation)
        for(int idx_ch = 0; idx_ch < label_num_channels; ++idx_ch){

        	// Compute the coordinates of the keypoint in the label (after accounting for the fact that
        	// the label will (usually) be smaller than the current image, i.e., the current image will
        	// usually be reduced in size due to pooling layers in the network)
        	float x = label_resize_fact * cur_label[2*idx_ch];
        	float y = label_resize_fact * cur_label[2*idx_ch + 1];

        	for(int i = 0; i < heatmap_data_param.label_height(); ++i){
        		for(int j = 0; j < heatmap_data_param.label_width(); ++j){
        			int label_idx = idx_img*label_img_size + idx_ch*label_channel_size + i*heatmap_data_param.label_height() + j;
        			float gaussian = ( 1 / (heatmap_data_param.label_sigma() * sqrt(2 * M_PI))) * exp( -0.5 * ( pow(i-y,2.0) + pow(j-x,2.0)) * pow(1/heatmap_data_param.label_sigma(), 2) );
        			// (???) Don't understand why Tomas Pfister did this!
        			gaussian = 4 * gaussian;
        			top_label[label_idx] = gaussian;

        			// Tomas Pfister did this too. Maybe for visualization/debugging? (???)
        			if(idx_ch == 0){
        				dataMatrix.at<float>((int) j, (int) i) = gaussian;
        			}
        		}
        	}

        }

        DLOG(INFO) << "Next image";

        // Move to the next image
        this->AdvanceCurImg();

        // Loop forever, if visualizing
        if (heatmap_data_param.visualize()){
            cv::waitKey(0);
        }


    } // Loop over all images

    // Time stats
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}


// Get the current image and the label
template<typename Dtype>
void HeatmapDataLayer<Dtype>::GetCurImg(string& img_name, std::vector<float>& img_label){

	img_name = img_label_list_[cur_img_].first;
	img_label = img_label_list_[cur_img_].second;

}


// Move to the next image in the list
// TODO: Specify other strategies (randomization a possibility ?)
template<typename Dtype>
void HeatmapDataLayer<Dtype>::AdvanceCurImg()
{
	if(cur_img_ < img_label_list_.size() - 1){
		cur_img_++;
	}
	else{
		cur_img_ = 0;
	}

}


// Visualize annotations
template<typename T>
void HeatmapDataLayer<T>::VisualizeAnnotations(cv::Mat img_annotation_vis, int label_num_channels, std::vector<float>& img_class)
{
    // Color for the keypoint
    const static cv::Scalar colors[] = {CV_RGB(0, 0, 255)};

    // Number of keypoint coordinates

    // Draw keypoint on the image
    cv::Point centers[(int) label_num_channels];
    for(int i = 0; i < label_num_channels; i += 2){
        int coordInd = int(i/2);
        centers[coordInd] = cv::Point(img_class[i], img_class[i+1]);
        cv::circle(img_annotation_vis, centers[coordInd], 1, colors[coordInd], 3);
    }

}


template <typename Dtype>
float HeatmapDataLayer<Dtype>::Uniform(const float min, const float max) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

INSTANTIATE_CLASS(HeatmapDataLayer);
REGISTER_LAYER_CLASS(HeatmapData);

} // namespace caffe


#endif // USE_OPENCV
