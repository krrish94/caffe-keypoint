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
#include "caffe/data_layers.hpp"
#include "caffe/layers/heatmap_data.hpp"
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


// Performs layer-specific setup
template<typename Dtype>
void HeatmapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    // Get heatmap data parameters (parameters passed to this layer in the prototxt)
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Shortcuts

    // Batch size
    const int batchsize = heatmap_data_param.batch_size();
    // Height of each label
    const int label_width = heatmap_data_param.label_width();
    // Width of each label
    const int label_height = heatmap_data_param.label_height();
    // Output size
    const int outsize = heatmap_data_param.outsize();
    // Batch size for each label (number of labels)
    const int label_batchsize = batchsize;
    // Base directory containing the images
    root_img_dir_ = heatmap_data_param.root_img_dir();


    // Seed the pseudo rng
    const unsigned int rng_seed = caffe_rng_rand();
    srand(rng_seed);

    // Path to file containing ground-truth annotations
    std::string gt_path = heatmap_data_param.source();
    LOG(INFO) << "Loading annotations from " << gt_path;

    // Open the input file stream
    std::ifstream infile(gt_path.c_str());
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

    }

    // Initialize image counter to 0
    cur_img_ = 0;

    // Mean image subtraction (yet to be implemented)

    // If mean file is not present, you do not need to subtract it
    if(!heatmap_data_param.has_mean_file()){
    	// Assume input images are RGB
    	this->datum_channels_ = 3;
    	sub_mean_ = false;
    }



    // // Mean image subtraction
    // if (!heatmap_data_param.has_meanfile())
    // {
    //     // If image mean isn't specified, assume input images are RGB (3 channels)
    //     this->datum_channels_ = 3;
    //     // Mean subtraction is not to be performed
    //     sub_mean_ = false;
    // } else {
        
    //     // Implementation of per-video mean removal

    //     // Get path of mean file
    //     sub_mean_ = true;
    //     string mean_path = heatmap_data_param.meanfile();

    //     LOG(INFO) << "Loading mean file from " << mean_path;
    //     BlobProto blob_proto, blob_proto2;
    //     Blob<Dtype> data_mean;
    //     ReadProtoFromBinaryFile(mean_path.c_str(), &blob_proto);
    //     data_mean.FromProto(blob_proto);
    //     LOG(INFO) << "mean file loaded";

    //     // Read config (number of channels in the mean image)
    //     this->datum_channels_ = data_mean.channels();
    //     // Number of means (one mean is assumed per video)
    //     num_means_ = data_mean.num();
    //     LOG(INFO) << "num_means: " << num_means_;

    //     // Copy the per-video mean images to an array of OpenCV structures
    //     const Dtype* mean_buf = data_mean.cpu_data();

    //     // Extract means from beginning of proto file
    //     const int mean_height = data_mean.height();
    //     const int mean_width = data_mean.width();
    //     int mean_heights[num_means_];
    //     int mean_widths[num_means_];

    //     // Offset in memory to mean images
    //     const int meanOffset = 2 * (num_means_);
    //     for (int n = 0; n < num_means_; n++)
    //     {
    //         mean_heights[n] = mean_buf[2 * n];
    //         mean_widths[n] = mean_buf[2 * n + 1];
    //     }

    //     // Save means as OpenCV-compatible files
    //     for (int n = 0; n < num_means_; n++)
    //     {
    //         cv::Mat mean_img_tmp_;
    //         mean_img_tmp_.create(mean_heights[n], mean_widths[n], CV_32FC3);
    //         mean_img_.push_back(mean_img_tmp_);
    //         LOG(INFO) << "per-video mean file array created: " << n << ": " << mean_heights[n] << "x" << mean_widths[n] << " (" << size << ")";
    //     }

    //     LOG(INFO) << "mean: " << mean_height << "x" << mean_width << " (" << size << ")";

    //     for (int n = 0; n < num_means_; n++)
    //     {
    //         for (int i = 0; i < mean_heights[n]; i++)
    //         {
    //             for (int j = 0; j < mean_widths[n]; j++)
    //             {
    //                 for (int c = 0; c < this->datum_channels_; c++)
    //                 {
    //                     mean_img_[n].at<cv::Vec3f>(i, j)[c] = mean_buf[meanOffset + ((n * this->datum_channels_ + c) * mean_height + i) * mean_width + j]; //[c * mean_height * mean_width + i * mean_width + j];
    //                 }
    //             }
    //         }
    //     }

    //     LOG(INFO) << "mean file converted to OpenCV structures";
    // }


    // Reshape data to the desired output size (outsize is specified in heatmap_params)
    this->transformed_data_.Reshape(heatmap_data_param.batch_size(), this->datum_channels_, 
    	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    // Reshape the first top blob (data) according to the dimensions 
    // (batch_size x num_channels x data_output_size x data_output_size)
    top[0]->Reshape(batchsize, this->datum_channels_, 
    	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    // Reshape accordingly the prefetched data
    for (int i = 0; i < this->PREFETCH_COUNT; ++i){
        this->prefetch_[i].data_.Reshape(heatmap_data_param.batch_size(), this->datum_channels_, 
        	heatmap_data_param.data_output_size(), heatmap_data_param.data_output_size());
    }
    // Compute the size of the datum (for a single image) output by this layer
    this->datum_size_ = this->datum_channels_ * outsize * outsize;


    // Initialize the label

    // Number of channels in the label (number of labels per image)
    int label_num_channels;
    // Get the number of channels in the label
    if (!sample_per_cluster_)
        label_num_channels = img_label_list_[0].second.first.size();
    else
        label_num_channels = img_list_[0][0].second.first.size();
    label_num_channels /= 2;
    // Reshape the top blob corresponding to the label
    top[1]->Reshape(label_batchsize, label_num_channels, label_height, label_width);
    // Reshape the prefetched label according to the size of the top blob
    for (int i = 0; i < this->PREFETCH_COUNT; ++i){
        this->prefetch_[i].label_.Reshape(label_batchsize, label_num_channels, label_height, label_width);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
    LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
    LOG(INFO) << "number of label channels: " << label_num_channels;
    LOG(INFO) << "datum channels: " << this->datum_channels_;

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
    cv::Mat img, img_res, img_annotation_vis, img_mean_vis, img_vis, img_res_vis, mean_img_this, seg, segTmp;

    // Shortcuts to params

    // Whether or not to visualize data
    const bool visualise = this->layer_param_.visualise();
    // (???)
    const Dtype scale = heatmap_data_param.scale();
    // Data batch size
    const int batchsize = heatmap_data_param.batchsize();
    // Height and width of each label
    const int label_height = heatmap_data_param.label_height();
    const int label_width = heatmap_data_param.label_width();
    // Maximum angle by which random rotations are allowed
    const float angle_max = heatmap_data_param.angle_max();
    // Flag that specifies not to flip image first
    const bool dont_flip_first = heatmap_data_param.dont_flip_first();
    // Flag to specify whether or not to flip joint labels
    const bool flip_joint_labels = heatmap_data_param.flip_joint_labels();
    // Factor by which labels have to be multiplied
    const int multfact = heatmap_data_param.multfact();
    // (???)
    const bool segmentation = heatmap_data_param.segmentation();
    // Crop size (for randomly generated crops)
    const int size = heatmap_data_param.cropsize();
    // Output size of heatmap data blob
    const int outsize = heatmap_data_param.outsize();
    // Number of augmentations (This is initially set to 1, for the actual image. As augmentations 
    // are made (cropping, jittering, etc.), this is incremented)
    const int num_aug = 1;
    // Resize factor (to resize the random crops to the required output (data) blob size)
    const float resizeFact = (float)outsize / (float)size;
    // Set random crops to false
    const bool random_crop = heatmap_data_param.random_crop();

    
    // Shortcuts to global vars

    // Whether or not to subtract mean image
    const bool sub_mean = this->sub_mean_;
    // Number of channels in the blob
    const int channels = this->datum_channels_;

    // What coordinates should we flip when mirroring images?
    // For pose estimation with joints assumes i=0,1 are for head, and i=2,3 left wrist, i=4,5 right wrist etc
    //     in which case dont_flip_first should be set to true.
    int flip_start_ind;
    if (dont_flip_first) flip_start_ind = 2;
    else flip_start_ind = 0;

    
    // If data is to be visualized, create corresponding windows
    if (visualise)
    {
        // Original image
        cv::namedWindow("original image", cv::WINDOW_AUTOSIZE);
        // Random crop
        // cv::namedWindow("cropped image", cv::WINDOW_AUTOSIZE);
        // Resized image (???)
        // cv::namedWindow("interim resize image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("resulting image", cv::WINDOW_AUTOSIZE);
    }


    // Collect "batchsize" images

    // Label and crop info for the current instance
    std::vector<float> cur_label, cur_cropinfo;
    // Image name
    std::string img_name;
    // Class of the current image
    int cur_class;

    // Loop over non-augmented images
    for (int idx_img = 0; idx_img < batchsize; idx_img++)
    {
        // Get image name and class
        this->GetCurImg(img_name, cur_label, cur_cropinfo, cur_class);

        // Get number of channels for image label
        int label_num_channels = cur_label.size();

        std::string img_path = this->root_img_dir_ + img_name;
        DLOG(INFO) << "img: " << img_path;
        img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        // Show visualization of original image, overlaying annotations joined appropriately by lines
        if (visualise)
        {
            img_annotation_vis = img.clone();
            this->VisualiseAnnotations(img_annotation_vis, label_num_channels, cur_label, multfact);
            cv::imshow("original image", img_annotation_vis);
        }

        // If the parameter 'segmentation' is set to true, this looks for a directory named 'segs'
        // in the root image directory.
        if (segmentation)
        {
            std::string seg_path = this->root_img_dir_ + "segs/" + img_name;
            std::ifstream ifile(seg_path.c_str());

            // Skip this file if segmentation doesn't exist
            if (!ifile.good())
            {
                LOG(INFO) << "file " << seg_path << " does not exist!";
                idx_img--;
                this->AdvanceCurImg();
                continue;
            }
            ifile.close();

            // Load the segmentated image (in grayscale)
            seg = cv::imread(seg_path, CV_LOAD_IMAGE_GRAYSCALE);
        }

        // Width and height of the image
        int width = img.cols;
        int height = img.rows;
        // Subtract crop size from width and height (to get a list of indices from where
        // we can choose any pair at random, without having to worry if it will lie entirely
        // inside the iamge)
        int x_border = width - size;
        int y_border = height - size;

        // Convert from BGR (OpenCV) to RGB (Caffe)
        cv::cvtColor(img, img, CV_BGR2RGB);

        // To float-32
        img.convertTo(img, CV_32FC3);

        // If segmentation is set to true, threshold the segmented image, and copy it to the variable img
        if (segmentation)
        {
            segTmp = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
            int threshold = 40;
            seg = (seg > threshold);
            segTmp.copyTo(img, seg);
        }

        if (visualise){
            img_vis = img.clone();
        }

        // Subtract per-video mean, if used
        int meanInd = 0;
        if (sub_mean)
        {
            std::string delimiter = "/";
            std::string img_name_subdirImg = img_name.substr(img_name.find(delimiter) + 1, img_name.length());
            std::string meanIndStr = img_name_subdirImg.substr(0, img_name_subdirImg.find(delimiter));
            meanInd = atoi(meanIndStr.c_str()) - 1;

            // subtract the cropped mean
            mean_img_this = this->mean_img_[meanInd].clone();

            DLOG(INFO) << "Image size: " << width << "x" << height;
            DLOG(INFO) << "Crop info: " << cur_cropinfo[0] << " " <<  cur_cropinfo[1] << " " << cur_cropinfo[2] << " " << cur_cropinfo[3] << " " << cur_cropinfo[4];
            DLOG(INFO) << "Crop info after: " << cur_cropinfo[0] << " " <<  cur_cropinfo[1] << " " << cur_cropinfo[2] << " " << cur_cropinfo[3] << " " << cur_cropinfo[4];
            DLOG(INFO) << "Mean image size: " << mean_img_this.cols << "x" << mean_img_this.rows;
            DLOG(INFO) << "Cropping: " << cur_cropinfo[0] - 1 << " " << cur_cropinfo[2] - 1 << " " << width << " " << height;

            // crop and resize mean image
            cv::Rect crop(cur_cropinfo[0] - 1, cur_cropinfo[2] - 1, cur_cropinfo[1] - cur_cropinfo[0], cur_cropinfo[3] - cur_cropinfo[2]);
            mean_img_this = mean_img_this(crop);
            cv::resize(mean_img_this, mean_img_this, img.size());

            DLOG(INFO) << "Cropped mean image.";

            img -= mean_img_this;

            DLOG(INFO) << "Subtracted mean image.";

            // if (visualise)
            // {
            //     img_vis -= mean_img_this;
            //     img_mean_vis = mean_img_this.clone() / 255;
            //     cv::cvtColor(img_mean_vis, img_mean_vis, CV_RGB2BGR);
            //     cv::imshow("mean image", img_mean_vis);
            // }
        }

        // Pad images that aren't wide enough to support cropping
        if (x_border < 0)
        {
            DLOG(INFO) << "padding " << img_path << " -- not wide enough.";

            cv::copyMakeBorder(img, img, 0, 0, 0, -x_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            width = img.cols;
            x_border = width - size;

            // add border offset to joints
            for (int i = 0; i < label_num_channels; i += 2)
                cur_label[i] = cur_label[i] + x_border;

            DLOG(INFO) << "new width: " << width << "   x_border: " << x_border;
            if (visualise)
            {
                img_vis = img.clone();
                cv::copyMakeBorder(img_vis, img_vis, 0, 0, 0, -x_border, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            }
        }

        DLOG(INFO) << "Entering jitter loop.";

        // Loop over the jittered versions
        for (int idx_aug = 0; idx_aug < num_aug; idx_aug++)
        {
            // Augmented image index in the resulting batch
            const int idx_img_aug = idx_img * num_aug + idx_aug;
            std::vector<float> cur_label_aug = cur_label;

            if (random_crop)
            {
                // random sampling
                DLOG(INFO) << "random crop sampling";

                // horizontal flip
                if (rand() % 2)
                {
                    // flip
                    cv::flip(img, img, 1);

                    if (visualise)
                        cv::flip(img_vis, img_vis, 1);

                    // "flip" annotation coordinates
                    for (int i = 0; i < label_num_channels; i += 2)
                        cur_label_aug[i] = (float)width / (float)multfact - cur_label_aug[i];

                    // "flip" annotation joint numbers
                    // assumes i=0,1 are for head, and i=2,3 left wrist, i=4,5 right wrist etc
                    // where coordinates are (x,y)
                    if (flip_joint_labels)
                    {
                        float tmp_x, tmp_y;
                        for (int i = flip_start_ind; i < label_num_channels; i += 4)
                        {
                            CHECK_LT(i + 3, label_num_channels);
                            tmp_x = cur_label_aug[i];
                            tmp_y = cur_label_aug[i + 1];
                            cur_label_aug[i] = cur_label_aug[i + 2];
                            cur_label_aug[i + 1] = cur_label_aug[i + 3];
                            cur_label_aug[i + 2] = tmp_x;
                            cur_label_aug[i + 3] = tmp_y;
                        }
                    }
                }

                // left-top coordinates of the crop [0;x_border] x [0;y_border]
                int x0 = 0, y0 = 0;
                x0 = rand() % (x_border + 1);
                y0 = rand() % (y_border + 1);

                // do crop
                cv::Rect crop(x0, y0, size, size);

                // NOTE: no full copy performed, so the original image buffer is affected by the transformations below
                cv::Mat img_crop(img, crop);

                // "crop" annotations
                for (int i = 0; i < label_num_channels; i += 2)
                {
                    cur_label_aug[i] -= (float)x0 / (float) multfact;
                    cur_label_aug[i + 1] -= (float)y0 / (float) multfact;
                }

                // // show image
                // if (visualise)
                // {
                //     DLOG(INFO) << "cropped image";
                //     cv::Mat img_vis_crop(img_vis, crop);
                //     cv::Mat img_res_vis = img_vis_crop / 255;
                //     cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                //     this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                //     cv::imshow("cropped image", img_res_vis);
                // }

                // rotations
                float angle = Uniform(-angle_max, angle_max);
                cv::Mat M = this->RotateImage(img_crop, angle);

                // also flip & rotate labels
                for (int i = 0; i < label_num_channels; i += 2)
                {
                    // convert to image space
                    float x = cur_label_aug[i] * (float) multfact;
                    float y = cur_label_aug[i + 1] * (float) multfact;

                    // rotate
                    cur_label_aug[i] = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2);
                    cur_label_aug[i + 1] = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2);

                    // convert back to joint space
                    cur_label_aug[i] /= (float) multfact;
                    cur_label_aug[i + 1] /= (float) multfact;
                }

                img_res = img_crop;
            } else {
                // Determinsitic sampling
                DLOG(INFO) << "deterministic crop sampling (centre)";

                // Centre crop
                const int y0 = y_border / 2;
                const int x0 = x_border / 2;

                DLOG(INFO) << "cropping image from " << x0 << "x" << y0;

                // Crop
                cv::Rect crop(x0, y0, size, size);
                cv::Mat img_crop(img, crop);

                DLOG(INFO) << "cropping annotations.";

                // "Crop" annotations
                for (int i = 0; i < label_num_channels; i += 2)
                {
                    cur_label_aug[i] -= (float)x0 / (float) multfact;
                    cur_label_aug[i + 1] -= (float)y0 / (float) multfact;
                }

                // if (visualise)
                // {
                //     cv::Mat img_vis_crop(img_vis, crop);
                //     cv::Mat img_res_vis = img_vis_crop.clone() / 255;
                //     cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
                //     this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
                //     cv::imshow("cropped image", img_res_vis);
                // }
                img_res = img_crop;
            }

            // // show image
            // if (visualise)
            // {
            //     cv::Mat img_res_vis = img_res / 255;
            //     cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
            //     this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
            //     cv::imshow("interim resize image", img_res_vis);
            // }

            DLOG(INFO) << "Resizing output image.";

            // resize to output image size
            cv::Size s(outsize, outsize);
            cv::resize(img_res, img_res, s);

            // "resize" annotations
            for (int i = 0; i < label_num_channels; i++){
                cur_label_aug[i] *= resizeFact;
            }

            // // show image
            // if (visualise)
            // {
            //     cv::Mat img_res_vis = img_res / 255;
            //     cv::cvtColor(img_res_vis, img_res_vis, CV_RGB2BGR);
            //     this->VisualiseAnnotations(img_res_vis, label_num_channels, cur_label_aug, multfact);
            //     cv::imshow("resulting image", img_res_vis);
            // }

            // // Show image
            // if (visualise && sub_mean)
            // {
            //     cv::Mat img_res_meansub_vis = img_res / 255;
            //     cv::cvtColor(img_res_meansub_vis, img_res_meansub_vis, CV_RGB2BGR);
            //     cv::imshow("mean-removed image", img_res_meansub_vis);
            // }

            // Multiply by scale
            if (scale != 1.0){
                img_res *= scale;
            }

            // Resulting image dims
            const int channel_size = outsize * outsize;
            const int img_size = channel_size * channels;

            // Store image data
            DLOG(INFO) << "storing image";
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < outsize; i++)
                {
                    for (int j = 0; j < outsize; j++)
                    {
                        top_data[idx_img_aug * img_size + c * channel_size + i * outsize + j] = img_res.at<cv::Vec3f>(i, j)[c];
                    }
                }
            }

            // Store label as gaussian
            
            DLOG(INFO) << "storing labels";
            // Size of each channel of the label
            const int label_channel_size = label_height * label_width;
            // Size of each image of the label
            const int label_img_size = label_channel_size * label_num_channels / 2;

            // Create a data matrix to store label
            cv::Mat dataMatrix = cv::Mat::zeros(label_height, label_width, CV_32FC1);
            float label_resize_fact = (float) label_height / (float) outsize;
            // 'Spread' of the gaussian around the ground-truth keypoint
            float sigma = 1.5;

            // Write the label data to the image (after applying gaussian smoothing to the ground-truth 
            // keypoint location)
            for (int idx_ch = 0; idx_ch < label_num_channels / 2; idx_ch++)
            {
                float x = label_resize_fact * cur_label_aug[2 * idx_ch] * multfact;
                float y = label_resize_fact * cur_label_aug[2 * idx_ch + 1] * multfact;
                for (int i = 0; i < label_height; i++)
                {
                    for (int j = 0; j < label_width; j++)
                    {
                        int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                        float gaussian = ( 1 / ( sigma * sqrt(2 * M_PI) ) ) * exp( -0.5 * ( pow(i - y, 2.0) + pow(j - x, 2.0) ) * pow(1 / sigma, 2.0) );
                        gaussian = 4 * gaussian;
                        top_label[label_idx] = gaussian;

                        if (idx_ch == 0)
                            dataMatrix.at<float>((int)j, (int)i) = gaussian;
                    }
                }
            }

        } // jittered versions loop

        DLOG(INFO) << "next image";

        // move to the next image
        this->AdvanceCurImg();

        if (visualise)
            cv::waitKey(0);


    } // original image loop

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}


// Get the current image
template<typename Dtype>
void HeatmapDataLayer<Dtype>::GetCurImg(string& img_name, std::vector<float>& img_label, std::vector<float>& crop_info, int& img_class)
{

    if (!sample_per_cluster_)
    {
        img_name = img_label_list_[cur_img_].first;
        img_label = img_label_list_[cur_img_].second.first;
        crop_info = img_label_list_[cur_img_].second.second.first;
        img_class = img_label_list_[cur_img_].second.second.second;
    }
    else
    {
        img_class = cur_class_;
        img_name = img_list_[img_class][cur_class_img_[img_class]].first;
        img_label = img_list_[img_class][cur_class_img_[img_class]].second.first;
        crop_info = img_list_[img_class][cur_class_img_[img_class]].second.second.first;
    }
}


// Move to the next image in the list
template<typename Dtype>
void HeatmapDataLayer<Dtype>::AdvanceCurImg()
{
    if (!sample_per_cluster_)
    {
        if (cur_img_ < img_label_list_.size() - 1)
            cur_img_++;
        else
            cur_img_ = 0;
    }
    else
    {
        const int num_classes = img_list_.size();

        if (cur_class_img_[cur_class_] < img_list_[cur_class_].size() - 1)
            cur_class_img_[cur_class_]++;
        else
            cur_class_img_[cur_class_] = 0;

        // move to the next class
        if (cur_class_ < num_classes - 1)
            cur_class_++;
        else
            cur_class_ = 0;
    }

}


// Visualize annotations (modified by KM)
template<typename T>
void HeatmapDataLayer<T>::VisualiseAnnotations(cv::Mat img_annotation_vis, int label_num_channels, std::vector<float>& img_class, int multfact)
{
    // Color for the keypoint
    const static cv::Scalar colors[] = {CV_RGB(0, 0, 255)};

    // Number of keypoint coordinates
    int numCoordinates = int(label_num_channels / 2);

    // Draw keypoint on the image
    cv::Point centers[numCoordinates];
    for(int i = 0; i < label_num_channels; i += 2){
        int coordInd = int(i/2);
        centers[coordInd] = cv::Point(img_class[i] * multfact, img_class[i+1] * multfact);
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

template <typename Dtype>
cv::Mat HeatmapDataLayer<Dtype>::RotateImage(cv::Mat src, float rotation_angle)
{
    cv::Mat rot_mat(2, 3, CV_32FC1);
    cv::Point center = cv::Point(src.cols / 2, src.rows / 2);
    double scale = 1;

    // Get the rotation matrix with the specifications above
    rot_mat = cv::getRotationMatrix2D(center, rotation_angle, scale);

    // Rotate the warped image
    cv::warpAffine(src, src, rot_mat, src.size());

    return rot_mat;
}

INSTANTIATE_CLASS(HeatmapDataLayer);
REGISTER_LAYER_CLASS(HeatmapData);

} // namespace caffe


#endif // USE_OPENCV
