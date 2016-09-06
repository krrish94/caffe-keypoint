/*
 Copyright 2015 Tomas Pfister
 Adapted by J. Krishna Murthy for caffe-keypoint
*/

// Include this layer only if Caffe has been compiled with OpenCV

#ifdef USE_OPENCV

#ifndef CAFFE_HEATMAP_HPP_
#define CAFFE_HEATMAP_HPP_

#include "caffe/layer.hpp"
#include <vector>
#include <boost/timer/timer.hpp>
#include <opencv2/core/core.hpp>

#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{


// HeatmapDataLayer class
template<typename Dtype>
class HeatmapDataLayer: public BasePrefetchingDataLayer<Dtype>
{

public:

    // Constructor
    explicit HeatmapDataLayer(const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype>(param) {}
    // Destructor
    virtual ~HeatmapDataLayer();
    // Performs data specific set-up
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    // Returns the name of the layer
    virtual inline const char* type() const { return "HeatmapData"; }

    // Returns the number of top blobs produced by the layer
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    // Returns the number of bottom blobs produced by the layer (data, label)
    virtual inline int ExactNumTopBlobs() const { return 2; }


protected:

    // Loads a mini batch
    virtual void load_batch(Batch<Dtype>* batch);

    // Filename of current image
    inline void GetCurImg(string& img_name, std::vector<float>& img_label);

    // Advance to the next image
    inline void AdvanceCurImg();

    // Visualise keypoint annotations
    inline void VisualizeAnnotations(cv::Mat img_annotation_vis, int numChannels, std::vector<float>& cur_label);

    // Random number generator
    inline float Uniform(const float min, const float max);

    
    // Global vars

    // (???) Something to do with Caffe's RNG
    shared_ptr<Caffe::RNG> rng_data_;
    shared_ptr<Caffe::RNG> prefetch_rng_;
    // Lines (in the input file)
    vector<std::pair<std::string, int> > lines_;
    // Id of the current line
    int lines_id_;
    // Number of channels in the blob
    int datum_channels_;
    // Height of the blob
    int datum_height_;
    // Width of the blob
    int datum_width_;
    // Size of the blob (num_channels * height * width)
    int datum_size_;

    // Number of means
    int num_means_;
    // Holds mean image
    vector<cv::Mat> mean_img_;
    // Whether or not the mean is to be subtracted
    bool sub_mean_;
    // Base directory containing the images
    string root_img_dir_;
    // Index of the current image
    int cur_img_;
    // Current image for each class
    vector<int> img_idx_map_;

    // Array of lists: one list of image names per class
    vector< vector< pair<string, pair<vector<float>, pair<vector<float>, int> > > > > img_list_;

    // Vector of (image, label) pairs
    vector< pair<string, vector<float> > > img_label_list_;
};

}

#endif // CAFFE_HEATMAP_HPP_

#endif // USE_OPENCV
