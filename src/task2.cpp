#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <tuple>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tuple;
using std::make_tuple;
using std::tie;


using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;
typedef Matrix<std::tuple<uint, uint, uint>> Image;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);

    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

#define PI 3.14159265
#define pi_seg_num 8
#define vert_seg_num 4
#define hor_seg_num 8
#define L 0.35
void get_features(Matrix<float> &Intens, vector<float> &one_image_features){
    int src_h = Intens.n_rows, src_w = Intens.n_cols;
    int h = src_h - 2, w = src_w - 2;
        Matrix <float>  Direct(h, w), Abs(h, w);
        for (int i = 1; i < src_h - 1; ++i)
            for (int j = 1; j < src_w - 1; ++j){
                float y = Intens(i + 1, j) - Intens(i - 1, j);
                float x = Intens(i, j + 1) - Intens(i, j - 1);
                Direct(i - 1, j - 1) = atan2(y, x);
                Abs(i - 1, j - 1) = sqrt(x * x + y * y);
            }
        float d_pi = 2 * PI/pi_seg_num, d_w = w / hor_seg_num, d_h = h / vert_seg_num;
        int first_hor_pix, first_vert_pix, last_hor_pix = -1, last_vert_pix = -1;
        for (int ind_h = 0; ind_h < vert_seg_num; ++ind_h) {
            first_vert_pix = last_vert_pix + 1;
            if ( ind_h == vert_seg_num - 1)
                last_vert_pix = h - 1;
            else
                last_vert_pix = round((ind_h + 1) * d_h);
            last_hor_pix = -1;
            for (int ind_w = 0; ind_w < hor_seg_num; ++ind_w){
                first_hor_pix = last_hor_pix + 1;
                if ( ind_w == hor_seg_num - 1)
                    last_hor_pix = w - 1;
                else
                    last_hor_pix = round((ind_w + 1) * d_w);
                vector<float> histogram;
                histogram.resize(pi_seg_num);
                for (int i = 0; i < pi_seg_num; ++i)
                    histogram[i] = 0;
                Matrix<float> Sub_direct = Direct.submatrix(first_vert_pix, first_hor_pix, last_vert_pix - first_vert_pix + 1, last_hor_pix - first_hor_pix + 1);
                Matrix<float> Sub_abs = Abs.submatrix(first_vert_pix, first_hor_pix,  last_vert_pix - first_vert_pix + 1, last_hor_pix - first_hor_pix + 1);
                uint sub_h = Sub_abs.n_rows, sub_w = Sub_abs.n_cols;
                for (uint i = 0; i < sub_h; ++i)
                    for (uint j = 0; j < sub_w; ++ j){
                        float carry = -PI, num = 0;
                        while ((carry + (num + 1) * d_pi < Sub_direct(i, j)) && (num < pi_seg_num))
                            ++num;
                        histogram[num] += Sub_abs(i, j);
                    }
                float norm = 0;
                for (int i = 0; i < pi_seg_num; ++i)
                    norm += histogram[i] * histogram[i];
                norm = sqrt(norm);
                if (abs(norm) > 0.00000001)
                    for (int i = 0; i < pi_seg_num; ++i)
                        histogram[i] /= norm;
                //add unlinear SVM kernel
                vector<float> unlinear;
                for (int i = 0; i < pi_seg_num; ++i){
                    float carry_cos = 0, carry_sin = 0, x = histogram[i];
                    if (x > 0.00000001)
                    for (int n = -1; n <=1; ++n) {
                        carry_cos += cos(-log(x) * n * L) * sqrt(x * 2 / (exp(PI * n * L) + exp(-PI * n * L)));
                        carry_sin += sin(-log(x) * n * L) * sqrt(x * 2 / (exp(PI * n * L) + exp(-PI * n * L)));
                    }
                    unlinear.push_back(carry_cos);
                    unlinear.push_back(carry_sin);
                }


                one_image_features.insert(one_image_features.end(), unlinear.begin(), unlinear.end());
            }
        }
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        vector<float> one_image_features, one_image_features0;
        BMP *im = data_set[image_idx].first;
        int src_h = im->TellWidth(), src_w = im->TellHeight();
        Matrix<float> Intens(src_h, src_w);
        Image img(src_h, src_w);
        for (int i = 0; i < src_h; ++i)
            for (int j = 0; j < src_w; ++j){
                RGBApixel pixel = im->GetPixel(i, j);
                Intens(i, j) = 0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue;
                img(i, j) = make_tuple(pixel.Red, pixel.Green, pixel.Blue);
            }
        // add descriptor pyramid
        get_features(Intens, one_image_features0);
        one_image_features.insert(one_image_features.end(), one_image_features0.begin(), one_image_features0.end());
        one_image_features0.clear();
        int middle_i = src_h / 2, middle_j = src_w / 2;

        Matrix<float> Int1 = Intens.submatrix(0, 0, middle_i, middle_j);
        get_features(Int1, one_image_features0);
        one_image_features.insert(one_image_features.end(), one_image_features0.begin(), one_image_features0.end());
        one_image_features0.clear();

        Matrix<float> Int2 = Intens.submatrix(middle_i, 0, src_h - middle_i, middle_j);
        get_features(Int2, one_image_features0);
        one_image_features.insert(one_image_features.end(), one_image_features0.begin(), one_image_features0.end());
        one_image_features0.clear();

        Matrix<float> Int3 = Intens.submatrix(0, middle_j, middle_i, src_w - middle_j);
        get_features(Int3, one_image_features0);
        one_image_features.insert(one_image_features.end(), one_image_features0.begin(), one_image_features0.end());
        one_image_features0.clear();

        Matrix<float> Int4 = Intens.submatrix(middle_i, middle_j, src_h - middle_i,  src_w - middle_j );
        get_features(Int4, one_image_features0);
        one_image_features.insert(one_image_features.end(), one_image_features0.begin(), one_image_features0.end());
        one_image_features0.clear();


        //add colors_features
        #define div 8
        float d_w = src_w / div, d_h = src_h / div;
        int first_hor_pix, first_vert_pix, last_hor_pix = -1, last_vert_pix = -1;
        for (int ind_h = 0; ind_h < div; ++ind_h) {
            first_vert_pix = last_vert_pix + 1;
            if ( ind_h == div - 1)
                last_vert_pix = src_h - 1;
            else
                last_vert_pix = round((ind_h + 1) * d_h);
            last_hor_pix = -1;
            for (int ind_w = 0; ind_w < div; ++ind_w){
                first_hor_pix = last_hor_pix + 1;
                if ( ind_w == div - 1)
                    last_hor_pix = src_w - 1;
                else
                    last_hor_pix = round((ind_w + 1) * d_w);
                Image sub_img  = img.submatrix(first_vert_pix, first_hor_pix, last_vert_pix - first_vert_pix + 1, last_hor_pix - first_hor_pix + 1);
                uint sub_h = sub_img.n_rows, sub_w = sub_img.n_cols;
                float sumr = 0, sumg = 0, sumb = 0;
                for (uint i = 0; i < sub_h; ++i)
                    for (uint j = 0; j < sub_w; ++j){
                        int r, g, b;
                        tie(r, g, b) = sub_img(i, j);
                        sumr += (1.f * r)/255;
                        sumg += (1.f * g)/255;
                        sumb += (1.f * b)/255;
                    }
                sumr /= sub_h; sumr/= sub_w;
                sumg /= sub_h; sumg/= sub_w;
                sumb /= sub_h; sumb/= sub_w;
                one_image_features.push_back(sumr);
                one_image_features.push_back(sumg);
                one_image_features.push_back(sumb);
            }
        }

        features->push_back(make_pair(one_image_features,  data_set[image_idx].second));
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");

        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
