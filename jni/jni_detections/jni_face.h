/*
 * detector.h using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once
#include <jni_common/jni_fileutils.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_saver/save_jpeg.h>
#include <glog/logging.h>
#include <jni.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <dlib/dnn.h>
#include <dlib/dnn/loss.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/serialize.h>
#include <dlib/image_io.h>
#include <dlib/svm_threaded.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <stdlib.h>
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;
//
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;
//
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;
//
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
//
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;
using sample_type = dlib::matrix<float, 0, 1>;
using ovo_trainer = dlib::one_vs_one_trainer<dlib::any_trainer<sample_type>>;
using rbf_kernel = dlib::radial_basis_kernel<sample_type>;


/******* FACE RECOGNITION **********/
class DlibFaceRecognition {
public:
  DlibFaceRecognition(const std::string& modelPath)
      : mModelPath(modelPath) {
    init();
  }

  inline void init() {
    LOG(INFO) << "Model Path: " << mModelPath;
    if (jniutils::fileExists(mModelPath)) {
      try {
        dlib::deserialize(mModelPath) >> net;
      } catch( std::exception &e) {
        LOG(INFO) << " exception " << e.what() << std::endl;
        exit(0);
      }
    } else {
      LOG(INFO) << "Not exist " << mModelPath;
    }
    LOG(INFO) << "successfully deserialized " << mModelPath;
  }

  std::vector<dlib::matrix<float,0,1>> getFaceDescriptors(
    dlib::array<dlib::matrix<dlib::rgb_pixel>>& faces) {
    std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
    return face_descriptors;
  }

  void populateData(const std::string &dirName, std::vector<sample_type>& samples,
    std::vector<double>& labels) {
    DIR* dirp = opendir(dirName.c_str());
    if(!dirp) {
      LOG(INFO) << "top level dir doesn't exist " << dirName;
      return;
    }
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::ostringstream ss;
        ss << dirName << "/" << dp->d_name;
        string innerDirName = ss.str();
        DIR* innerDirp = opendir(innerDirName.c_str());
        if(!innerDirp) {
          LOG(INFO) << "skipping " << innerDirName;
          continue;
        }

        struct dirent * innerDp;
        while ((innerDp = readdir(innerDirp)) != NULL) {
            // find file descriptor file
          if(string(innerDp->d_name).find("_fd_") != string::npos) {
            std::ostringstream fileSs;
            fileSs << innerDirName << "/" << innerDp->d_name;
            sample_type fd;
            double label = atof(dp->d_name);
            LOG(INFO) << "deserializing " << fileSs.str() << " as file descriptor";
            dlib::deserialize(fileSs.str()) >> fd;
            samples.push_back(fd);
            labels.push_back(label);
            LOG(INFO) << "adding label " << label << " fd " << innerDp->d_name;
          }
        }
        closedir(innerDirp);
    }
    closedir(dirp);
  }

  void buildClassifier(const std::string& srcPath) {
    std::vector<sample_type> samples;
    std::vector<double> labels;
    populateData(srcPath, samples, labels);
    dlib::krr_trainer<rbf_kernel> rbf_trainer;
    ovo_trainer trainer;
    rbf_trainer.set_kernel(rbf_kernel(0.1));
    trainer.set_trainer(rbf_trainer);
    mDecisionFn = trainer.train(samples, labels);
    LOG(INFO) << " Label for test " << mDecisionFn(samples[0]);
      //testClassifier(srcPath, samples);
  }

  void storeModel(const std::string& srcPath) {
    std::ostringstream ss;
    ss << srcPath << "/df_classifier.dat";
    LOG(INFO) << " storing classifier " << ss.str();
    dlib::serialize(ss.str()) << mDecisionFn;
  }

  dlib::one_vs_one_decision_function<ovo_trainer> mDecisionFn;
  std::string mModelPath;
  anet_type net;
};


/******* FACE DETECTION **********/

class DLibHOGFaceDetector {
 private:
  std::string mLandMarkModel;
  std::vector<dlib::rectangle> mRets;
  dlib::shape_predictor msp;
  std::unordered_map<int, dlib::full_object_detection> mFaceShapeMap;
  dlib::frontal_face_detector mFaceDetector;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = dlib::get_frontal_face_detector();
  }

 public:
  DLibHOGFaceDetector() {
    init();
  }

  DLibHOGFaceDetector(const std::string& landmarkmodel)
      : mLandMarkModel(landmarkmodel) {
    init();
    if (!mLandMarkModel.empty() && jniutils::fileExists(mLandMarkModel)) {
      dlib::deserialize(mLandMarkModel) >> msp;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
  }

  #define FACE_DOWNSAMPLE_RATIO 4
  std::vector<dlib::matrix<float,0,1>> getFaceDescriptorsFromImage(
    DlibFaceRecognition *recogPtr, const cv::Mat& image) {
      std::vector<int> labels;
      if (image.channels() == 1) {
        cv::cvtColor(image, image, CV_GRAY2BGR);
      }
      CHECK(image.channels() == 3);
      cv::Mat image_small;
      cv::resize(image, image_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO,
      1.0/FACE_DOWNSAMPLE_RATIO);

      dlib::cv_image<dlib::bgr_pixel> img(image);
      dlib::cv_image<dlib::bgr_pixel> imgSmall(image_small);

      int num_faces = 0;
      LOG(INFO) << __PRETTY_FUNCTION__ << " " << __LINE__;
      std::vector<dlib::full_object_detection> shapes;
      //for (auto face : mFaceDetector(img))
      for (auto face : mFaceDetector(imgSmall))
      {
        dlib::rectangle r(
                     (long)(face.left() * FACE_DOWNSAMPLE_RATIO),
                     (long)(face.top() * FACE_DOWNSAMPLE_RATIO),
                     (long)(face.right() * FACE_DOWNSAMPLE_RATIO),
                     (long)(face.bottom() * FACE_DOWNSAMPLE_RATIO)
                  );
          auto shape = msp(img, r);
          shapes.push_back(move(shape));
      }
      dlib::array<dlib::matrix<dlib::rgb_pixel> > face_chips;
      extract_image_chips(img, get_face_chip_details(shapes, 150,0.25), face_chips);
      return recogPtr->getFaceDescriptors(face_chips);
  }

  virtual inline std::vector<int> recognizeFaceChipsBitmap(DlibFaceRecognition *recogPtr,
    const cv::Mat& image) {
    std::vector<int> labels;

    if (image.empty())
      return labels;

    for(auto fd : getFaceDescriptorsFromImage(recogPtr, image)) {
      auto _label = recogPtr->mDecisionFn(fd);
      LOG(INFO) << " recognized label : " << _label;
      labels.push_back(_label);
    }
    return labels;
  }

  virtual inline std::vector<int> recognizeFaceChips(DlibFaceRecognition *recogPtr,
    const std::string& srcPath) {

    return recognizeFaceChipsBitmap(recogPtr, cv::imread(srcPath, CV_LOAD_IMAGE_COLOR));
  }

  virtual inline int saveFaceChipsAndDescriptors(DlibFaceRecognition *recogPtr,
    const std::string& srcPath, const std::string& dstPath) {
    int num_fd = 0;
    for(auto fd : getFaceDescriptorsFromImage(recogPtr,
      cv::imread(srcPath, CV_LOAD_IMAGE_COLOR))) {
      std::ostringstream ss;
      ss << dstPath << "_fd_" << num_fd << ".dat";
      dlib::serialize(ss.str()) << fd;
      num_fd++;
    }
    return num_fd;
  }
};
