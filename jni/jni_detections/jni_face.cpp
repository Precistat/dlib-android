/*
 * jni_pedestrian_det.cpp using google-style
 *
 *  Created on: Oct 20, 2015
 *      Author: Tzutalin
 *
 *  Copyright (c) 2015 Tzutalin. All rights reserved.
 */
#include <android/bitmap.h>
#include <jni_common/jni_bitmap2mat.h>
#include <jni_common/jni_primitives.h>
#include <jni_common/jni_fileutils.h>
#include <jni_common/jni_utils.h>
#include <jni_face.h>
#include <jni.h>

using namespace cv;
extern JNI_VisionDetRet* g_pJNI_VisionDetRet;

namespace {

#define JAVA_NULL 0
using DetectorPtr = DLibHOGFaceDetector*;
using RecogPtr = DlibFaceRecognition*;

class JNI_FaceDet {
 public:
  JNI_FaceDet(JNIEnv* env) {
    jclass clazz = env->FindClass(CLASSNAME_FACE_DET);
    mNativeDetContext = env->GetFieldID(clazz, "mNativeDetContext", "J");
    mNativeRecogContext = env->GetFieldID(clazz, "mNativeRecogContext", "J");
    env->DeleteLocalRef(clazz);
  }

  DetectorPtr getDetectorPtrFromJava(JNIEnv* env, jobject thiz) {
    DetectorPtr const p = (DetectorPtr)env->GetLongField(thiz, mNativeDetContext);
    return p;
  }

  void setDetectorPtrToJava(JNIEnv* env, jobject thiz, jlong ptr) {
    env->SetLongField(thiz, mNativeDetContext, ptr);
  }

  RecogPtr getRecogPtrFromJava(JNIEnv* env, jobject thiz) {
    RecogPtr const p = (RecogPtr)env->GetLongField(thiz, mNativeRecogContext);
    return p;
  }

  void setRecogPtrToJava(JNIEnv* env, jobject thiz, jlong ptr) {
    env->SetLongField(thiz, mNativeRecogContext, ptr);
  }

  jfieldID mNativeDetContext;
  jfieldID mNativeRecogContext;
};

// Protect getting/setting and creating/deleting pointer between java/native
std::mutex gLock;

std::shared_ptr<JNI_FaceDet> getJNI_FaceDet(JNIEnv* env) {
  static std::once_flag sOnceInitflag;
  static std::shared_ptr<JNI_FaceDet> sJNI_FaceDet;
  std::call_once(sOnceInitflag, [env]() {
    sJNI_FaceDet = std::make_shared<JNI_FaceDet>(env);
  });
  return sJNI_FaceDet;
}

DetectorPtr const getDetectorPtr(JNIEnv* env, jobject thiz) {
  std::lock_guard<std::mutex> lock(gLock);
  return getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
}

RecogPtr const getRecogPtr(JNIEnv* env, jobject thiz) {
  std::lock_guard<std::mutex> lock(gLock);
  return getJNI_FaceDet(env)->getRecogPtrFromJava(env, thiz);
}

// The function to set a pointer to java and delete it if newPtr is empty
void setDetectorPtr(JNIEnv* env, jobject thiz, DetectorPtr newPtr) {
  std::lock_guard<std::mutex> lock(gLock);
  DetectorPtr oldPtr = getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
  if (oldPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager delete old ptr : " << oldPtr;
    delete oldPtr;
  }

  if (newPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager set new ptr : " << newPtr;
  }

  getJNI_FaceDet(env)->setDetectorPtrToJava(env, thiz, (jlong)newPtr);
}

void setRecogPtr(JNIEnv* env, jobject thiz, RecogPtr newPtr) {
  std::lock_guard<std::mutex> lock(gLock);
  RecogPtr oldPtr = getJNI_FaceDet(env)->getRecogPtrFromJava(env, thiz);
  if (oldPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager delete old ptr : " << oldPtr;
    delete oldPtr;
  }

  if (newPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager set new ptr : " << newPtr;
  }

  getJNI_FaceDet(env)->setRecogPtrToJava(env, thiz, (jlong)newPtr);
}

}  // end unnamespace

#ifdef __cplusplus
extern "C" {
#endif

#define DLIB_FACE_JNI_METHOD(METHOD_NAME) \
  Java_com_keye_karthiksubraveti_keye_FaceDet_##METHOD_NAME

void JNIEXPORT
    DLIB_FACE_JNI_METHOD(jniNativeClassInit)(JNIEnv* env, jclass _this) {}

JNIEXPORT jint JNICALL
    DLIB_FACE_JNI_METHOD(jniSaveFaceChips)(JNIEnv* env, jobject thiz,
										jstring imgSrcPath, jstring imgDstPath) {
  LOG(INFO) << "jniSaveFaceChips";
  const char* img_src_path = env->GetStringUTFChars(imgSrcPath, 0);
  const char* img_dst_path = env->GetStringUTFChars(imgDstPath, 0);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  RecogPtr recogPtr = getRecogPtr(env, thiz);
  jint size = detPtr->saveFaceChipsAndDescriptors(
    recogPtr, std::string(img_src_path), std::string(img_dst_path));
  env->ReleaseStringUTFChars(imgSrcPath, img_src_path);
  env->ReleaseStringUTFChars(imgDstPath, img_dst_path);
  LOG(INFO) << "det face size: " << size;
  return size;
}

JNIEXPORT jintArray JNICALL
    DLIB_FACE_JNI_METHOD(jniRecognizeFaceChips)(JNIEnv* env, jobject thiz,
										jstring imgSrcPath) {
  LOG(INFO) << "jniRecognizeFaceChips";
  const char* img_src_path = env->GetStringUTFChars(imgSrcPath, 0);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  RecogPtr recogPtr = getRecogPtr(env, thiz);
  std::vector<int> labelVec = detPtr->recognizeFaceChips(
    recogPtr, std::string(img_src_path));
  env->ReleaseStringUTFChars(imgSrcPath, img_src_path);
  jintArray labels;
  labels  = env->NewIntArray(labelVec.size());
  labels = (jintArray)env->NewGlobalRef(labels);
  if(!labels) {
    LOG(INFO) << "jintarray for labels couldn't be allocated";
    return NULL;
  }
  jint *labelsJint = new jint[labelVec.size()];
  for(int i = 0; i < labelVec.size(); i++) {
    labelsJint[i] = labelVec[i];
  }
  env->SetIntArrayRegion(labels, 0, labelVec.size(), labelsJint);

  return labels;
}

JNIEXPORT jintArray JNICALL
    DLIB_FACE_JNI_METHOD(jniRecognizeFaceChipsBitmap)(JNIEnv* env, jobject thiz,
										jobject bitmap) {
  LOG(INFO) << "jniRecognizeFaceChipsBitmap";

  cv::Mat rgbaMat;
  cv::Mat bgrMat;
  jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
  cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  RecogPtr recogPtr = getRecogPtr(env, thiz);

  std::vector<int> labelVec = detPtr->recognizeFaceChipsBitmap(recogPtr, bgrMat);

  jintArray labels;
  labels  = env->NewIntArray(labelVec.size());
  labels = (jintArray)env->NewGlobalRef(labels);
  if(!labels) {
    LOG(INFO) << "jintarray for labels couldn't be allocated";
    return NULL;
  }
  jint *labelsJint = new jint[labelVec.size()];
  for(int i = 0; i < labelVec.size(); i++) {
    labelsJint[i] = labelVec[i];
  }
  env->SetIntArrayRegion(labels, 0, labelVec.size(), labelsJint);
  return labels;
}


JNIEXPORT void JNICALL
    DLIB_FACE_JNI_METHOD(jniTrainClassifier)(JNIEnv* env, jobject thiz, jstring srcPath) {
  LOG(INFO) << "jniTrainClassifier";
  const char* src_path = env->GetStringUTFChars(srcPath, 0);
  RecogPtr recogPtr = getRecogPtr(env, thiz);
  recogPtr->buildClassifier(std::string(src_path));
  env->ReleaseStringUTFChars(srcPath, src_path);
  LOG(INFO) << "classifier build successfully ";
}

JNIEXPORT void JNICALL
    DLIB_FACE_JNI_METHOD(jniStoreModel)(JNIEnv* env, jobject thiz, jstring srcPath) {
  LOG(INFO) << "jniStoreModel";
  const char* src_path = env->GetStringUTFChars(srcPath, 0);
  RecogPtr recogPtr = getRecogPtr(env, thiz);
  recogPtr->storeModel(std::string(src_path));
  env->ReleaseStringUTFChars(srcPath, src_path);
  LOG(INFO) << "classifier model stored successfully ";
}

jint JNIEXPORT JNICALL DLIB_FACE_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz,
                                       jstring jLandmarkPath, jstring jRecogPath) {
  LOG(INFO) << "jniInit";
  std::string landmarkPath = jniutils::convertJStrToString(env, jLandmarkPath);
  std::string recogPath = jniutils::convertJStrToString(env, jRecogPath);
  DetectorPtr detPtr = new DLibHOGFaceDetector(landmarkPath);
  RecogPtr recogPtr = new DlibFaceRecognition(recogPath);
  setDetectorPtr(env, thiz, detPtr);
  setRecogPtr(env, thiz, recogPtr);
  return JNI_OK;
}

jint JNIEXPORT JNICALL
    DLIB_FACE_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
  LOG(INFO) << "jniDeInit";
  setDetectorPtr(env, thiz, JAVA_NULL);
  return JNI_OK;
}

#ifdef __cplusplus
}
#endif
