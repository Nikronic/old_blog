Topics covered in this blog:
1. Digital image processing (DIP)
2. Pattern recognition (PR)
3. Artificial neural networks (ANN)
4. Computer vision (CV)
5. Data mining (DM)


## 1. Digital Image Processing (DIP)
1. [Shutter, Camera and Histogram Equalization](post/dip01/): Discussion around camera and shutter speed and its effect on quality along side histogram equalization
2. [Convolutions, Separable Kernels and Gaussian Filter](post/dip02/): How separate kernels provide efficiency in convolution operation and implementation of Gaussian smoothing filter in comparison to OpenCV counter part.
3. [DCT, DFT, Transformation, Noise Reduction](post/dip03/): The patterns in an image and how they affect frequency spectrum and its direction which leads to designing noise reduction methods using DCT and DFT methods. This phenomenon has been shown on both natural real-world images and synthetic images for better comprehension. Also, the usage of filtering high-frequency for compression has been demonstrated.
4. [DCT, DWT, Wavelet, Haar and Noise Reduction](post/dip04/): Similar to previous discussion, but this time around differences between wavelet and DCT and DFT.
5. [Canny Edge Detector, Hough Transform, LineSegmentDetector and CamScanner Clone](post/dip05/): This post covers implementation of Canny edge detector in numpy, utilizing gradient direction for Hough transform to obtain lines in an image, trying to build a CamScanner/Adobe Scan clone using LineSegmentDetector and Homography via OpenCV.
6. [Ellipse Specific Fitting, RANSAC and Extracting Ellipse](post/dip06/): Discussion around extracting ellipses from image and dealing with noises using RANSAC method for parameter fitting.
7. [Face Detection, Face Landmarks and dlib](post/dip07/): Discussion around the differences between OpenCV and dlib implementation of face landmark detection.
8. [JPEG, Entropy, Compression Ratio, Huffman and Multiresolution Segmentation](post/dip08/): This post covers implementation of JPEG algorithm in numpy, the information in an random or "predictable" image and how it affects the compression ratio and compression algorithms such as Huffman. 
9. [Morphological Operation, Hit-or-Miss, Textural Feature, Soft LBP, HOG, SVM and kNN](post/dip09/): Discussion around how morphological operations act on images and different operators such as hit-or-miss. This post ends with implementation of a machine learning model for Persian handwritten digit classification using textural and geometrical feature engineering and models such as Random Forest, kNN, SVM, etc. In the end, Confusion matrix is being used to evaluate methods.
10. [FCN or CNN, AlexNet, VGG, ResNet, Inception(GoogleNet), Xception and CIFAR10 classifier](post/dip10/): This post describes the main difference between fully connected and convolutional neural networks and architecture and improvement of famous CNN architectures from LeNet to Xception has been discussed. In the end, a CNN model using Keras has been trained for CIFAR10 classification.
11. [RCNN, Fast RCNN, Faster RCNN and Template Matching](post/dip11/): This post develops a good understanding around the evolution of RCNN model for object detection. In the end, a classic Template Matching method has been implemented to read numbers on a car plates.
12. [Dilated Convolution, Mask RCNN and Number of Parameters](post/dip13/): What are dilated convolutions and why are they used? How Mask RCNN is related to Faster RCNN and its older generation, RCNN? These question have been answered in this post.
13. [Cycle GAN, PCA, AutoEncoder and CIFAR10 Generator](post/dip14/): To know how Cycle GAN works and what the the differences between PCA and AutoEncoder are, and also to see and implementation of a generative model for CIFAR10 impaired images in TF1.x.

## 2. Pattern Recognition (PR)
1. [Movie Review Sentiment Analysis](post/pr01/): We start of by preparing the dataset, from cleaning to lemmatization; then we extract features using Bag of Words, BeRT embeddings, TF-IDF and Word2Vec methods. We train machine learning models such as SVM and Naive Bayes to learn this data and in the end, we report metrics such as F1, precision, recall and ROC_AUC curves.
2. [Imbalance Learning and Evaluation using AdaBoost, RUSBoost, SMOTEBoost, RBBoost and ANOVA Measure](post/pr02/): In this post, we dive deep into Imbalance Learning pipeline from data preparation to model fitting. We discuss normalization and K-fold validation for data preparation, then we define and discuss Ensemble Learning methods, bagging, boosting and implement them in numpy from scratch. These methods include, AdaBoost, AdaBoostM2, SMOTEBoost, RBBoost. Then we compare accuracies and precision-recall bars and AUC_ROC curves. In the end, to show statistical differences between models, we implement and report ANOVA test.
3. [Image Classification using Pretrained VGG Activation Maps as Features for SVM and FCNN models](post/pr03/): To do image classification on 3 classes, we first prepare data, then we use Pretrained VGG model's weight as feature extraction method. Then we reduce features using PCA and t-SNE to 2 and 3 dimensions for visualization purposes too. In the end we train Linear SVM and RBF SVM on both t-SNE and PCA applied set of data and compare the results.

## 3. Artificial Neural Networks (ANN)
1. [Perceptron With Hebb Rule For Linear Data](post/ann01/): In this post, we first explain then implement a basic Perceptron model using numpy and update its weight using Hebb rule for a linearly separable data. In the end we compare the effect of choosing Tanh and Sigmoid as the activation function.
2. [MLP With Backpropagation For Non-Linear Data (XOR)](post/ann02/): Here, we extend previous model that only worked for linear data, to a Multi-Layer Perceptron (MLP) to fit non-linear data and we update its weights using backpropagation. This model has been explained and implemented in numpy.
3. [Kohonen Self Organizing Maps For Learning MNIST](post/ann03/): In this post, we discuss the Kohonen Self-Organizing Maps as a different type of neural networks that can fit non-linear data. We test this by implementing the model in numpy and test it on MNIST dataset. In the end, we argue the method for computing accuracy and the best model size and its symmetric structure.
4. [Training Tensorflow 1.x Model for MNIST](post/ann04/): This post just contains a common implementation of TF1.x model for MNIST dataset, but this time we explain for a bit why the particular architecture/structure has been used in comparison to older methods.

## 4. Computer Vision (CV)
1. [Phase Amplitude Combination, Hybrid Images and Cutoff Effect On It](post/cv02/): Have you ever seen those images that look different to different people? Or more closer to this post, when you see them in different distances, you see different image? Well, this has to do with the Phase and Amplitude of combination of two images. By combining low-frequency information of one image and high-frequency of the other image, we can generate Hybrid Images.

## 5. Data Mining (DM)
1. [Statistical Evaluation Metrics](post/minidm01/): This post compares many of statistical metrics for evaluating models with same result but different scenarios. We try to bring real-world examples here too. Some of these metrics are: ROC-AUC vs F1 and PR-Curve; P@1, P@10, P@K, MAP, MRR, NDCG, r-Prec, bPref; F-Macro and F-Micro; Miss and Fallout; Specificity, Sensitivity, PPV and NPV in Medical Predictor Assessment.