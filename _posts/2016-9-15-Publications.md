---
layout: post
title: Publications
---

1. [A New Data Representation Based on Training Data Characteristics to Extract Drug Named-Entity in Medical Text](https://mlcv-lab.github.io/Publications#a-new-data-representation-based-on-training-data-characteristics-to-extract-drug-named-entity-in-medical-text)
2. [Adaptive Online Sequential ELM for Concept Drift Tackling](https://mlcv-lab.github.io/Publications#adaptive-online-sequential-elm-for-concept-drift-tackling)
3. [Classifying Abnormal Activities in Exam Using Multi-class Markov Chain LDA Based on MODEC Features](https://mlcv-lab.github.io/Publications#classifying-abnormal-activities-in-exam-using-multi-class-markov-chain-lda-based-on-modec-features)
4. [Combining Generative and Discriminative Neural Networks for Sleep Stages Classification](https://mlcv-lab.github.io/Publications#combining-generative-and-discriminative-neural-networks-for-sleep-stages-classification)
5. [Ischemic Stroke Identification Based on EEG and EOG using 1D Convolutional Neural Network and Batch Normalization](https://mlcv-lab.github.io/Publications#ischemic-stroke-identification-based-on-eeg-and-eog-using-1d-convolutional-neural-network-and-batch-normalization)
6. [Metaheuristic Algorithms for Convolution Neural Network](https://mlcv-lab.github.io/Publications#metaheuristic-algorithms-for-convolution-neural-network)
7. [Multiple Regularizations Deep Learning for Paddy Growth Stages Classification from LANDSAT-8](https://mlcv-lab.github.io/Publications#multiple-regularizations-deep-learning-for-paddy-growth-stages-classification-from-landsat-8)
8. [Sequence-based Sleep Stage Classification using Conditional Neural Fields](https://mlcv-lab.github.io/Publications#sequence-based-sleep-stage-classification-using-conditional-neural-fields)

## A New Data Representation Based on Training Data Characteristics to Extract Drug Named-Entity in Medical Text

### Abstract

One essential task in information extraction from the medical corpus is drug name recognition. Compared with text sources come from other domains, the medical text is special and has unique characteristics. In addition, the medical text mining poses more challenges, e.g., more unstructured text, the fast growing of new terms addition, a wide range of name variation for the same drug. The mining is even more challenging due to the lack of labeled dataset sources and external knowledge, as well as multiple token representations for a single drug name that is more common in the real application setting. Although many approaches have been proposed to overwhelm the task, some problems remained with poor F-score performance (less than 0.75). This paper presents a new treatment in data representation techniques to overcome some of those challenges. We propose three data representation techniques based on the characteristics of word distribution and word similarities as a result of word embedding training. The first technique is evaluated with the standard NN model, i.e., MLP (Multi-Layer Perceptrons). The second technique involves two deep network classifiers, i.e., DBN (Deep Belief Networks), and SAE (Stacked Denoising Encoders). The third technique represents the sentence as a sequence that is evaluated with a recurrent NN model, i.e., LSTM (Long Short Term Memory). In extracting the drug name entities, the third technique gives the best F-score performance compared to the state of the art, with its average F-score being 0.8645.

[Arxiv](https://arxiv.org/abs/1610.01891){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Adaptive Online Sequential ELM for Concept Drift Tackling

### Abstract

A machine learning method needs to adapt to over time changes in the environment. Such changes are known as concept drift. In this paper, we propose concept drift tackling method as an enhancement of Online Sequential Extreme Learning Machine (OS-ELM) and Constructive Enhancement OS-ELM (CEOS-ELM) by adding adaptive capability for classification and regression problem. The scheme is named as adaptive OS-ELM (AOS-ELM). It is a single classifier scheme that works well to handle real drift, virtual drift, and hybrid drift. The AOS-ELM also works well for sudden drift and recurrent context change type. The scheme is a simple unified method implemented in simple lines of code. We evaluated AOS-ELM on regression and classification problem by using concept drift public data set (SEA and STAGGER) and other public data sets such as MNIST, USPS, and IDS. Experiments show that our method gives higher kappa value compared to the multiclassifier ELM ensemble. Even though AOS-ELM in practice does not need hidden nodes increase, we address some issues related to the increasing of the hidden nodes such as error condition and rank values. We propose taking the rank of the pseudoinverse matrix as an indicator parameter to detect “underfitting” condition.

[Paper](https://www.hindawi.com/journals/cin/2016/8091267){:target="_blank"} [Arxiv](https://arxiv.org/abs/1610.01922){:target="_blank"}

[Source code](https://github.com/mlcv-lab/adaptive-OS-ELM){:target="_blank"}

[Data](https://drive.google.com/?authuser=0#folders/0B8Db7VyHy5jocnNuOGJzTW4xMVU){:target="_blank"}

Indexing: [Pubmed](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4993962/){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Classifying Abnormal Activities in Exam Using Multi-class Markov Chain LDA Based on MODEC Features

### Abstract

In this paper, we apply MCMCLDA (Multi-class Markov Chain Latent Dirichlet Allocation) model to classify abnormal activity of students in an examination. Abnormal activity in exams is defined as a cheating activity. We compare the usage of Harris3D interest point detector and a human joints detector, MODEC (Multimodal Decomposable Models), as the feature detector. Experiment results show that using MODEC to detect arm joints and head location as interest point gives better performance in accuracy and computational time than Harris3D when classifying cheating activity. MODEC suffers low accuracy due to its inability to differentiate elbow and wrist when the object wears clothes with indistinguishable colors. Meanwhile, Harris3D detects too many irrelevant interest point to recognize cheating activity reliably.

[Paper](https://www.researchgate.net/publication/301202274_Classifying_Abnormal_Activities_in_Exam_Using_Multi-class_Markov_Chain_LDA_Based_on_MODEC_Features){:target="_blank"}

[Source code](https://github.com/jansonh/Cheating-Detection-MCMCLDA){:target="_blank"}

[Data](https://drive.google.com/open?id=0Bz96X-nFVG-kUW5IUXllY0F6eXc){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Combining Generative and Discriminative Neural Networks for Sleep Stages Classification

### Abstract

Sleep stages pattern provides important clues in diagnosing the presence of sleep disorder. By analyzing sleep stages pattern and extracting its features from EEG, EOG, and EMG signals, we can classify sleep stages. This study presents a novel classification model for predicting sleep stages with a high accuracy. The main idea is to combine the generative capability of Deep Belief Network (DBN) with a discriminative ability and sequence pattern recognizing capability of Long Short-term Memory (LSTM). We use DBN that is treated as an automatic higher level features generator. The input to DBN is 28 "handcrafted" features as used in previous sleep stages studies. We compared our method with other techniques which combined DBN with Hidden Markov Model (HMM).In this study, we exploit the sequence or time series characteristics of sleep dataset. To the best of our knowledge, most of the present sleep analysis from polysomnogram relies only on single instanced label (nonsequence) for classification. In this study, we used two datasets: an open data set that is treated as a benchmark; the other dataset is our sleep stages dataset (available for download) to verify the results further. Our experiments showed that the combination of DBN with LSTM gives better overall accuracy 98.75\% (Fscore=0.9875) for benchmark dataset and 98.94\% (Fscore=0.9894) for MKG dataset. This result is better than the state of the art of sleep stages classification that was 91.31\%.

[Arxiv](https://arxiv.org/abs/1610.01741){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Ischemic Stroke Identification Based on EEG and EOG using 1D Convolutional Neural Network and Batch Normalization

### Abstract

In 2015, stroke was the number one cause of death in Indonesia. The majority type of stroke is ischemic. The standard tool for diagnosing stroke is CT-Scan. For developing countries like Indonesia, the availability of CT-Scan is very limited and still relatively expensive. Because of the availability, another device that potential to diagnose stroke in Indonesia is EEG. Ischemic stroke occurs because of obstruction that can make the cerebral blood flow (CBF) on a person with stroke has become lower than CBF on a normal person (control) so that the EEG signal have a deceleration. On this study, we perform the ability of 1D Convolutional Neural Network (1DCNN) to construct classification model that can distinguish the EEG and EOG stroke data from EEG and EOG control data. To accelerate training process our model we use Batch Normalization. Involving 62 person data object and from leave one out the scenario with five times repetition of measurement we obtain the average of accuracy 0.86 (F-Score 0.861) only at 200 epoch. This result is better than all over shallow and popular classifiers as the comparator (the best result of accuracy 0.69 and F-Score 0.72 ). The feature used in our study were only 24 handcrafted feature with simple feature extraction process.

[Arxiv](https://arxiv.org/abs/1610.01757){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Metaheuristic Algorithms for Convolution Neural Network

### Abstract

A typical modern optimization technique is usually either heuristic or metaheuristic. This technique has managed to solve some optimization problems in the research area of science, engineering, and industry. However, implementation strategy of metaheuristic for accuracy improvement on convolution neural networks (CNN), a famous deep learning method, is still rarely investigated. Deep learning relates to a type of machine learning technique, where its aim is to move closer to the goal of artificial intelligence of creating a machine that could successfully perform any intellectual tasks that can be carried out by a human. In this paper, we propose the implementation strategy of three popular metaheuristic approaches, that is, simulated annealing, differential evolution, and harmony search, to optimize CNN. The performances of these metaheuristic methods in optimizing CNN on classifying MNIST and CIFAR dataset were evaluated and compared. Furthermore, the proposed methods are also compared with the original CNN. Although the proposed methods show an increase in the computation time, their accuracy has also been improved (up to 7.14 percent).

[Paper](https://www.hindawi.com/journals/cin/2016/1537325/){:target="_blank"} [Arxiv](https://arxiv.org/abs/1610.01925){:target="_blank"}

[Source code](https://github.com/mlcv-lab/Metaheuristic-Algorithms-CNN){:target="_blank"}

Indexing:
[Pubmed](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916328/){:target="_blank"} [Ebsco](http://web.a.ebscohost.com/abstract?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=16875265&AN=115985825&h=REeXbc79ljlHu1rpuJUNNgNhpkItEzTm1Oh9s%2bhEgZgSMKYAoR7Nae0SvQDyDmpTNBW%2b8CW6FO33NnoDM%2bCyMQ%3d%3d&crl=c&resultNs=AdminWebAuth&resultLocal=ErrCrlNotAuth&crlhashurl=login.aspx%3fdirect%3dtrue%26profile%3dehost%26scope%3dsite%26authtype%3dcrawler%26jrnl%3d16875265%26AN%3d115985825){:target="_blank"} [SemanticScholar](https://www.semanticscholar.org/paper/Metaheuristic-Algorithms-for-Convolution-Neural-Rere-Fanany/55e41ba8798bdc4cd07d3977e8d10f994f95ee6c){:target="_blank"} [NewsCentra](http://newscentral.exsees.com/item/dc94092311963f52023c6a0054c335fe-c1301184d53038c25b03600541a316dc){:target="_blank"} [MySizzle](http://www.myscizzle.com/search/abstract?id=27375738){:target="_blank"} [SaskatoonLibrary](http://saskatoonlibrary.ca/eds/item?dbid=edb&an=115985825){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Multiple Regularizations Deep Learning for Paddy Growth Stages Classification from LANDSAT-8

### Abstract

This study uses remote sensing technology that can provide information about the condition of the earth's surface area, fast, and spatially. The study area was in Karawang District, lying in the Northern part of West Java-Indonesia. We address a paddy growth stages classification using LANDSAT 8 image data obtained from multi-sensor remote sensing image taken in October 2015 to August 2016. This study pursues a fast and accurate classification of paddy growth stages by employing multiple regularizations learning on some deep learning methods such as DNN (Deep Neural Networks) and 1-D CNN (1-D Convolutional Neural Networks). The used regularizations are Fast Dropout, Dropout, and Batch Normalization. To evaluate the effectiveness, we also compared our method with other machine learning methods such as (Logistic Regression, SVM, Random Forest, and XGBoost). The data used are seven bands of LANDSAT-8 spectral data samples that correspond to paddy growth stages data obtained from i-Sky (eye in the sky) Innovation system. The growth stages are determined based on paddy crop phenology profile from time series of LANDSAT-8 images. The classification results show that MLP using multiple regularization Dropout and Batch Normalization achieves the highest accuracy for this dataset.

[Arxiv](https://arxiv.org/abs/1610.01795){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)

## Sequence-based Sleep Stage Classification using Conditional Neural Fields

### Abstract

Sleep signals from a polysomnographic database are sequences in nature. Commonly employed analysis and classification methods, however, ignored this fact and treated the sleep signals as non-sequence data. Treating the sleep signals as sequences, this paper compared two powerful unsupervised feature extractors and three sequence-based classifiers regarding accuracy and computational (training and testing) time after 10-folds cross-validation. The compared feature extractors are Deep Belief Networks (DBN) and Fuzzy C-Means (FCM) clustering. Whereas the compared sequence-based classifiers are Hidden Markov Models (HMM), Conditional Random Fields (CRF) and its variants, i.e., Hidden-state CRF (HCRF) and Latent-Dynamic CRF (LDCRF); and Conditional Neural Fields (CNF) and its variant (LDCNF). In this study, we use two datasets. The first dataset is an open (public) polysomnographic dataset downloadable from the Internet, while the second dataset is our polysomnographic dataset (also available for download). For the first dataset, the combination of FCM and CNF gives the highest accuracy (96.75\%) with relatively short training time (0.33 hours). For the second dataset, the combination of DBN and CRF gives the accuracy of 99.96\% but with 1.02 hours training time, whereas the combination of DBN and CNF gives slightly less accuracy (99.69\%) but also less computation time (0.89 hours).

[Arxiv](https://arxiv.org/abs/1610.01935){:target="_blank"}

[[Back to Top]](https://mlcv-lab.github.io/Publications)
