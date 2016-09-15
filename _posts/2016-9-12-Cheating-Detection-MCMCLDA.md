---
layout: post
title: Classifying Abnormal Activities in Exam Using Multi-class Markov Chain LDA Based on MODEC Features
---

## Abstract

In this paper, we apply MCMCLDA (Multi-class Markov Chain Latent Dirichlet Allocation) model to classify abnormal activity of students in an examination. Abnormal activity in exams is defined as a cheating activity. We compare the usage of Harris3D interest point detector and a human joints detector, MODEC (Multimodal Decomposable Models), as the feature detector. Experiment results show that using MODEC to detect arm joints and head location as interest point gives better performance in accuracy and computational time than Harris3D when classifying cheating activity. MODEC suffers low accuracy due to its inability to differentiate elbow and wrist when the object wears clothes with indistinguishable colors. Meanwhile, Harris3D detects too many irrelevant interest point to recognize cheating activity reliably.

[Paper](ttps://www.researchgate.net/publication/301202274_Classifying_Abnormal_Activities_in_Exam_Using_Multi-class_Markov_Chain_LDA_Based_on_MODEC_Features)

[Source code](https://github.com/jansonh/Cheating-Detection-MCMCLDA)

[Data](https://drive.google.com/open?id=0Bz96X-nFVG-kUW5IUXllY0F6eXc)

[//]: # (Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).)

[//]: # (![_config.yml]({{ site.baseurl }}/images/config.png))

[//]: # (The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.)

