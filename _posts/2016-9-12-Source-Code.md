---
layout: post
title: MLCV Lab Source Code Repository
---

# Adaptive Online Sequential ELM for Concept Drift Tackling

## Abstract

A machine learning method needs to adapt to over time changes in the environment. Such changes are known as concept drift. In this paper, we propose concept drift tackling method as an enhancement of Online Sequential Extreme Learning Machine (OS-ELM) and Constructive Enhancement OS-ELM (CEOS-ELM) by adding adaptive capability for classification and regression problem. The scheme is named as adaptive OS-ELM (AOS-ELM). It is a single classifier scheme that works well to handle real drift, virtual drift, and hybrid drift. The AOS-ELM also works well for sudden drift and recurrent context change type. The scheme is a simple unified method implemented in simple lines of code. We evaluated AOS-ELM on regression and classification problem by using concept drift public data set (SEA and STAGGER) and other public data sets such as MNIST, USPS, and IDS. Experiments show that our method gives higher kappa value compared to the multiclassifier ELM ensemble. Even though AOS-ELM in practice does not need hidden nodes increase, we address some issues related to the increasing of the hidden nodes such as error condition and rank values. We propose taking the rank of the pseudoinverse matrix as an indicator parameter to detect “underfitting” condition.

[Source code](https://github.com/mlcv-lab/adaptive-OS-ELM)

[//]: # (Next you can update your site name, avatar and other options using the _config.yml file in the root of your repository (shown below).)

[//]: # (![_config.yml]({{ site.baseurl }}/images/config.png))

[//]: # (The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.)

