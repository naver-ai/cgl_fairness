# Learning Fair Classifiers with Partially Annotated Group Labels (CVPR 2022)

Official Pytorch implementation of <strong>Learning Fair Classifiers with Partially Annotated Group Labels</strong> | [Paper](https://arxiv.org/abs/2111.14581)

[Sangwon Jung](https://scholar.google.com/citations?user=WdC_a5IAAAAJ&hl=ko)<sup>1</sup> [Sanghyuk Chun](https://sanghyukchun.github.io/home/)<sup>2</sup> [Taesup Moon](https://scholar.google.com/citations?user=lQlioBoAAAAJ&hl=ko)<sup>1, 3</sup>

<sup>1</sup><sub>Department of ECE/ASRI, Seoul National University<br>
<sup>2</sup><sub>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)</sub><br>
<sup>3</sup><sub>Interdisciplinary Program in Artificial Intelligence, Seoul National University

Recently, fairness-aware learning have become increasingly crucial, but most of those methods operate by assuming the availability of fully annotated demographic group labels. We emphasize that such assumption is unrealistic for real-world applications since group label annotations are expensive and can conflict with privacy issues. In this paper, we consider a more practical scenario, dubbed as Algorithmic Group <strong>Fair</strong>ness with the <strong>P</strong>artially annotated <strong>G</strong>roup labels (<strong>Fair-PG</strong>). We observe that the existing methods to achieve group fairness perform even worse than the vanilla training, which simply uses full data only with target labels, under Fair-PG. To address this problem, we propose a simple <strong>C</strong>onfidence-based <strong>G</strong>roup <strong>L</strong>abel assignment (<strong>CGL</strong>) strategy that is readily applicable to any fairness-aware learning method. CGL utilizes an auxiliary group classifier to assign pseudo group labels, where random labels are assigned to low confident samples. We first theoretically show that our method design is better than the vanilla pseudo-labeling strategy in terms of fairness criteria. Then, we empirically show on several benchmark datasets that by combining CGL and the state-of-the-art fairness-aware in-processing methods, the target accuracies and the fairness metrics can be jointly improved compared to the baselines. Furthermore, we convincingly show that CGL enables to naturally augment the given group-labeled dataset with external target label-only datasets so that both accuracy and fairness can be improved.

## Updates

- 11 Apr, 2022: Initial upload.
    
## Dataset preparation
1. Download dataset
- UTKFace :
    [link](https://susanqq.github.io/UTKFace/) (We used Aligned&Cropped Faces from the site)
- CelebA :
    [link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (We used Aligned&Cropped Faces from the site)
- Tabular datasets (Propublica Compas & Adult) : 
    [link](https://github.com/Trusted-AI/AIF360)
2. Locate downloaded datasets to './data' directory
    
## How to train 
We note that CGL first trains a group classifier and a proper threshold using a train set and a validation set splitted from a group-labeled training dataset. And then, the group-unlabeled training samples are annotated with pseudo group labels based on CGL's assignment rules. Finally, we can train a fair model using base fair training methods such as LBC, FairHSIC or MFD. 
   
### 1. Train a group classifier
```
$ python main_groupclf.py --model <model_type> --method scratch \
    --dataset <dataset_name> \
    --version groupclfval \
    --sv <group_label ratio> 
```
    
In above command, the 'version' can be chosen beween 'groupclf' or 'groupclfval' that indicate whether the group-labeled data is splitted into a train/validation set or not. For CGL, you should choice 'groupclfval' and for the pseudo-label baseline, you should choice 'groupclf'.
    
### 2. Find a threshold and save the predictions of group classifier 
```
$ python main_groupclf.py --model <model_type> --method scratch \ 
    --dataset <dataset_name> \
    --mode eval \
    --version groupclfval \
    --sv <group_label_ratio>  
```
    
### 3. Train a fair model using any fair-training methods

- MFD. For the feature distilation, MFD needs a teacher model that is trained from scratch. 
```
# train a scratch model
$ python main.py --model <model_type> --method scratch --dataset dataset_name 
$ python main.py --model <model_type> --method mfd \
    --dataset <dataset_name> \
    --labelwise \
    --version cgl \
    --sv {group_label_ratio} \
    --lamb 100 \
    --teacher-path <your_model_trained_from_scratch> 
```
- FairHSIC
```
$ python main.py --model <model_type> --method fairhsic \
    --dataset <dataset_name> \
    --labelwise  \
    --version cgl \
    --sv {group_label_ratio} \
    --lamb 100  
```
- LBC
```
$ python main.py --model <model_type> --method lbc \
    --dataset <dataset_name> \
    --iter <iter> \
    --version cgl \
    --sv {group_label_ratio} \
```
## How to cite

```
@inproceedings{jung2021cgl,
    title={Learning Fair Classifiers with Partially Annotated Group Labels}, 
    author={Sangwon Jung and Sanghyuk Chun and Taesup Moon},
    year={2022},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
## License

```
Copyright (c) 2022-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
