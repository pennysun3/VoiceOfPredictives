projectModeling
================

data prep
=========

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
voice <- read.csv("voice.csv")
y <- voice["label"]
voice <- data.frame(scale(select(voice,-label)))
voice$label <- y
```

``` r
pc <- prcomp(select(voice,-label))
voice_pc <- data.frame(pc$x[,1:6])
voice_pc$label <- y
```

cross validation function
=========================

``` r
CVInd <- function(n,K) {  #n is sample size; K is number of parts; returns K-length list of indices for each part
   m<-floor(n/K)  #approximate size of each part
   r<-n-m*K  
   I<-sample(n,n)  #random reordering of the indices
   Ind<-list()  #will be list of indices for all K parts
   length(Ind)<-K
   for (k in 1:K) {
      if (k <= r) kpart <- ((m+1)*(k-1)+1):((m+1)*k)  
         else kpart<-((m+1)*r+m*(k-r-1)+1):((m+1)*r+m*(k-r))
      Ind[[k]] <- I[kpart]  #indices for kth part of data
   }
   Ind
}
```

GAM
===

Without pca
-----------

``` r
library(mgcv)
```

    ## Loading required package: nlme

    ## 
    ## Attaching package: 'nlme'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse

    ## This is mgcv 1.8-23. For overview type 'help("mgcv-package")'.

``` r
set.seed(12345)

Nrep<-1 #number of replicates of CV
K<-3  #K-fold CV on each replicate
n.models = 1 #number of different models to fit and compare
n=nrow(voice)
yhat=matrix(0,n,n.models)
ccr<-matrix(0,Nrep,n.models)
y <- voice["label"]

for (j in 1:Nrep) {
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    out <- gam(as.integer(voice[-Ind[[k]],"label"]=="female")~s(Q25)+s(Q75)+s(IQR)+s(sp.ent)+s(sfm)+s(minfun)+s(meanfreq)+s(centroid),data=voice[-Ind[[k]],],family=binomial)
    yhat[Ind[[k]],1]<-predict(out,voice[Ind[[k]],],type="response")
  }
  ccr[j,] <- sum(ifelse(yhat>=0.5,"female","male") == y)/n
} 

ccr
```

    ##           [,1]
    ## [1,] 0.9400253

with PCA
--------

``` r
set.seed(12345)

Nrep<-1 #number of replicates of CV
K<-3 #K-fold CV on each replicate
n.models = 1 #number of different models to fit and compare
n=nrow(voice_pc)
yhat=matrix(0,n,n.models)
ccr<-matrix(0,Nrep,n.models)
y <- voice_pc["label"]

for (j in 1:Nrep) {
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    out <- gam(as.integer(voice_pc[-Ind[[k]],"label"]=="female")~s(PC1)+s(PC2)+s(PC3)+s(PC4)+s(PC5)+s(PC6),data=voice_pc[-Ind[[k]],],family=binomial)
    yhat[Ind[[k]],1]<-predict(out,voice_pc[Ind[[k]],])
  }
  ccr[j,] <- sum(ifelse(yhat>=0.5,"female","male") == y)/n
} 

ccr
```

    ##          [,1]
    ## [1,] 0.927399
