################################################
################################################
# Performance Evaluation of Predictive AI in   # 
# Medical Practice: Overview and Guidance      #
################################################
################################################

# Written by Ben Van Calster
# October 2024



########################################
# LOAD PACKAGES AND SET WORK DIRECTORY #
########################################

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# packages having multiple performance measures: 
# ModelMetrics, PerfMeas, MLeval, ROCR, yardstick, classifierplots, precrec, 
# runway, modEvA, mlr3measures

# Load packages
pacman::p_load(DescTools, ggplot2, psych, pROC, PRROC, yardstick, pROC, rms, 
               CalibrationCurves, dplyr, ResourceSelection, gtools, rmda, 
               dcurves, ROCR)


####################
# CUSTOM FUNCTIONS #
####################


# Simple function to calculate AUROC
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
fastAUC <- function(p, y) { 
  x1 = p[y==1] # vector with risk estimates for cases with Y=1
  n1 = length(x1) 
  x2 = p[y==0] # vector with risk estimates for cases with Y=0
  n2 = length(x2);
  r = rank(c(x1, x2)) # get ranks of risk estimates, first for x1 then for x2
  auc = (sum(r[1:n1]) - n1 * (n1 + 1) / 2) / n1 / n2 # formula to get AUROC
  return(auc)
}


# Function to calculate average precision (AP)
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
avgprec <- function(p,y){
  probsort = sort(p)
  prcpts = as.data.frame(matrix(NA,nrow = length(probsort),ncol=3))
  for (i in 1:length(probsort)){
    # Get recall/sensitivity for probability threshold that equals probsort[i]:
    prcpts[i,1]=sum(p[y==1] >= probsort[i]) / sum(y==1)
    # Get precision/PPV for probability threshold that equals probsort[i]:
    prcpts[i,2]=sum(p[y==1] >= probsort[i]) / sum(p >= probsort[i])
  }
  for (i in 1:length(probsort)){
    # for every probability threshold (sorted), multiply precision with
    # difference in recall with next threshold; for the last threshold, multiply
    # precision with recall (recall for higher thresholds is 0 by definition):
    prcpts[i,3] = ifelse(i==length(probsort),
                         prcpts[i,2]*(prcpts[i,1]),
                         prcpts[i,2]*(prcpts[i,1]-prcpts[i+1,1]))
  }
  # get sum of precision*(recall differences) per threshold:
  return(sum(prcpts[,3]))
}


# Function to get measures for discrimination performance
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
#  paucf: whether pAUROC is based on tolerable values for sensitivity or
#         specificity, i.e. paucf = c("se", "sp"); "se" is default
#  paucr: for pAUROC, the range of sed on tolerable values for paucf, default
#         is c(1, 0.8)
DiscPerfBin <- function(y, p, paucf = "se", paucr = c(1, .8)){
  # AUROC or c-statistic using custom function (cf above):
  cstat = fastAUC(p = p, y = y) 
  # AUPRC using pr.curve function from PRROC package:
  auprc = pr.curve(p[y==1], p[y==0], curve = T)[["auc.davis.goadrich"]]
  # Average Precision using custom function (cf above):
  ap = avgprec(p = p, y = y)
  # pAUROC using roc function from pROC package:
  pauc = roc(y, p, partial.auc = paucr, partial.auc.focus = paucf)$auc[1] 
  # Put all measures together and creat column names:
  discperf = as.data.frame(t(c(cstat, auprc, ap, pauc)))
  colnames(discperf) = c("AUROC/c statistic", "AUPRC", "AP", "pAUROC") 
  return(discperf)
}


# Function to get measures for calibration performance
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
#  flexcal: whether ECI and ICI are based on a smoothed calibration plot using 
#           loess, restricted cubic splines with 3 knots, or restricted cubic 
#           splines with 5 knots, i.e. flexcal = c("loess", "rcs3", "rcs5") 
#  ngr: number of groups for the ECE, default is 10
CalPerfBin <- function(y, p, flexcal, ngr = 10){
  # O:E ratio:
  oe = sum(y) / sum(p)
  # Calibration intercept by using logit(p) as offset in log reg analysis:
  int = summary(glm(y ~ 1, offset = qlogis(p), 
                    family = "binomial"))$coefficients[1, 1]
  # Calibration slope:
  sl = summary(glm(y ~ qlogis(p), family = "binomial"))$coefficients[2, 1]
  # Get observed proportions from flexible calibration plot using method
  # specified in flexcal:
  if (flexcal == "loess"){
    flc = predict(loess(y ~ p, degree = 2))
  }
  if (flexcal == "rcs5"){
    flc = predict(glm(y ~ rcs(qlogis(p), 5), family = "binomial"), 
                  type = "response")
  }
  if (flexcal == "rcs3"){
    flc = predict(glm(y ~ rcs(qlogis(p), 3), family = "binomial"), 
                  type = "response")
  }
  # ECI and ICI using observed proportions from flexible calibration plot:
  eci = mean((flc - p)^2) / mean((rep(mean(y), length(y)) - p)^2)
  ici = mean(abs(flc - p))
  # Use hoslem.test function (ResourceSelection package) to get results for all
  # ngr groups, which is then used to calculate ECE:
  hlt = hoslem.test(y, p, g = ngr)
  ece = sum(abs(hlt$expected[, 2] - hlt$observed[, 2]) / length(y))
  # Put measures together and create column names:
  calperf = as.data.frame(t(c(oe, int, sl, eci, ici, ece)))
  colnames(calperf) = c("O:E ratio", "Cal. intercept", "Cal. slope", "ECI", 
                        "ICI", "ECE") 
  return(calperf)
}


# Function to get all evaluated measures for overall performance
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
OvPerfBin <- function(y, p){
  # Loglikelihood (lli), loglikelihood for null model (ll0), and logloss (llo):
  lli = sum(dbinom(y, prob = p, size = 1, log = TRUE))
  ll0 = sum(dbinom(y, prob = rep(mean(y), length(y)), size = 1, log = TRUE))
  llo = -sum(dbinom(y, prob = p, size = 1, log = TRUE))
  # Brier score, Brier skill score (a.k.a. scaled Brier or IPA):
  br = mean((y - p)^2)
  bss = 1 - (mean((y - p)^2) / 
               mean((y - rep(mean(y), length(y)))^2) )
  # R-squared: McFadden (mfr2), Cox-Snell (csr2), Nagelkerke (nr2):
  mfr2 = 1 - (lli / ll0)
  csr2 = 1 - exp(2 * (ll0 - lli) / length(y))
  nr2 = csr2 / (1 - exp(2 * ll0 / length(y)))
  # Discrimination slope (ds) and MAPE:
  ds = mean(p[y==1]) - mean(p[y==0])
  mape = mean(abs(y - p))
  # Put measures together and create column names
  ovperf = as.data.frame(t(c(lli, llo, br, bss, mfr2, csr2, nr2, ds, mape)))
  colnames(ovperf) = c("Loglikelihood", "Logloss", "Brier", "Scaled Brier", 
                       "McFadden R2", "Cox-Snell R2", "Nagelkerke R2", 
                       "Discrimination slope", "MAPE") 
  return(ovperf)
}


# Function to calculate measures for classification performance
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
#  cut: decision threshold
ClassPerfBin <- function(y, p, cut){
  # Get number of true positives, false negatives, true negatives, 
  # false positives:
  TP = mean((p>=cut) * (y==1))
  FN = mean((p<cut) * (y==1))
  TN = mean((p<cut) * (y==0))
  FP = mean((p>=cut) * (y==0))
  # Get partial measures for classification: sensitivity/recall and specificity,
  # and positive predictive value/precision and negative predictive value:
  Sens = TP / (TP + FN)
  Spec = TN / (TN + FP)
  PPV = TP / (TP + FP)
  NPV = TN / (TN + FN)
  # Get Accuracy (Acc), Balanced accuracy (Bar), Youden index (You), and 
  # diagnostic odds ratio (DOR):
  Acc = TP + TN
  Bar = 0.5 * Sens + 0.5 * Spec
  You = Sens + Spec - 1
  DOR = (Sens / (1 - Spec)) / ((1 - Sens) / Spec)
  # Get kappa, which needs accuracy by chance (Acc_E):
  Acc_E = mean(y) * (TP + FP) + (1 - mean(y)) * (FN + TN)
  Kap = (Acc - Acc_E) / (1 - Acc_E)
  # Get F1 score and Matthew's Correlation Coefficient:
  F1 = 2 * ((PPV * Sens) / (PPV + Sens))
  MCC = (TP * TN - FP * FN) / 
        sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  # Put all measures together and create column names:
  classperf = as.data.frame(t(c(Acc, Bar, You, DOR, Kap, F1, MCC, Sens, Spec, 
                                PPV, NPV)))
  colnames(classperf) = c("Accuracy", "Balanced Accuracy", "Youden index", 
                          "DOR", "Kappa", "F1", "MCC", "Sensitivity/Recall", 
                          "Specificity", "PPV/precision", "NPV") 
  return(classperf)
}


# Function to calculate measures for utility performance
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
#  cut: decision threshold
#  costratio: cost of false negative / cost of false positive to be used for
#  Expected Cost (EC); this is related to the decision threshold used in Net
#  Benefit (NB): if you want NB for cut = 0.1, then costratio for EC is 
#  1/odds(cut) = (1-cut)/cut
UtilPerfBin <- function(y, p, cut, costratio){
  # Apply NB and SNB formulas:
  NB = mean((p>=cut) * (y==1)) - 
       (cut / (1-cut)) * mean((p>=cut) * (y==0))
  SNB = NB / mean(y)
  # to calculate EC, sort probability estimates:
  risksort = sort(p)
  # Then, for every possible threshold (based on sorted probabilities),
  # Calculate number of false negatives, number of false positives, and EC
  # based on FN, FP and the desired costratio:
  ecpts = as.data.frame(matrix(NA, nrow = length(risksort), ncol = 3))
  for (i in 1:length(risksort)){
    ecpts[i, 1] = sum(p[y==1] < risksort[i]) / sum(y==1)
    ecpts[i, 2] = sum(p[y==0] >= risksort[i]) / sum(y==0)
    ecpts[i, 3] = (ecpts[i, 1] * mean(y) * 
                  (costratio * (costratio>1) + 1 * (costratio<=1))) + 
                  (ecpts[i, 2] * (1-mean(y)) * 
                  (costratio * (costratio<1) + 1 * (costratio>=1)))
  }
  # the final EC is then the minimum EC over all possible thresholds:
  EC = min(ecpts[,3])
  # Get the probability estimate that gave the minimum EC:
  ECthreshold = risksort[which.min(ecpts[, 3])]
  # Put all measures together and create column names:
  utilperf = as.data.frame(t(c(NB, SNB, EC, ECthreshold)))
  colnames(utilperf) = c("Net benefit", "Standardized net benefit", 
                         "Expected cost (EC)", "Threshold for EC") 
  return(utilperf)
}


# Function to get data for a plot of EC
# Input: 
#  p: vector with risk estimates
#  y: vector with outcomes, i.e. 0 or 1
#  ncostfp: normalized costs of false positive to be used
#  for calculating EC values; normalized means both costs sum to one, such that
#  the normalized cost of a false negative is 1 minus the normalized cost of a
#  false positive
ecplotv <- function(y, p, ncostfp){
  # to calculate EC, sort probability estimates:
  risksort = sort(p)
  # then, per normalized cost of a false positive, find the minimum EC and the
  # probability threshold for which the minimum EC is achieved
  ec_res = data.frame(matrix(NA,2,length(ncostfp)))
  for (i in 1:length(ncostfp)){ 
    # For every possible threshold (based on sorted probabilities),
    # calculate number of false negatives, number of false positives, and EC
    # based on FN, FP and the desired normalized costs of a false positive:
    ecpts = as.data.frame(matrix(NA, nrow = length(risksort), ncol = 3))
    for (j in 1:length(risksort)){
      ecpts[j, 1] = sum(p[y==1] < risksort[j]) / sum(y==1)
      ecpts[j, 2] = sum(p[y==0] >= risksort[j]) / sum(y==0)
      ecpts[j, 3] = (ecpts[j, 1] * mean(y) * (1-ncostfp[i]) + 
                       (ecpts[j, 2] * (1-mean(y)) * ncostfp[i]))
    }
    # the final EC is then the minimum EC over all possible thresholds:
    ec_res[1,i] = min(ecpts[,3])
    # this is the probability threshold for which the minimum EC was obtained
    ec_res[2,i] = risksort[which.min(ecpts[, 3])]
    # ec_res[3,i] = mean(risksort[which.min(ecpts[, 3])],rev(risksort)[which.min(ecpts[, 3])])
  }
  return(ec_res)
}



##########################################################################
# LOAD DATA (transIOTA  study, final data complete cases (MCAR assumed)) #
##########################################################################

dcase=read.table(file = "https://raw.githubusercontent.com/benvancalster/PerfMeasuresOverview/refs/heads/main/data_case_study.txt",
                 sep = " ", header = TRUE, na.strings = c(""))

# Get factor version of the outcome, to be used in ggplot applications:
dcase$Out1 = factor(dcase$Outcome1, levels = c(0,1), 
                    labels = c("Benign", "Malignant"))

# Create updated risk estimates based on 'logistic recalibration':
dcase$pmalwou = predict(lrm(Outcome1 ~ qlogis(pmalwo), data = dcase), 
                        type = "fitted")



##############################
# DISCRIMINATION PERFORMANCE #
##############################

# GET DISCRIMINATION MEASURES USING DISCPERFBIN FUNCTION

DiscADNEX = DiscPerfBin(y = dcase$Outcome1, p = dcase$pmalwo)


# GET POINTS THAT MAKE UP THE ROC CURVE, PLOT IT LATER ON

# Sort probability estimates
adnexsort = sort(dcase$pmalwo)

# For every possible probability threshold (based on sorted probability
# estimates), calculate Sensitivity/recall and false positive rate (which is
# 1 minus specificity)
rocptsadnex = as.data.frame(matrix(NA, nrow = length(adnexsort), ncol = 2))
colnames(rocptsadnex) = c("Sens", "1-Spec")
for (i in 1:length(adnexsort)){
  rocptsadnex[i, 1] = sum(dcase$pmalwo[dcase$Outcome1==1] >= adnexsort[i]) / 
                      sum(dcase$Outcome1==1)
  rocptsadnex[i, 2] = sum(dcase$pmalwo[dcase$Outcome1==0] >= adnexsort[i]) / 
                      sum(dcase$Outcome1==0)
}

# Add a row with probability threshold 0 (sensitivity and false positive rate
# equal 1), and a row with probability threshold 1 (sensitivity and false
# positive rate equal 0):
rocptsadnex = rbind(c(1,1), rocptsadnex,c(0,0))


# GET POINTS FOR THE PRECISION RECALL CURVE, PLOT IT LATER

# Precision - Recall curve with linear interpolation (own code)
# In large datasets the interpolation is no big deal, although smart nonlinear 
# interpolation is the way to go (see Davis & Goadrich paper)
# In small datasets: why even bother?

# sort probability estimates
adnexsort = sort(dcase$pmalwo)

# For every possible probability threshold (based on sorted probability
# estimates), calculate Sensitivity/recall and PPV/precision
prcptsadnex = as.data.frame(matrix(NA, nrow = length(adnexsort), ncol = 2))
for (i in 1:length(adnexsort)){
  prcptsadnex[i,1] = sum(dcase$pmalwo[dcase$Outcome1==1]>=adnexsort[i])/ 
                     sum(dcase$Outcome1==1)
  prcptsadnex[i,2] = sum(dcase$pmalwo[dcase$Outcome1==1]>=adnexsort[i])/ 
                     sum(dcase$pmalwo >= adnexsort[i])
}


# PLOT WITH ROC IN PANEL A, PR CURVE IN PANEL B, pAUROC in panel C

# Prepare panels:
par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

# ROC curve:
plot(rocptsadnex[, 2], rocptsadnex[, 1], type = "l", lwd = 2, col = "black", 
     main = "A", xlab = "", ylab = "", cex.axis = 0.8)

legend("bottomright", inset = .02, legend = c("AUROC 0.91"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)
title(xlab="1 - Specificity", line=2.2, cex.lab = 1.1)

# PR curve:
plot(prcptsadnex[, 1], prcptsadnex[, 2], type = "l", lwd = 2, col = "black", 
     main="B", xlab = "", ylab = "", ylim = c(0,1), cex.axis = 0.8)
abline(h=mean(dcase$Outcome1), lwd = 1, col = "gray")

legend("bottomright", inset = .02, legend = c("AUPRC 0.89"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Precision / PPV", line=2.2, cex.lab = 1.1)
title(xlab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)

# pAUROC illustration:
fpr <- rocptsadnex[, 2]
tpr <- rocptsadnex[, 1]
tpr_threshold <- 0.8  # Set threshold value for tpr
plot(rocptsadnex[, 2], rocptsadnex[, 1], type = "l", lwd = 2, col = "black", 
     main = "C", xlab = "", ylab = "", cex.axis = 0.8)
# add rectangle to hide the intolerable part of the ROC plot:
# polygon(c(-0.01, 1.01, 1.01, -0.01), c(0, 0, 0.8, 0.8), col = "lightgray", 
#         border = NA)
polygon(c(fpr+0.003, rev(fpr)), 
        c(ifelse(tpr <= tpr_threshold, tpr, 0.8), rep(0, length(fpr))), 
        col = "gray", border = NA)

text(x = 0.5, y = 0.4, "Not used to calculate pAUROC", adj = 0.5, cex=1)
legend("bottomright", inset = .02, legend = c("pAUROC 0.14"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)
title(xlab="1 - Specificity", line=2.2, cex.lab = 1.1)

# Close panel:
par(mfrow = c(1,1))

# PR curve using function from PRROC package (not used):
# plot(prcadnex[["curve"]][,1], prcadnex[["curve"]][,2], type="l", col="black", 
#      xlab="True Positive Rate / Sensitivity / Recall", 
#      ylab = "Positive Predictive Value / Precision", ylim=c(0,1), cex.lab=1.2)



###########################
# CALIBRATION PERFORMANCE #
###########################


# CALIBRATION IN THE MODERATE SENSE: THE CALIBRATION PLOT

# We use the val.prob.ci.2 function from the CalibrationCurves package
# Using loess smoothing:

dev.off()
val.prob.ci.2(dcase$pmalwo, dcase$Outcome1, CL.smooth = "fill", 
              logistic.cal = F, g = 10, col.ideal = "gray", lty.ideal = 2, 
              lwd.ideal = 2, dostats = F, col.log = "gray", lty.log = 1, 
              lwd.log = 2.5, col.smooth = "black", lty.smooth = 1, 
              lwd.smooth = 2.5, xlab="Estimated probability", cex.lab = 1.3)


# Code just to get the (estimated) observed proportions based on the same loess
# fit as above:
loessadnex = predict(loess(Outcome1 ~ pmalwo, data = dcase, degree = 2))


# OBTAIN CALIBRATION MEASURES

CalADNEX = CalPerfBin(y = dcase$Outcome1, p = dcase$pmalwo, flexcal = "loess")


# CALIBRATION IN THE STRONG SENSE: looking at subgroups goes in that direction

# Calibration plots by menopausal status

# dev.off()
# val.prob.ci.2(dcase$pmalwo[dcase$Postmenopausal2=="no"], 
#               dcase$Outcome1[dcase$Postmenopausal2=="no"],
#               CL.smooth = "fill", logistic.cal = F, g = 10, col.ideal = "gray", 
#               lty.ideal = 2, lwd.ideal = 2, dostats = F, col.log = "gray", 
#               lty.log = 1, lwd.log = 2.5, col.smooth = "black", lty.smooth = 1, 
#               lwd.smooth = 2.5, xlab="Estimated probability", cex.lab = 1.3, 
#               cex.leg = 0.8)
# 
# val.prob.ci.2(dcase$pmalwo[dcase$Postmenopausal2=="yes"], 
#               dcase$Outcome1[dcase$Postmenopausal2=="yes"], 
#               CL.smooth = "fill", logistic.cal = F, g = 10, col.ideal = "gray", 
#               lty.ideal = 2, lwd.ideal = 2, dostats = F, col.log = "gray", 
#               lty.log = 1, lwd.log = 2.5, col.smooth = "black", lty.smooth = 1, 
#               lwd.smooth = 2.5, xlab="Estimated probability", cex.lab = 1.3, 
#               cex.leg = 0.8)



#######################
# OVERALL PERFORMANCE #
#######################

# PLOTS TO SHOW THE DISTRIBUTION OF THE ESTIMATED PROBABILITIES

# Violin plot with jittered dots:
p <- ggplot(dcase, aes(x = Out1, y = pmalwo, fill = Out1)) +
  geom_violin(fill="white") + 
  geom_jitter(cex=0.7, shape=16, position=position_jitter(0.1)) +
  theme(legend.position = "none", 
        axis.text.x = element_text(size = 14, face="bold", color="black"), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 14, face="bold")) +
        ylab("Estimated risk of malignancy")+ theme_classic()
p

# Box plot with jittered dots (not used):
# p <- ggplot(dcase, aes(x = Out1, y = pmalwo, fill = Out1)) +
#   geom_boxplot(width=0.5, fill="white", outlier.shape=NA) +
#   geom_jitter(cex=0.7, shape=16, position=position_jitter(0.1)) +
#   theme(legend.position = "none", axis.text = element_text(size = 12),
#         axis.title=element_text(size = 14, face="bold")) +
#   xlab("Outcome") +
#   ylab("Estimated risk of malignancy")
# p


# OVERALL PERFORMANCE MEASURES

OvADNEX = OvPerfBin(y = dcase$Outcome1, p = dcase$pmalwo)


##############################
# CLASSIFICATION PERFORMANCE #
##############################

# CLASSIFICATION PLOT

# sort probability estimates
adnexsort = sort(dcase$pmalwo)

# For every possible probability threshold (based on sorted probability
# estimates), calculate Sensitivity/recall, Specificity, PPV/precision, and NPV
clsptsadnex = as.data.frame(matrix(NA, nrow = length(adnexsort), ncol = 4))
for (i in 1:length(adnexsort)){
  # Sensitivity/recall
  clsptsadnex[i, 1] = sum(dcase$pmalwo[dcase$Outcome1==1] >= adnexsort[i]) / 
                      sum(dcase$Outcome1==1)
  # Specificity
  clsptsadnex[i, 2] = sum(dcase$pmalwo[dcase$Outcome1==0] < adnexsort[i]) / 
                      sum(dcase$Outcome1==0)
  # PPV/precision
  clsptsadnex[i, 3] = sum(dcase$Outcome1[dcase$pmalwo>=adnexsort[i]] == 1) / 
                      sum(dcase$pmalwo>=adnexsort[i])
  # NPV
  clsptsadnex[i, 4] = sum(dcase$Outcome1[dcase$pmalwo<adnexsort[i]] == 0) / 
                      sum(dcase$pmalwo<adnexsort[i])
}

# Prepare panels:
par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

# Classfication plot with sensitivity and specificity
plot(adnexsort, clsptsadnex[, 1], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "A", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexsort, clsptsadnex[, 2], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, legend = c("Sensitivity", "Specificity"), 
       col = c("black","black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

# Classfication plot with PPV and NPV
plot(adnexsort, clsptsadnex[, 3], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "B", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexsort, clsptsadnex[, 4], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, legend = c("PPV", "NPV"), 
       col = c("black", "black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

# Classfication plot with precision and recall
plot(adnexsort, clsptsadnex[, 3], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "C", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexsort, clsptsadnex[, 1], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, legend = c("PPV / Precision", "Sensitivity / Recall"), 
       col = c("black", "black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

# Close panel:
par(mfrow = c(1,1))


# OBTAIN CLASSIFICATION MEASURES

ClassADNEX = ClassPerfBin(y = dcase$Outcome1, p = dcase$pmalwo, cut = 0.1)



####################
# CLINICAL UTILITY #
####################

# NET BENEFIT AND DCA

# Code to generate decision curves using dcurves and rmda R packages
# What we did not use in the paper is commented out

# dcurves package
# The curve is plotted over the full [0,1] range for didactic purposes
# dcaadnex0 = dca(Outcome1 ~ pmalwo,
#             data = dcase,
#             # list of decision thresholds to use:
#             thresholds = seq(0, 1, by = 0.01), 
#             label = list(pmalwo = "ADNEX")) 
#            %>% plot(smooth = T) # This smooths the curve!

# Decision curve using rmda package (no smoothing)
# Again, curve plotted over the full [0,1] range for didactic purposes
dcaadnex <- decision_curve(Outcome1~pmalwo,
                           data = dcase,
                           # pmalwo has risk estimates from existing model:
                           fitted.risk = TRUE,
                           # grid of decision thresholds to use:
                           thresholds = seq(0, 1, by = .01),
                           # do not calculate confidence intervals:
                           confidence.intervals = F)
# plot_decision_curve( list(dcaadnex), 
#                      curve.names = c("ADNEX"),
#                      col = c("blue"), lty = c(1), 
#                      xlim = c(0,1), ylim = c(0, 0.5),
#                      # do not show confidence intervals:
#                      confidence.intervals = F,
#                      # show NB, not standardized NB:
#                      standardize = FALSE, 
#                      legend.position = "topright",
#                      xlab = "Decision threshold",
#                      # Extra x-axis with harm:benefit ratio:
#                      cost.benefit.xlab = "Harm to benefit ratio") 

# Get smoothed net benefit using central moving average
# First, get the NB values from the dcaadnex object
dcasm=as.data.frame(dcaadnex[[1]][1:length(seq(0, 1, by = .01)), 6])
colnames(dcasm) = c("orig")
# for every decision threshold except 0 and 1, get a central moving average
# For thresholds 0.01 and 0.99 (right next to 0 and 1), the moving average
# involve the previous and the next threshold; for other thresholds, the moving
# average involves the 2 prevous and two next thresholds
for (i in 1:length(dcasm$orig)){
  dcasm$sm[i] = ifelse(i %in% c(1,length(dcasm)),
                       dcasm$orig[i],
                       ifelse(i %in% c(2,length(dcasm$orig)-1),
                              mean(dcasm$orig[c((i-1):(i+1))]),
                              mean(dcasm$orig[c((i-2):(i+2))]))
  )
}


# GET DATA TO PLOT EC BY NORMALIZED COST OF A FALSE POSITIVE

# EC values for the model
ecmodel = ecplotv(y = dcase$Outcome1, p = dcase$pmalwo, 
                  ncostfp = c(1:99)/100)

# EC values for treat all and treat none
ncostfp = c(1:99)/100
ecTA = ncostfp * (mean(dcase$Outcome1) / (1 - mean(dcase$Outcome1))) 
ecTN = (1-ncostfp) * ((1 - mean(dcase$Outcome1)) / mean(dcase$Outcome1))


# CODE TO PLOT COST CURVES (not used)

# use prediction function from ROCR package to put risk estimates in right
# right format to move on with calculations for cost curves
# ccadnex0 = prediction(dcase$pmalwo, dcase$Outcome1)

# use performance function from ROCR package to get calculations for cost curves
# ccadnex = performance(ccadnex0, measure = "ecost") 

# extract x-axis values (ccadnex_pc) and y-axis values (ccadnex_nec) for the
# cost curve
# ccadnex_pc = ccadnex@x.values[[1]]
# ccadnex_nec = ccadnex@y.values[[1]]


# PLOT WITH DECISION CURVES USING NB, SNB OR EC

grid = seq(0, 1, by = .01)
ngrid = length(grid)

# Prepare panels:
par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

# Decision curve with NB and smoothed version
plot(grid, dcasm$sm, type = "l", col = "darkgray", lwd = 5, lty = 3, main = "A", 
     xlim = c(0,1), ylim = c(0,0.5), xlab = "", ylab = "", cex.axis=0.8)
lines(grid, dcaadnex[[1]][1:ngrid, 6], col = "black", lwd = 2)
lines(grid, dcaadnex[[1]][(ngrid + 1):(2 * ngrid),6], col = "gray", lwd = 2)
lines(grid, rep(0, ngrid), col = "gray", lwd = 1)

legend("topright", inset = .02, col = c("black", "gray", "gray", "gray"), 
       lty = c(1, 3, 1, 1), lwd = c(2, 5, 2, 1), cex = 0.9,
       legend = c("ADNEX", "ADNEX smoothed", "All", "None"), seg.len = 3)
title(ylab="Net benefit", line=2.2, cex.lab = 1.1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)

# Decision curve with standardized NB
plot(grid, dcaadnex[[1]][1:ngrid, 7], type = "l", col = "black", lwd = 2, 
     main = "B", xlim = c(0,1), ylim = c(0,1), xlab = "", ylab = "", 
     cex.axis=0.8)
lines(grid, dcaadnex[[1]][(ngrid + 1):(2 * ngrid),7], col = "gray", lwd = 2)
lines(grid, rep(0, ngrid), col = "gray", lwd = 1)
legend("topright", inset = .02, legend = c("ADNEX", "All", "None"), 
       col = c("black", "gray", "gray"), lty = c(1, 1, 1), lwd = c(2, 2, 1), 
       cex = 1)
title(ylab="Standardized net benefit", line=2.2, cex.lab = 1.1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)

# EC curve
plot(1-ncostfp, ecmodel[1,], type = "l", col = "black", lwd = 2, main = "C", 
     xlim = c(0,1), ylim = c(0,0.5), xlab = "", ylab = "", 
     cex.axis=0.8)
lines(1-ncostfp,ecTA, col="gray",lwd=2)
lines(1-ncostfp,ecTN, col="gray",lwd=1)
legend("topright", inset = .02, legend = c("ADNEX", "All", "None"), 
       col = c("black", "gray", "gray"), lty = c(1, 1, 1), lwd = c(2, 2, 1), 
       cex = 1)
title(ylab="Expected cost", line=2.2, cex.lab = 1.1)
title(xlab="Normalized cost of false negative", line=2.2, cex.lab = 1.1)

# Cost curve
# plot(ccadnex_pc, ccadnex_nec, type = "l", lwd = 2, lty = 1, col = "black", 
#      main = "C", xlab = "", ylab = "", cex.lab = 1.2, cex.axis=0.8, 
#      xlim = c(0, 1), ylim = c(0, 0.55))
# lines(c(0, .5), c(0, .5), lwd = 2, lty = 1, col = "gray")
# lines(c(.5, 1), c(.5, 0), lwd = 2, lty = 1, col = "gray")
# legend("topright", inset = .02, legend = c("ADNEX", "Default"), 
#        col = c("black", "gray"), lty = c(1, 1), lwd = c(2, 2), cex = 1)
# title(ylab="Normalized expected cost", line=2.2, cex.lab = 1.1)
# title(xlab="PC(+) - Probability*Cost", line=2.2, cex.lab = 1.1)

# Close panel:
par(mfrow = c(1,1))


# OBTAIN UTILITY MEASURES

UtilADNEX = UtilPerfBin(y = dcase$Outcome1, p = dcase$pmalwo, cut = 0.1, 
                        costratio = 9)



####################################################################
# BOOTSTRAPPING FOR CONFIDENCE INTERVALS, CORRELATIONS, HISTOGRAMS #
####################################################################

# Set seed
set.seed(2345)

# Number of bootstrap samples
nboot = 1000

# Bootstrapping:
resadnex = as.data.frame(matrix(NA, nrow = nboot, ncol = 34)) 
for (bootnr in 1:nboot){
  bootdata <- dcase[sample(row.names(dcase), replace = TRUE),]
  resadnex[bootnr, c(1:9)] = OvPerfBin(y = bootdata$Outcome1, 
                                       p = bootdata$pmalwo)
  resadnex[bootnr, c(10:13)] = DiscPerfBin(y = bootdata$Outcome1, 
                                           p = bootdata$pmalwo)
  resadnex[bootnr, c(14:19)] = CalPerfBin(y = bootdata$Outcome1, 
                                          p = bootdata$pmalwo, 
                                          flexcal = "loess")
  resadnex[bootnr, c(20:30)] = ClassPerfBin(y = bootdata$Outcome1, 
                                            p = bootdata$pmalwo, cut = 0.1)
  resadnex[bootnr, c(31:34)] = UtilPerfBin(y = bootdata$Outcome1, 
                                           p = bootdata$pmalwo, 
                                           cut = 0.1, costratio = 9)
}
colnames(resadnex) = c("Loglik", "Logloss", "Brier", "Scaled B", "McF R2", 
                       "C-S R2", "Nag R2", "DS", "MAPE", "AUROC", "AUPRC", "AP",
                       "pAUROC", "O:E", "Cint", "Cslope", "ECI", "ICI", "ECE", 
                       "Acc", "BA", "Youden", "DOR", "Kappa", "F1", "MCC", 
                       "Sens", "Spec", "PPV", "NPV", "NB", "SNB", "EC", "ECt")


# Simple percentile CIs

pctci = as.data.frame(matrix(NA, ncol = 2, nrow = 34))
for (i in 1:34){
  pctci[i, 1:2] = quantile(resadnex[, i], c(0.025, 0.975))
}

bootci = format(round(cbind(c(t(OvADNEX), t(DiscADNEX), t(CalADNEX), 
                              t(ClassADNEX), t(UtilADNEX)),pctci), digits = 3) , 
                scientific = FALSE)
colnames(bootci) = c("Point estimate", "LCL", "UCL")
rownames(bootci) = c(colnames(OvADNEX), colnames(DiscADNEX), colnames(CalADNEX),
                     colnames(ClassADNEX), colnames(UtilADNEX))



######################
# RECALIBRATED MODEL #
######################

# ALL CALCULATIONS ARE DONE AS ABOVE #
######################################


# DISCRIMINATION

# measures for discrimination performance

DiscADNEXu = DiscPerfBin(y = dcase$Outcome1, p = dcase$pmalwo, 
                         paucf = "se", paucr = c(1, .8))

# get points for ROC curve

adnexusort = sort(dcase$pmalwou)
rocptsadnexu = as.data.frame(matrix(NA, nrow = length(adnexusort), ncol = 2))
for (i in 1:length(adnexusort)){
  rocptsadnexu[i, 1] = sum(dcase$pmalwou[dcase$Outcome1==1] >= adnexusort[i]) / 
                       sum(dcase$Outcome1==1)
  rocptsadnexu[i, 2] = sum(dcase$pmalwou[dcase$Outcome1==0] >= adnexusort[i]) / 
                       sum(dcase$Outcome1==0)
}

# get points for PR curve

prcptsadnexu = as.data.frame(matrix(NA, nrow = length(adnexusort), ncol = 2))
for (i in 1:length(adnexusort)){
  prcptsadnexu[i, 1] = sum(dcase$pmalwou[dcase$Outcome1==1] >= adnexusort[i]) / 
                       sum(dcase$Outcome1==1)
  prcptsadnexu[i, 2] = sum(dcase$pmalwou[dcase$Outcome1==1] >= adnexusort[i]) / 
                       sum(dcase$pmalwou >= adnexusort[i])
}


# Plot with ROC, PR curve, and pAUROC

par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

plot(rocptsadnexu[, 2], rocptsadnexu[, 1], type = "l", lwd = 2, col = "black", 
     main = "A", xlab = "", ylab = "", cex.axis = 0.8)

legend("bottomright", inset = .02, legend = c("AUROC 0.91"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)
title(xlab="1 - Specificity", line=2.2, cex.lab = 1.1)

plot(prcptsadnexu[, 1], prcptsadnexu[, 2], type = "l", lwd = 2, col = "black", 
     main="B", xlab = "", ylab = "", ylim = c(0,1), cex.axis = 0.8)
abline(h=mean(dcase$Outcome1), lwd = 1, col = "gray")

legend("bottomright", inset = .02, legend = c("AUPRC 0.89"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Precision / PPV", line=2.2, cex.lab = 1.1)
title(xlab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)

plot(rocptsadnexu[, 2], rocptsadnexu[, 1], type = "l", lwd = 2, col = "black", 
     main = "C", xlab = "", ylab = "", cex.axis = 0.8)
# add rectangle to hide the intolerable part of the ROC plot:
polygon(c(-0.01, 1.01, 1.01, -0.01), c(0, 0, 0.8, 0.8),
        col = "lightgray", border = NA)

text(x = 0.5, y = 0.4, "Not used to calculate pAUROC", adj = 0.5, cex=1)
legend("bottomright", inset = .02, legend = c("pAUROC 0.14"), col = c("black"), 
       lty = c(1), lwd = c(2), cex = 1)
title(ylab="Sensitivity / Recall", line=2.2, cex.lab = 1.1)
title(xlab="1 - Specificity", line=2.2, cex.lab = 1.1)

par(mfrow = c(1,1))


# CALIBRATION

# Calibration plot

dev.off()
val.prob.ci.2(dcase$pmalwou, dcase$Outcome1, CL.smooth = "fill", 
              logistic.cal = F, g = 10, col.ideal = "gray", lty.ideal = 2, 
              lwd.ideal = 2, dostats = F, col.log = "gray", lty.log = 1, 
              lwd.log = 2.5, col.smooth = "black", lty.smooth = 1, 
              lwd.smooth = 2.5, xlab="Estimated probability", cex.lab = 1.3)

# measures for calibration performance

CalADNEXu = CalPerfBin(y = dcase$Outcome1, p = dcase$pmalwou, 
                       flexcal = "loess")


# OVERALL PERFORMANCE

# Violin plot with jittered dots:

p <- ggplot(dcase, aes(x = Out1, y = pmalwou, fill = Out1)) +
  geom_violin(fill="white") + 
  geom_jitter(cex=0.7, shape=16, position=position_jitter(0.1)) +
  theme(legend.position = "none", 
        axis.text.x = element_text(size = 14, face="bold", color="black"), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 14, face="bold")) +
  ylab("Estimated risk of malignancy")+ theme_classic()
p

# measures for overall classification

OvADNEXu = OvPerfBin(y = dcase$Outcome1, p = dcase$pmalwou)


# CLASSIFICATION

# Classification plot (as above)

adnexusort = sort(dcase$pmalwou)

clsptsadnexu = as.data.frame(matrix(NA, nrow = length(adnexusort), ncol = 4))
for (i in 1:length(adnexusort)){
    clsptsadnexu[i, 1] = sum(dcase$pmalwou[dcase$Outcome1==1] >= adnexusort[i])/ 
      sum(dcase$Outcome1==1)
    clsptsadnexu[i, 2] = sum(dcase$pmalwou[dcase$Outcome1==0] < adnexusort[i]) / 
      sum(dcase$Outcome1==0)
    clsptsadnexu[i, 3] = sum(dcase$Outcome1[dcase$pmalwou>=adnexusort[i]] == 1)/ 
      sum(dcase$pmalwou>=adnexusort[i])
    clsptsadnexu[i, 4] = sum(dcase$Outcome1[dcase$pmalwou<adnexusort[i]] == 0) / 
      sum(dcase$pmalwou<adnexusort[i])
  }  

par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

plot(adnexusort, clsptsadnexu[, 1], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "A", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexusort, clsptsadnexu[, 2], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, legend = c("Sensitivity", "Specificity"), 
       col = c("black","black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

plot(adnexusort, clsptsadnexu[, 3], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "B", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexusort, clsptsadnexu[, 4], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, legend = c("PPV", "NPV"), 
       col = c("black", "black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

plot(adnexusort, clsptsadnexu[, 3], type = "l", lwd = 2, lty = 1, col = "black", 
     main = "C", xlab = "", ylab = "", cex.axis=0.8, xlim = c(0,1), 
     ylim = c(0,1))
lines(adnexusort, clsptsadnexu[, 1], lwd = 2, lty = 2, col = "black")

legend("bottom", inset = .02, 
       legend = c("PPV / Precision", "Sensitivity / Recall"), 
       col = c("black", "black"), lty = c(1,2), lwd = c(2,2), cex = 1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)
title(ylab="Proportion", line=2.2, cex.lab = 1.1)

par(mfrow = c(1,1))

# measures for classification

ClassADNEXu = ClassPerfBin(y = dcase$Outcome1, p = dcase$pmalwou, cut = 0.1)


# CLINICAL UTILITY

# get data for decision curve, smoothed decision curve, and cost curve

dcaadnexu <- decision_curve(Outcome1~pmalwou,
                           data = dcase,
                           fitted.risk = TRUE, 
                           thresholds = seq(0, 1, by = .01), 
                           confidence.intervals = F)

dcasmu=as.data.frame(dcaadnexu[[1]][1:length(seq(0, 1, by = .01)), 6])
colnames(dcasmu) = c("orig")
for (i in 1:length(dcasmu$orig)){
  dcasmu$sm[i] = ifelse(i %in% c(1,length(dcasmu)),
                       dcasmu$orig[i],
                       ifelse(i %in% c(2,length(dcasmu$orig)-1),
                              mean(dcasmu$orig[c((i-1):(i+1))]),
                              mean(dcasmu$orig[c((i-2):(i+2))]))
  )
}

# EC values for the model
ecmodelu = ecplotv(y = dcase$Outcome1, p = dcase$pmalwou, 
                  ncostfp = c(1:99)/100)

# EC values for treat all and treat none
ncostfp = c(1:99)/100
ecTAu = ncostfp * (mean(dcase$Outcome1) / (1 - mean(dcase$Outcome1))) 
ecTNu = (1-ncostfp) * ((1 - mean(dcase$Outcome1)) / mean(dcase$Outcome1))

# Decision curves using NB, SNB, or EC

grid = seq(0, 1, by = .01)
ngrid = length(grid)

# Prepare panels:
par(mfrow = c(2,2))
par(mar = c(3.7, 4, 2, 2)) # Set the margin on all sides to 2

# Decision curve with NB and smoothed version
plot(grid, dcasmu$sm, type = "l", col = "darkgray", lwd = 5, lty = 3, main = "A", 
     xlim = c(0,1), ylim = c(0,0.5), xlab = "", ylab = "", cex.axis=0.8)
lines(grid, dcaadnexu[[1]][1:ngrid, 6], col = "black", lwd = 2)
lines(grid, dcaadnexu[[1]][(ngrid + 1):(2 * ngrid),6], col = "gray", lwd = 2)
lines(grid, rep(0, ngrid), col = "gray", lwd = 1)

legend("topright", inset = .02, col = c("black", "gray", "gray", "gray"), 
       lty = c(1, 3, 1, 1), lwd = c(2, 5, 2, 1), cex = 0.9,
       legend = c("ADNEX", "ADNEX smoothed", "All", "None"), seg.len = 3)
title(ylab="Net benefit", line=2.2, cex.lab = 1.1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)

# Decision curve with standardized NB
plot(grid, dcaadnexu[[1]][1:ngrid, 7], type = "l", col = "black", lwd = 2, 
     main = "B", xlim = c(0,1), ylim = c(0,1), xlab = "", ylab = "", 
     cex.axis=0.8)
lines(grid, dcaadnexu[[1]][(ngrid + 1):(2 * ngrid),7], col = "gray", lwd = 2)
lines(grid, rep(0, ngrid), col = "gray", lwd = 1)
legend("topright", inset = .02, legend = c("ADNEX", "All", "None"), 
       col = c("black", "gray", "gray"), lty = c(1, 1, 1), lwd = c(2, 2, 1), 
       cex = 1)
title(ylab="Standardized net benefit", line=2.2, cex.lab = 1.1)
title(xlab="Decision threshold", line=2.2, cex.lab = 1.1)

# EC curve
plot(1-ncostfp, ecmodelu[1,], type = "l", col = "black", lwd = 2, main = "C", 
     xlim = c(0,1), ylim = c(0,0.5), xlab = "", ylab = "", 
     cex.axis=0.8)
lines(1-ncostfp,ecTAu, col="gray",lwd=2)
lines(1-ncostfp,ecTNu, col="gray",lwd=1)
legend("topright", inset = .02, legend = c("ADNEX", "All", "None"), 
       col = c("black", "gray", "gray"), lty = c(1, 1, 1), lwd = c(2, 2, 1), 
       cex = 1)
title(ylab="Expected cost", line=2.2, cex.lab = 1.1)
title(xlab="Normalized cost of false negative", line=2.2, cex.lab = 1.1)

# Close panel:
par(mfrow = c(1,1))

# measures for utility

UtilADNEXu = UtilPerfBin(y = dcase$Outcome1, p = dcase$pmalwou, cut = 0.1, 
                         costratio = 9)


# BOOTSTRAPPING FOR CONFIDENCE INTERVALS, CORRELATIONS, HISTOGRAMS

# Prepare the bootstrapping

set.seed(2345)
nboot = 1000

# do the bootstrapping

resadnexu = as.data.frame(matrix(NA, nrow = nboot, ncol = 34)) 
for (bootnr in 1:nboot){
  bootdata <- dcase[sample(row.names(dcase), replace = TRUE),]
  resadnexu[bootnr,c(1:9)] = OvPerfBin(y = bootdata$Outcome1, 
                                       p = bootdata$pmalwou)
  resadnexu[bootnr,c(10:13)] = DiscPerfBin(y = bootdata$Outcome1, 
                                           p = bootdata$pmalwou)
  resadnexu[bootnr,c(14:19)] = CalPerfBin(y = bootdata$Outcome1, 
                                          p = bootdata$pmalwou, 
                                          flexcal = "loess")
  resadnexu[bootnr,c(20:30)] = ClassPerfBin(y = bootdata$Outcome1, 
                                            p = bootdata$pmalwou, cut = 0.1)
  resadnexu[bootnr,c(31:34)] = UtilPerfBin(y = bootdata$Outcome1, 
                                           p = bootdata$pmalwou, 
                                           cut = 0.1, 
                                           costratio = 9)
}

# Simple percentile CIs

pctciu = as.data.frame(matrix(NA, ncol = 2, nrow = 34))
for (i in 1:34){
  pctciu[i, 1:2] = quantile(resadnexu[,i], c(0.025, 0.975))
}

bootciu = format(round(cbind(c(t(OvADNEXu), t(DiscADNEXu), t(CalADNEXu), 
                               t(ClassADNEXu), t(UtilADNEXu)), pctciu), 
                       digits = 3) , scientific = FALSE)
colnames(bootciu) = c("Point estimate", "LCL", "UCL")
rownames(bootciu) = c(colnames(OvADNEXu), colnames(DiscADNEXu), 
                      colnames(CalADNEXu), colnames(ClassADNEXu), 
                      colnames(UtilADNEXu))


#########
## END ##
#########