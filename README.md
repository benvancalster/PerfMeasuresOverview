# PerfMeasuresOverview
[R code](https://github.com/benvancalster/PerfMeasuresOverview/blob/main/PerfMeasuresOverview.R) and [Python](https://github.com/benvancalster/PerfMeasuresOverview/blob/main/PerfMeasuresPython.py) code and [predictions for the case study](https://github.com/benvancalster/PerfMeasuresOverview/blob/main/data_case_study.txt) from Van Calster et al (Performance evaluation of predictive AI in Medical Practice: Overview and Guidance)

This paper was developed under the wings of the STRATOS (STRengthening Analytical Thinking for Observational Studies) consortium. It reviews 32 performance measures, and provides recommendations on choosing performance measures.
The context is a clinical risk prediction model (broadly referred to as Predictive AI) that is developed with the aim of being used in clinical/medical practice.

The measures are illustrated with a case study on the external validation of a model to estimate the risk of malignancy in patients with an ovarian tumor that was selected for surgery (cf Landolfo et al, Br J Cancer 2024).
In this GitHub project, we share the R code that we used to calculate measures and create plots. In addition, we share risk estimates and outcomes (1-malignant, 0-benign) for the patients in the study. No other patient data is provided to preserve anonymity.
