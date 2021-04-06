rm(list = ls())
library(readxl)
library(tidyverse)
library(survival)
library(survminer)
library(fastDummies)
library(data.table)

setwd('~/Studio/VO2max_prediction')
df <- fread("./Data/Survival_set/MF_general_eq_survival.csv")

"df <- df %>% 
  mutate_if(is.integer, as.factor)

df <- df %>% 
  mutate_if(is.character, as.factor)"


df$Smoke <- as.factor(df$Smoke)
df$Hypertension <- as.factor(df$Hypertension)
df$Hyperlipidemia <- as.factor(df$Hyperlipidemia)
#df$HTN_med <- as.factor(df$HTN_med)
df$Stroke <- as.factor(df$Stroke)
df$Angina <- as.factor(df$Angina)
df$MI <- as.factor(df$Asthma)
df$sex <- as.factor(df$sex)
df$MVPA <- as.factor(df$MVPA)
df$ALC <- as.factor(df$ALC)
df$Hepatatis <- as.factor(df$Hepatatis)
df$Diabetes <- as.factor(df$Diabetes)
df$Asthma <- as.factor(df$Asthma)
df$ALC <- as.factor(df$ALC)

surv_obejct <- Surv(time = df$delta_time, event=df$death)

############################################## BMI ###########################################################

##### Estimated CRF - BMI(ABRP)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABRP_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

##### Estimated CRF - BMI(ABR)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABR_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

##### Estimated CRF - BMI(ABP)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABP_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

############################################## BMI Tertile ###########################################################

##### Estimated CRF Tertile - BMI(ABRP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABRP_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Tertile - BMI(ABR)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABR_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)


##### Estimated CRF Tertile - BMI(ABP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABP_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

############################################## BMI Qualtile ###########################################################

##### Estimated CRF Qualtile - BMI(ABRP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABRP_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Qualtile - BMI(ABR)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABR_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Qualtile - BMI(ABP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + ABP_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)


############################################## Percentage Fat ###########################################################

##### Estimated CRF - Percentage Fat(ABRP)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APRP_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

##### Estimated CRF - Percentage Fat(ABR)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APR_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

##### Estimated CRF - Percentage Fat(ABP)
coxph_e <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APP_CRF, data = df, method = 'breslow')
ggforest(coxph_e, data= df)

############################################## Percentage Fat Tertile ###########################################################

##### Estimated CRF Tertile - Percentage Fat(APRP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APRP_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Tertile - Percentage Fat(APR)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APR_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)


##### Estimated CRF Tertile - Percentage Fat(APP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APP_CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

############################################## Percentage Fat Qualtile ###########################################################

##### Estimated CRF Qualtile - Percentage Fat(APRP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APRP_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Qualtile - Percentage Fat(APR)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APR_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)

##### Estimated CRF Qualtile - Percentage Fat(APP)
coxph_e_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + APP_CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_e_t, data= df)


############################################## CRF ###########################################################

##### Real CRF
coxph_r <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                   Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + CRF, data = df, method = 'breslow')
ggforest(coxph_r, data= df)

##### Real CRF Tertile
coxph_r_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + CRF_tertile , data = df, method = 'breslow')
ggforest(coxph_r_t, data= df)

##### Real CRF Qualtile
coxph_r_t <- coxph(formula = surv_obejct ~ AGE + sex + Smoke + ALC + BMI + MVPA + Diabetes + Hypertension + 
                     Hyperlipidemia + Hepatatis + max_heart_rate + `HDL-C` + MBP + CRF_qualtile, data = df, method = 'breslow')
ggforest(coxph_r_t, data= df)
