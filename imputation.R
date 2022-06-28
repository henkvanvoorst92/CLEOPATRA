library("clipr")
library('MASS')
library(writexl)
library(rms)
library(Hmisc)
library(tidyr)
library(dplyr)
library("PerformanceAnalytics")
library(tableone)
library('readxl')
library(psych)
library(mice)
library(mitools)

ds <- read_excel('E:/CLEOPATRA/unimputed_database.xlsx', na="NA")

ds <- ds %>%
  mutate(
    r_sex = as.factor(r_sex),
    r_r_sidestroke = as.factor(r_sidestroke),
    bl_occloc = as.factor(bl_occloc),
    bl_collaterals = ordered(bl_collaterals),
    bl_hist_premrs = ordered(bl_hist_premrs),
    ivt_given = as.factor(ivt_given),
    mrs_def = ordered(mrs_def),
    iat_post_etici = ordered(iat_post_etici)
  )


#options(datadist = NULL)
#dist <- datadist(ds)
#options(datadist = "dist")

set.seed(15)
ds_imp <- aregImpute(~ core_vol + r_age + bl_aspects_sum + bl_nihss_sum + 
                       t_otg + r_sex + r_sidestroke + bl_occloc +
                       bl_collaterals + bl_hist_premrs + ivt_given +
                       mrs_def + iat_post_etici,
                     data=ds, n.impute=1, nk=0)#, tlinear=FALSE)

#fetch single datase
completed <- ds
imputed <- impute.transcan(ds_imp, imputation=1, data=ds, list.out=TRUE,
                           pr=FALSE, check=FALSE)
completed[names(imputed)] <- imputed
write_clip(completed)


