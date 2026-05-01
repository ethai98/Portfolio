# Apply variable and value labels into dataframe when using csv version of the cleaned WERN data. 
# If user uses the RDS version, there is no need to apply labels since the variable and value labels are retained in the RDS file. 
# 10/15/2022

devtools::install_github("beckerbenj/eatGADS")
library(eatGADS)
library(here)
library(readr)
library(tidyverse)
# see vignette: https://cran.r-project.org/web/packages/eatGADS/vignettes/import_raw.html

# import csv dataframe that does not have variable and value labeled.
wern_cleaned <- read_csv(here("WERN_cleaned_data", "wern_cleaned_v3.0.csv"))

# import dictionary for variable and value labels
wern_dictionary_val <- read_csv(here("WERN_cleaned_data","wern_dictionary_val_v3.0.csv"))
wern_dictionary_var <- read_csv(here("WERN_cleaned_data","wern_dictionary_var_v3.0.csv"))

# set up, change date into character variables (it's not allowed by the package)
wern_cleaned <- wern_cleaned %>%
  mutate(StartDate = as.character(StartDate),
         EndDate = as.character(EndDate),
         RecordedDate = as.character(RecordedDate)) %>%
  rename(Duration__in_seconds_ = "Duration (in seconds)")

# use import_raw to easily apply variable and value labels
wern_cleaned_gads_obj <- import_raw(df = wern_cleaned, varLabels = wern_dictionary_var, valLabels = wern_dictionary_val)

# use extractData() to arrive at data frame with variable labeled (excluding labeled value - not a feature of the package)
wern_cleaned_gads_obj_data <- extractData(wern_cleaned_gads_obj,  convertLabels = "numeric")

# check the labelling on a variable
wern_cleaned_gads_obj_data$pay_type_num

# to check the value labels use:
wern_cleaned_gads_obj_meta <- extractMeta(wern_cleaned_gads_obj)

# to check the meta data for specific variables use:
extractMeta(wern_cleaned_gads_obj, vars = c("rk_wgt_final"))


# Alternatively, the RDS file provides both variable and value labels. ----

wern_cleaned_rds<- readRDS(here("WERN_cleaned_data", "wern_cleaned_v3.0.rds"))

# check
wern_cleaned_rds$rk_wgt_final
