# WERN Survey Experiment

library(tidyverse)
library(cobalt)
library(vtable)
library(fixest)
theme_set(theme_bw())
library(MASS)

wern_analysis_v3.0 <- readRDS("~/Documents/research/Portfolio/A3_WERN/WERN_analysis_data/wern_analysis_v3.0.rds")


# balanced table

# treatment
# vignette_coworkertreat
# vignette_neighbortreat


wern_treatment_df <- wern_analysis_v3.0 %>%
  select(starts_with("vignette_coworkertreat") & ends_with("num"), vignette_coworkertreat,  
         starts_with("vignette_control") & ends_with("num"), vignette_control, 
         starts_with("vignette_neighbortreat") & ends_with("num"), vignette_neighbortreat, 
         # demographics
         age_num, earnings_num, gender_num, hh_05_num, hh_618_num, hh_income_num, hh_non_eng_num, highest_degree_num, homeowner_num, pol_activity_count,
         pol_ideo7_num, pol_pid3_num, pol_register_num, pol_turnout2020_num, pol_vote2020_num, urban_num, wage_num, 
         race_num, state_live_num, industry_screen,
         # battery of union questions
         union_mem_num, union_rep_num, union_scam_num, union_support_num, union_tech_num, union_voice_num, union_weak_num,  union_ever_num, union_fam_num, union_friends_num
         )  %>%
  mutate_all(haven::zap_labels) %>%
  # filter(vignette_neighbortreat != 1) %>%
  mutate(across(is.numeric, ~na_if(.x, -77))) %>%
  mutate(treatment = case_when(vignette_coworkertreat == 1 ~ "Coworker",
                               vignette_neighbortreat == 1 ~ "Neighbor",
                               vignette_control == 1 ~ "Control"
                               ),
         treatment = factor(treatment, levels=c("Control", "Coworker", "Neighbor"))) %>%
  # merge all separate vignette outcomes into action DV
  mutate(DV_strike = select(., ends_with("strike_num")) %>% rowSums(na.rm=T),
         DV_petition = select(., ends_with("petition_num")) %>% rowSums(na.rm=T),
         DV_union = select(., ends_with("union_num")) %>% rowSums(na.rm=T),
         DV_social_med = select(., ends_with("social_med_num")) %>% rowSums(na.rm=T),
         DV_donate = select(., ends_with("donate_num")) %>% rowSums(na.rm=T),
         DV_call_rep = select(., ends_with("call_rep_num")) %>% rowSums(na.rm=T)
         ) %>%
  mutate(across(starts_with("DV_"), ~na_if(.x, 0))) %>%
  mutate(across(starts_with("DV_"), ~ifelse(.x >= 3, 1, 0), .names = "{.col}_binary")) %>%
  # calculate "propensity to act"
  mutate(DV_action = select(., starts_with("DV_")) %>% rowSums(na.rm=T),
         DV_action = ifelse(DV_action == 0 , NA, DV_action),
         DV_action_binary = ifelse(if_any(ends_with("_binary"), ~ .x == 1), 1, 0)) %>%
  # recode industry screen
  mutate(industry = case_when(str_detect(industry_screen, "Hospit") ~ "Hospitality",
                              str_detect(industry_screen, "Wareh") ~ "Warehousing",
                              TRUE ~ industry_screen))

WERN_dict <- c(DV_strike = "Strike",
               DV_petition = "Sign Petition", 
               DV_union = "Reach out to Union",
               DV_social_med = "Post on Social Media",
               DV_donate = "Donate $20",
               DV_call_rep = "Call Elected Official",
               DV_action = "Likeliness of Action",
               `industry_screenHospitality(foodservice,hotel,etc.)` = "Hospitality",
               industry_screenRetail = "Retail",
               industry_screenTelecommunications = "Telecommunications",
               `industry_screenWarehousing(suchasatafulfillmentcenter)` = "Warehousing",
               industry_screenHealthcare = "Health Care",
               treatmentCoworkerTreatment = "Treatment: Coworker",
               treatmentNeighborTreatment = "Treatment: Neighbor",
               hh_618_num = "Childen 6 - 18 in HH")



# balanced table
sumtable(wern_treatment_df, group = 'treatment', group.test = TRUE)
# hh_618_num is not balanced. Everything else is pretty much balanced. Well ran survey. 

# Focusing on Main DV ------
# analysis 
feols(sw(DV_strike, DV_petition, DV_union,  DV_social_med, DV_donate, DV_call_rep, DV_action) ~ treatment
     , 
      data = wern_treatment_df)

# w/ industry FE
feols(sw(DV_strike, DV_petition, DV_union,  DV_social_med, DV_donate, DV_call_rep, DV_action) ~ treatment
      |industry_screen, 
      data = wern_treatment_df)




# varying slope by industry -------
DV_overall_model <- feols(sw(DV_strike, DV_union,  DV_petition,DV_social_med, DV_call_rep, DV_donate, DV_action) ~
                           treatment + hh_618_num | industry, 
                         data = wern_treatment_df)

DV_indint_model <- feols(sw(DV_strike, DV_union,  DV_petition,DV_social_med, DV_call_rep, DV_donate, DV_action) ~
                         i(treatment, industry, ref = "Control") + hh_618_num | industry, 
                       data = wern_treatment_df)

etable(DV_indint_model, dict = WERN_dict,
       headers=c("Workplace", "Workplace", "Community", "Community", "Community", "Ambiguous", "Index"))

## extract coefficients ------
DV_indint_df <- summary(DV_indint_model, type = "compact") %>%
  reshape2::melt(id.vars=c("lhs")) %>%
  separate(value, into = c("coef", "se"), 
           sep = " ", extra = "merge") %>% # Separate into two columns
  mutate(coef = gsub("\\*", "", coef)) %>% # remove * 
  mutate(coef = gsub("\\.$", "", coef)) %>% # remove period at the end that signifies 90% confidence
  mutate(se = gsub("[()]", "", se)) %>% # Remove parentheses
  separate(variable, into = c("filler", "treatment", "industry"), sep = "::") %>%
  mutate(treatment = str_replace(treatment, ":industry", "")) %>%
  select(-filler) %>%
  mutate(coef = as.numeric(coef),
         se = as.numeric(se)) %>% # Convert to numeric
  # calculate lower and upper bounds
  mutate(lower95 = coef - 1.96 * se,
         upper95 = coef + 1.96 * se) %>%
  mutate(lower90 = coef - 1.645 * se,
         upper90 = coef + 1.645 * se) %>%
  drop_na(coef) %>%
  # mutate(sample = relevel(as.factor(sample), ref="Full sample")) %>%
  mutate(significance = ifelse(lower95 < 0 & upper95 > 0, "Not Significant", "Significant")) %>%
  # drop hh_618_num
  drop_na(treatment)


DV_overall_df <- summary(DV_overall_model, type = "compact") %>%
  reshape2::melt(id.vars=c("lhs")) %>%
  separate(value, into = c("coef", "se"), 
           sep = " ", extra = "merge") %>% # Separate into two columns
  mutate(coef = gsub("\\*", "", coef)) %>% # remove * 
  mutate(coef = gsub("\\.$", "", coef)) %>% # remove period at the end that signifies 90% confidence
  mutate(se = gsub("[()]", "", se)) %>% # Remove parentheses
  # separate(variable, into = c("filler", "treatment", "industry"), sep = "::") %>%
  # mutate(treatment = str_replace(treatment, ":industry", "")) %>%
  mutate(treatment = case_when(variable == "treatmentCoworker" ~ "Coworker",
                               variable == "treatmentNeighbor" ~ "Neighbor",
                               TRUE ~ NA)) %>%
  mutate(coef = as.numeric(coef),
         se = as.numeric(se)) %>% # Convert to numeric
  # calculate lower and upper bounds
  mutate(lower95 = coef - 1.96 * se,
         upper95 = coef + 1.96 * se) %>%
  mutate(lower90 = coef - 1.645 * se,
         upper90 = coef + 1.645 * se) %>%
  drop_na(coef) %>%
  # mutate(sample = relevel(as.factor(sample), ref="Full sample")) %>%
  mutate(significance = ifelse(lower95 < 0 & upper95 > 0, "Not Significant", "Significant")) %>%
  # drop hh_618_num
  drop_na(treatment) %>%
  # make "all" industry
  mutate(industry = "All")


DV_indint_df %>%
  bind_rows(DV_overall_df) %>%
  mutate(lhs_dict = unname(WERN_dict[trimws(lhs)])) %>%
  ggplot(aes(y = lhs_dict, x = coef, color = significance)) +
  geom_point(position = position_dodge(width = 1),
             size = 3) +
  geom_errorbar(aes(xmin = lower95,
                    xmax = upper95),
                width = 0, alpha = 0.7,
                position = position_dodge(width = 1),
                linewidth = 1) +
  geom_errorbar(aes(xmin = lower90,
                    xmax = upper90),
                width = 0, alpha = 0.7,
                position = position_dodge(width = 1),
                linewidth = 1.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey")  +
  theme(legend.position="bottom",
        strip.text.y =element_text(angle = 0),
        # axis.text.y = element_blank()
  ) +
  facet_grid(rows=vars(industry),
             cols = vars(treatment),
             scales = "free") +
  labs(y = "",
       x = "Estimates (95% CI)",
       color = "",
       shape = "",
       # title = "Phaseout Coverage"
  ) +
  # theme_classic() +
  scale_color_manual(values=c( "grey", "red")) 


# logistic regression on binary DV (likely or not likely to take action) ------
binDV_indint_logit <- feglm(sw(DV_strike_binary, DV_union_binary,  DV_petition_binary, DV_social_med_binary, DV_call_rep_binary, DV_donate_binary, DV_action_binary) ~
                           i(treatment, industry, ref = "Control") + hh_618_num | industry, 
                         family = binomial(link = "logit"),
                         data = wern_treatment_df)

etable(binDV_indint_logit, dict = WERN_dict,
       headers=c("Workplace", "Workplace", "Community", "Community", "Community", "Ambiguous", "Index"))

binDV_indint_df <- summary(binDV_indint_logit, type = "compact") %>%
  reshape2::melt(id.vars=c("lhs")) %>% 
  separate(value, into = c("coef", "se"), 
           sep = " ", extra = "merge") %>% # Separate into two columns
  mutate(coef = gsub("\\*", "", coef)) %>% # remove * 
  mutate(coef = gsub("\\.$", "", coef)) %>% # remove period at the end that signifies 90% confidence
  mutate(se = gsub("[()]", "", se)) %>% # Remove parentheses
  separate(variable, into = c("filler", "treatment", "industry"), sep = "::") %>%
  mutate(treatment = str_replace(treatment, ":industry", "")) %>%
  select(-filler) %>%
  mutate(coef = as.numeric(coef),
         se = as.numeric(se)) %>% # Convert to numeric
  # calculate lower and upper bounds
  mutate(lower95 = coef - 1.96 * se,
         upper95 = coef + 1.96 * se) %>%
  mutate(lower90 = coef - 1.645 * se,
         upper90 = coef + 1.645 * se) %>%
  drop_na(coef) %>%
  # mutate(sample = relevel(as.factor(sample), ref="Full sample")) %>%
  mutate(significance = ifelse(lower95 < 0 & upper95 > 0, "Not Significant", "Significant")) %>%
  # drop hh_618_num
  drop_na(treatment)

binDV_indint_df %>%
  ggplot(aes(y = lhs, x = coef, color = significance)) +
  geom_point(position = position_dodge(width = 1),
             size = 3) +
  geom_errorbar(aes(xmin = lower95,
                    xmax = upper95),
                width = 0, alpha = 0.7,
                position = position_dodge(width = 1),
                linewidth = 1) +
  geom_errorbar(aes(xmin = lower90,
                    xmax = upper90),
                width = 0, alpha = 0.7,
                position = position_dodge(width = 1),
                linewidth = 1.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey")  +
  theme(legend.position="bottom",
        strip.text.y =element_text(angle = 0),
        # axis.text.y = element_blank()
  ) +
  facet_grid(rows=vars(industry),
             cols = vars(treatment),
             scales = "free") +
  labs(y = "",
       x = "Estimates (95% CI)",
       color = "",
       shape = "",
       # title = "Phaseout Coverage"
  ) +
  # theme_classic() +
  scale_color_manual(values=c( "grey", "red")) 





# union battery as DV ------
# should all be null, because they weren't in the treatment. Placebo....
union_indint_model <- feols(sw(union_mem_num, union_rep_num, union_scam_num, union_support_num, union_tech_num, union_voice_num, union_weak_num,  union_ever_num, union_fam_num, union_friends_num) ~
                        i(treatment, industry_screen, ref = "Control") | industry_screen, 
                      data = wern_treatment_df)


etable(union_indint_model, dict = WERN_dict)

iplot()

