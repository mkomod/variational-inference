library(tidyverse)
theme_set(theme_bw(base_size = 12))

chart <- "MF_" # use "" for MVN VI

# df_samples_means <- read_csv(paste0(chart, "df_samples_means.csv"))
df_samples <- read_csv(paste0(chart, "df_samples.csv")) %>% 
  select(-X1) %>% 
  gather(var, val, -parameter) %>% 
  mutate(parameter = ifelse(parameter=="log_sigma", "Log-sigma", parameter))
df_hmc <- read_csv(paste0(chart,"df_hmc.csv")) %>% 
  select(-X1) %>% 
  gather(parameter, val) %>% 
  mutate(parameter = ifelse(parameter=="log_sigma", "Log-sigma", parameter))


df_samples  %>% 
  # can't remember why this didn't work for the facets; not super important any way.
  mutate(
    parameter = forcats::fct_relevel(parameter, "Intercept", after=0),
    parameter = forcats::fct_relevel(parameter, "Log-sigma", after=Inf)
  ) %>% 
  ggplot(aes(val, group=var)) +
  geom_density(size=.2, col="#6e6e6e") +
  facet_wrap(~parameter, scales = "free", nrow=3) +
  scale_colour_manual(values="darkred") +
  labs(x="", col="") +
  geom_density(
    inherit.aes = FALSE, aes(val, col="HMC posterior"),
    data=df_hmc, fill = "darkred", alpha=.2
  ) +
  theme(
    legend.position = "bottom", 
    legend.margin=margin(t = -.5, unit='cm')
  )


