packages <- c(
  "tidyverse", "readr", "janitor", "data.table",
  "dplyr", "tidyr", "stringr", "lubridate",
  "ggplot2", "ggpubr", "patchwork", "ggridges", "ggthemes", "scales",
  "psych", "rstatix", "car", "broom", "emmeans", "effectsize",
  "corrplot", "ggcorrplot", "readxl", "openxlsx", "lme4", "nlme"
)

installed_packages <- rownames(installed.packages())

for (pkg in packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg)
  }
}

library(tidyverse)
library(readr)
library(janitor)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)

library(ggplot2)
library(ggpubr)
library(patchwork)
library(ggridges)
library(ggthemes)
library(scales)

library(psych)
library(rstatix)
library(car)
library(broom)
library(emmeans)
library(effectsize)
library(corrplot)
library(ggcorrplot)
library(broom.mixed) 

library(performance)
library(bestNormalize)
library(car)

library(boot)
library(purrr)

library(readxl)
library(openxlsx)

library(lme4)
library(nlme)
library(glmm)
library(lmerTest)
library(glmmTMB)
library(DHARMa)
library(emmeans)
library(pbkrtest)
library(sjPlot)

# ---- Optional: Set default ggplot theme ----
theme_set(theme_minimal())