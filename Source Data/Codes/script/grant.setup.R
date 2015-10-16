
#install.packages("openxlsx")
#install.packages("sqldf")

#setwd()

library("openxlsx")
library("sqldf")
grant_app <- read.xlsx("Data/Projects Applications.xlsx", sheet = 1, startRow = 1, colNames = TRUE)
grant_award <- read.xlsx("Data/Projects Awarded.xlsx", sheet = 1, startRow = 1, colNames = TRUE)
grant_eval <- read.xlsx("Data/Projects Evaluations (2010 - Q2).xls", sheet = 1, startRow = 1, colNames = TRUE)
grant_vol_link <- read.xlsx("Data/Projects-Volunteers Links.xlss", sheet = 1, startRow = 1, colNames = TRUE)

  
  
grant_comb<-sqldf("select a.*, b.* 
                 from grant_app as a left outer join grant_award as b
                  on a.sgapp_id = b.sgapp_id
                  and a.org_id = b.org_id")


## how to join app,award to eval??