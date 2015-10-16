
#install.packages("openxlsx")
#install.packages("sqldf")

library("openxlsx")
library("sqldf")
vol_app <- read.xlsx("Data/Volunteers Applications.xlsx", sheet = 1, startRow = 1, colNames = TRUE)
vol_eval <- read.xlsx("Data/Volunteers Evaluation (2010 - Q2).xlsx", sheet = 1, startRow = 1, colNames = TRUE)

vol_comb<-sqldf("select a.*, b.* 
                 from vol_eval as a right outer join vol_app as b
                 on a.'Email.Address' = b.email")
