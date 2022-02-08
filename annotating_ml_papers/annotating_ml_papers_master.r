# ------------------------------------------------------------------------------
# Power is Nothing Without Control 
# Annotating Machine Learning Papers
# Chris Arnold, December 2021
# ------------------------------------------------------------------------------

# -- Housekeeping --------------------------------------------------------------
library(viridis)
library(scales)
library(stargazer)

# -- Data ---------------------------------------------------------------------- 
dat.temp <- read.csv("UML_Journals_all.csv", sep = ';', header = TRUE)
dat <- subset(dat.temp, subset = dat.temp$machine_learning == TRUE)

# Recoding some data 
dat$tuning_how <- car::recode(dat$tuning_how, "4 = 3")
dat$journal <- car::recode(dat$journal, "
            'American Political Science Review' = 'APSR';
            'Political Analysis' = 'PA';
            'Political Science Research and Methods' = 'PSRM'
            ")

# Defining Colour
colrs <- rev(viridis_pal(option = "D")(4))

# Function to write the label text 
label.text.writer <- function(table.dat.row, table.tuning.how.numbers.row,
                              y.pos, cex = 0.7){
  text(table.dat.row[1]/2, y.pos, 
       if(table.tuning.how.numbers.row[1] == 1){
         paste('No ML\nCode:\n',table.tuning.how.numbers.row[1],' Paper', sep = '')
       } else {
         paste('No ML\nCode:\n',table.tuning.how.numbers.row[1],' Papers', sep = '')
       },
       col = 'black', cex = cex)
  text((table.dat.row[1]+ table.dat.row[2]/2), y.pos, 
       if(table.tuning.how.numbers.row[2] == 1){
         paste('No Tuning\n& Default HP:\n', table.tuning.how.numbers.row[2],' Paper',sep = '')      
       }else {
         paste('No Tuning\n& Default HP:\n', table.tuning.how.numbers.row[2],' Papers',sep = '')
       },
       col = 'white', cex = cex)
  text((sum(table.dat.row[1:2])+ table.dat.row[3]/2), y.pos, 
       if(table.tuning.how.numbers.row[3] == 1){
         paste('No Tuning\n& Self Set HP:\n', table.tuning.how.numbers.row[3], ' Paper', sep = '')
       }else {
         paste('No Tuning\n& Self Set HP:\n', table.tuning.how.numbers.row[3], ' Papers', sep = '')
       },
       col = 'white', cex = cex)
  text((sum(table.dat.row[1:3])+ table.dat.row[4]/2), y.pos, paste(
    'HP Tuning:\n',table.tuning.how.numbers.row[4], ' Papers', sep = ''),
    col = 'white', cex = cex)
}

# -- Per Journal ---------------------------------------------------------------
table.tuning.how <- prop.table(table(dat$tuning_how, dat$journal), 2)
table.tuning.how.numbers <- table(dat$tuning_how, dat$journal)

pdf('hp_how.pdf', height = 6, width = 16)
barplot(table.tuning.how, xaxt = 'n',
        border = colrs, col = colrs, horiz = TRUE, las = 1)
axis(1, at = c(0, 0.2, 0.4, 0.6, 0.8, 1), las = 1,
     labels = c('0%', '20%', '40%', '60%', '80%', '100%'))
label.text.writer(table.tuning.how[,1], table.tuning.how.numbers[,1], .7)
label.text.writer(table.tuning.how[,2], table.tuning.how.numbers[,2], 1.9)
label.text.writer(table.tuning.how[,3], table.tuning.how.numbers[,3], 3.1)
dev.off()

# -- All journals --------------------------------------------------------------
table.tuning.how.all <- prop.table(table(dat$tuning_how, dat$machine_learning))
table.tuning.how.all.numbers <- table(dat$tuning_how, dat$machine_learning)

pdf('hpt_how_all.pdf', height = 3, width = 12)
barplot(table.tuning.how.all,  xaxt = 'n', yaxt = 'n',  
        border = colrs, col = colrs, horiz = TRUE)
axis(1, at = c(0, 0.2, 0.4, 0.6, 0.8, 1), las = 1,
     labels = c('0%', '20%', '40%', '60%', '80%', '100%'))
label.text.writer(table.tuning.how.all[,1], 
                  table.tuning.how.all.numbers[,1], .7, cex = 1)
dev.off()


# -- Numbers for the Paper -----------------------------------------------------
table.tuning.how.all
table.tuning.how.all.numbers

# How many papers?
sum(table.tuning.how.all.numbers)

# Who mentions HPs? 
sum(table.tuning.how.all[3:4])
sum(table.tuning.how.all[1:2])
sum(table.tuning.how.all.numbers[3:4])



# -- Appendix Table ------------------------------------------------------------
dat.annotated <- data.frame(dat['authors'], dat['journal'], dat['tuning_how'])

names(dat.annotated) <- c("Authors", "Journal", "Annotation")


write.csv(dat.annotated, 'annotations.csv')
stargazer(capture.output(dat.annotated), out = 'annotations.tex')

names(dat)
