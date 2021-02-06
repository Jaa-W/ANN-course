df <- read.csv("D:/PROJEKTY/MachineLearning/ANN-tutorial/Data Files/House_Price.csv", header = TRUE)

summary(df)
hist(df$crime_rate)
pairs(~price+crime_rate+n_hot_rooms+rainfall,data = df)
barplot(table(df$waterbody))

# Outlier treatment

uv = 3*quantile(df$n_hot_rooms, 0.99)
df$n_hot_rooms[df$n_hot_rooms > uv ] <- uv
lv = 0.3 * quantile(df$rainfall, 0.01)
df$rainfall[df$rainfall < lv] <- lv

# Missing value

df$n_hos_beds[is.na(df$n_hos_beds)] <- mean(df$n_hos_beds, na.rm = TRUE)


# Variable transformation

plot(df$price,df$crime_rate)

df$crime_rate = log(1+df$crime_rate)

df$avg_dist = (df$dist1 + df$dist2 + df$dist3 + df$dist4)/4

df2 <- df[,-7:-10]
df <- df2
rm(df2)

df <- df[,-14]


# DUMMY variables

#install.packages("dummies")
library(dummies)

df <- dummy.data.frame(df)
df <- df[,-9]
df <- df[,-14]
