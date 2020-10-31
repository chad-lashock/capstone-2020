library(ggplot2)
library(zoo)


# Read CAT data
data = read.csv("./data/CAT.csv", header = T)
cat = data.frame(date=as.Date(data$Date), price=data$Close)

#Calculate 200-day and 50-day moving averages
cat$ma200[200:5228] = rollmean(x=cat$price,k=200)
cat$ma50[50:5228] = rollmean(x=cat$price,k=50)


#Plot CAT
ggplot(data=cat, aes(x=date)) + 
  geom_line(color="black", aes(y = price)) +
  ylab("Price ($)") +
  xlab("Date") +
  ggtitle("CAT Stock Price") +
  theme(plot.title = element_text(hjust=0.5))

#Plot CAT 200-day and 50-day moving averages
ggplot(data=cat, aes(x=date)) +
  geom_line(color="dark green", aes(y = ma200)) +
  geom_line(color = "red", aes(y = ma50)) +
  ylab("Price ($)") +
  xlab("Date") +
  ggtitle("CAT Moving Averages") +
  theme(plot.title = element_text(hjust=0.5))



