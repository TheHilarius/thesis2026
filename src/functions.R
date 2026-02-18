



current_user <- Sys.info()[["user"]]

if (current_user == "your_username") {
  setwd("C:/Users/your_username/Projects/MyProject")
} else if (current_user == "friend_username") {
  setwd("C:/Users/friend_username/Documents/MyProject")
} else {
  stop("Unknown user. Please set working directory manually.")
}