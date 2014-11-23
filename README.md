machine_learning
================
A=read.csv("./pml-training.csv")
Pred=read.csv("./pml-testing.csv")
library(caret)
# create data partition for testing and trainning
inA=createDataPartition( y = A$classe, p = 0.7, list = FALSE )
totrain=A[inA,]
T=A[-inA,]
ss=nearZeroVar( totrain)
totrain= totrain[,-ss]
Pred=Pred[-ss]
T=T[,-ss]

torain     <- totrain[, -grep( "^min_|^max_|^var_|^stddev_|^amplitude_|^avg_", names(totrain) ) ]
T      <- T[, -grep( "^min_|^max_|^var_|^stddev_|^amplitude_|^avg_", names(T) ) ]
Pred  <- Pred[, -grep( "^min_|^max_|^var_|^stddev_|^amplitude_|^avg_",names(Pred) ) ]

# Convert user_name to an indicator variable
dummies <- dummyVars(classe ~user_name , data = totrain)
totrain     <- cbind( predict(dummies, newdata=totrain), totrain )
totrain     <- totrain[, !(names(totrain) == "user_name")]
Tdummies     <- dummyVars( classe ~ user_name, data = T )
T      <- cbind( predict(Tdummies, newdata=T), T)
T  <- T[, !(names(T) == "user_name")]

pdummies  <- dummyVars( X ~ user_name, data = Pred )
Pred  <- cbind( predict(pdummies, newdata=Pred), Pred )
Pred   <- Pred[, !(names(Pred) == "user_name")]


totrain   <- cbind( totrain[, c(64, 7:11)], totrain[, c(1:6, 12:63)] )
T    <- cbind( T[, c(64, 7:11)], T[, c(1:6, 12:63)] )
Pred  <- cbind( Pred[, c(7:11)], Pred[, c(1:6, 12:63)] )


anObj <- preProcess( totrain[, c(7:64)], method = c("center", "scale") )
totrain     <- cbind(  totrain[, c(1:6)], predict( anObj, totrain[, c(7:64)] )  )
T      <- cbind(  T[, c(1:6)], predict( anObj, T[, c(7:64)] )  )
Pred   <- cbind(  Pred[, c(1:5)], predict( anObj, Pred[, c(6:63)] )  )

Obj <- preProcess( totrain[, c(7:64)], method = c("knnImpute") )
totrain     <- cbind(  totrain[, c(1:6)], predict( Obj, totrain[, c(7:64)] )  )
T     <- cbind(  T[, c(1:6)], predict( Obj, T[, c(7:64)] )  )
Pred   <- cbind(  Pred[, c(1:5)], predict( Obj, Pred[, c(6:63)] )  )

totrain     <- cbind( list( "classe" = totrain[, 1] ), totrain[, c(6:64)] )
T     <- cbind( list( "classe" = T[, 1] ), T[, c(6:64)] )
Pred   <- Pred[, c(5:63)]





#  trained using Random forest which has an internal cross validation
trained= train(classe ~., method="rf", data=totrain, prox = T , trControl = trainControl(method = "cv", number = 5)

save( file = "trainingModel.Rdata", "trained")
pt_test= predict(trained, T)
validation=sum(T$classe==pt_test)*100/length(pt_test)
# with out of sample accuracy > 99



