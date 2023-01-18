## Pretrained RoBERTa model

install.packages("transformers")
library(transformers)


## Pre-trained RoBERTa
roberta_model <- RobertaModel.from_pretrained("roberta-base")
tokenizer <- RobertaTokenizer.from_pretrained("roberta-base")

# tokenize your input text and convert them to the format that the model expects.
train_data$encoded_text <- tokenizer$encode_batch(train_data$text)
test_data$encoded_text <- tokenizer$encode_batch(test_data$text)

# add a final dense layer for the classification task, and specify the number of classes(3 in this case) in the final dense layer

output_layer <- layer_dense(units = 3, activation = "softmax")

model <- keras_model(roberta_model, output_layer)

# compile the model by specifying the optimizer, loss function, and evaluation metric

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# fine-tune the model on your labeled data by using the fit() function.

history <- model %>% fit(
  x = train_data$encoded_text, 
  y = to_categorical(train_data$sentiment), 
  epochs = 10,
  batch_size = 32,
  validation_data = list(test_data$encoded_text, to_categorical(test_data$sentiment))
)


#  evaluate the model performance on the test data by using the evaluate() function

model %>% evaluate(
  x = test_data$encoded_text, 
  y = to_categorical(test_data$sentiment)
)
