import fasttext

model = fasttext.train_supervised(
    input="var/quality_train.txt",
    lr=1,
    epoch=1000,
    wordNgrams=2,
    dim=50
)
model.save_model("var/quality_classifier.bin")