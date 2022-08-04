from pytorch_lightning.callbacks import DeviceStatsMonitor
import torch

import pytorch_lightning as pl

from transformers import DistilBertTokenizer

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from QuestionAnsweringModule import QuestionAnsweringModule
from QuestionAnsweringModel import QuestionAnsweringModel
from local.Prediction import Prediction
import pandas as pd

from azureml.core.run import Run
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import shutil

import argparse
import os


def main():

    run = Run.get_context()
    run_child = run.child_run()
    print("Started run: ", run_child.id)
    id = run_child.id

    # get input dataset by name
    dataset_train = run.input_datasets['input_train']
    dataset_test = run.input_datasets['input_test']
    dataset_val = run.input_datasets['input_val']

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_train", type=str)
    arg_parser.add_argument("input_test", type=str)
    arg_parser.add_argument("input_val", type=str)
    arg_parser.add_argument("--logdir", type=str)
    arg_parser.add_argument("--output_dir", type=str)

    arg_parser.add_argument("--deepspeed_config", type=str)
    arg_parser.add_argument("--local_rank", type=str)
    arg_parser.add_argument("--with_aml_log", type=str)

    arguments = arg_parser.parse_args()
    output_path = arguments.output_dir
    print("Output_dir", output_path)

    train_df = dataset_train.to_pandas_dataframe()
    test_df = dataset_test.to_pandas_dataframe()
    val_df = dataset_val.to_pandas_dataframe()
    
    print(train_df.head())
    print(test_df.head())
    print(val_df.head())

    pl.seed_everything(1234)

    testing_mode = True
    use_own_dataset = False
    overfit_batches = False
    use_gooqa_pretraining = False
    use_squad_pretraining = False

    print("Testing mode activated?", testing_mode)
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    model = QuestionAnsweringModel(model_name)
    print("Done loading model :", model_name)
    BATCH_SIZE = 8
    logs_dir = "./logs"
    print("Using batch size :", BATCH_SIZE)

    output_dir = "./outputs"
    MAX_EPOCHS = 1
    print("Using epochs :", MAX_EPOCHS)
    
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./logs",
        filename="best_model.pt",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    tb_logger = TensorBoardLogger(logs_dir)

    
    with mlflow.start_run() as run:
        mlflow_uri = mlflow.get_tracking_uri()
        exp_id = run.info.experiment_id
        exp_name = mlflow.get_experiment(exp_id).name

        mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=mlflow_uri)
        mlf_logger._run_id = run.info.run_id

    device_stats = DeviceStatsMonitor()
   # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=False,mode="max")
   #gradient_clip_val=0.5,
   #auto_scale_batch_size="power",
   
   #Removed precision to possible avoid nan's in loss
   #precision=16,
   #strategy="dp"
   #devices = list(range(torch.cuda.device_count())),
   #accelerator="gpu", 
    print()

    
    trainer = pl.Trainer(default_root_dir="./logs", accumulate_grad_batches=4, checkpoint_callback=checkpoint_callback, max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=30, logger=[tb_logger,mlf_logger], callbacks=[device_stats], strategy="deepspeed", plugins=[AzureClusterEnvironment()])


    if use_own_dataset:
        prepare_df = PrepareDataFrame()
        dataframe = prepare_df.read_pickle("", "short_answers_without_youtube.pkl")
        dataframe = prepare_df.prepare_dataset(dataframe)
        dataframe = prepare_df.remove_empty_rows(dataframe)
    
        dataframe.to_csv(os.path.join(output_dir, "dataframe.csv"))

        x_train_df, x_test_df = train_test_split(dataframe, test_size=300,shuffle=True)
        x_train_df, x_val_df = train_test_split(x_train_df, test_size=50,shuffle=True)

        train_df = x_train_df.reset_index(drop=True)
        test_df = x_test_df.reset_index(drop=True)
        val_df = x_val_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(output_dir, "train_df.csv"))
        test_df.to_csv(os.path.join(output_dir, "test_df.csv"))
        val_df.to_csv(os.path.join(output_dir, "val_df.csv"))
            
        print("Train df", train_df.shape)
        print("Test df", test_df.shape)
        print("Val df", val_df.shape)
    

    if use_gooqa_pretraining:

        prepare_df = PrepareDataFrame()
        train_df = prepare_df.prepare_dataset(train_df)
        test_df = prepare_df.prepare_dataset(test_df)
        val_df = prepare_df.prepare_dataset(val_df)

        print("Train df", train_df.shape)
        print("Test df", test_df.shape)
        print("Val df", val_df.shape)

        train_df.to_csv(os.path.join(output_dir, "train_df.csv"))
        test_df.to_csv(os.path.join(output_dir, "test_df.csv"))
        val_df.to_csv(os.path.join(output_dir, "val_df.csv"))
    
    if use_squad_pretraining:
        print("USING SQUAD FOR PRETRAINING")
        prepare_df = PrepareDataFrame()
        train_df = pd.read_pickle(os.path.join("squad_datasets", "df_train.pkl"))
        test_df = pd.read_pickle(os.path.join("squad_datasets", "df_test.pkl"))
        val_df = pd.read_pickle(os.path.join("squad_datasets", "df_dev.pkl"))

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print("Train df", train_df.shape)
        print("Test df", test_df.shape)
        print("Val df", val_df.shape)

        train_df.to_csv(os.path.join(output_dir, "train_df.csv"))
        test_df.to_csv(os.path.join(output_dir, "test_df.csv"))
        val_df.to_csv(os.path.join(output_dir, "val_df.csv"))
        

    if overfit_batches:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        trainer = pl.Trainer(overfit_batches = 10)


    if testing_mode:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        train_df = train_df.reset_index(drop=True)[:1]
        test_df = test_df.reset_index(drop=True)[:1]
        val_df = val_df.reset_index(drop=True)[:1]
        BATCH_SIZE = 1

        print("Train df", train_df.shape)
        print("Test df", test_df.shape)
        print("Val df", val_df.shape)

        train_df.to_csv(os.path.join(output_dir, "train_df.csv"))
        test_df.to_csv(os.path.join(output_dir, "test_df.csv"))
        val_df.to_csv(os.path.join(output_dir, "val_df.csv"))
   

    data_module = QuestionAnsweringModule(train_df=train_df,
                                          test_df=test_df,
                                          val_df=val_df,
                                          tokenizer=tokenizer,
                                          batch_size=BATCH_SIZE)


    trainer.fit(model, data_module)
    trainer.validate(ckpt_path='best', dataloaders=data_module.test_dataloader())
    trainer.test(ckpt_path='best',dataloaders=data_module.val_dataloader())

    prediction = Prediction()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print(f"[Generating predictions]...\n")
    for epoch in range(1):
        predictions, actuals, questions, answers = prediction.validate(epoch, tokenizer, model, device, data_module.val_dataloader())
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals, "Questions": questions, "Target answers": answers})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))


        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals, "Questions": questions, "Answers": answers})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

        score_f1 = []
        score_em = []
        score_rouge = []

        print("Current pred before prediction", predictions)
        print("Current actuals before prediction", actuals)
                
        for pred, actual in zip(predictions, actuals):
            f1 = prediction.f1_score(pred, actual)
            em = prediction.exact_match_score(pred,actual)
            rouge = prediction.score_rouge(pred, actual)
            score_f1.append(f1)
            score_em.append(em)
            score_rouge.append(rouge)
        
        scores_df = pd.DataFrame({"Rouge": score_rouge, "F1": score_f1, "EM":score_em ,"Generated Text": predictions, "Actual Text": actuals, "Questions": questions, "Answers": answers})
        scores_df.to_csv(os.path.join(output_dir, "scores.csv"))

    print(f"[Done with predictions]...\n")

    #shutil copy file to ..
    chcekpoint_output_path = os.path.join(output_path, str(id))
    shutil.copytree("./logs", chcekpoint_output_path)
    print("OUTPUT-Path for checkpoints", chcekpoint_output_path)
   

if __name__ == "__main__":
    main()
