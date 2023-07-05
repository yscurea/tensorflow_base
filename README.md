# Windows, Anaconda, Tensorflow 環境における実験管理

## 使用するもの

- hydra
- mlflow
- optuna

## 使い方

- generator をはじめとした .py ファイルをタスクに合わせて編集する
- hydra で使用する config 配下を編集する。

$ ./batch/train_multirun.bat
$ ./batch/mlflow_ui.bat
