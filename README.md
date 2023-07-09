# ディープラーニング環境における実験管理

## 使用するもの

- hydra
- mlflow
- optuna

## 使い方

.envファイル作成、.env.sampleを参考に記述
以下実行
```
batch/train_multirun.bat
batch/mlflow_ui.bat
```

## TODO

- [ ] [ivy](https://github.com/unifyai/ivy)を試し、可能なら対応する.
  - [ ] pytorch に対応する.ivy で済むなら必要ない。
- [ ] Anaconda 環境以外の環境でも動作させられるようにする.
- [ ] 画像処理以外のタスクでも使用できるようにする.
- [ ] logファイル等の出力先のフォルダを任意に変更できるようにする.
- [ ] README.md を充実させる.
- [ ] DOCKERFILEなどでDockerで構築できるようにする.