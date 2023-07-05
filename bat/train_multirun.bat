@echo off

chcp 65001

cd %~dp0/../

FOR /F "tokens=* delims=" %%G IN (.env) DO SET %%G

@rem activate anaconda
call %USERPROFILE%\anaconda3\Scripts\activate.bat
call activate %ANACONDA_ENV_NAME%

set comment="バッチサイズを調整"

echo %comment%

@rem execute train.py with multirun
%USERPROFILE%\anaconda3\envs\%ANACONDA_ENV_NAME%\python.exe train.py -m^
    "comment='%comment%'"^
    "optimizer_config.accumulation_steps=range(2, 6)"^
    "optimizer_config.learning_rate=choice(1.0e-3,1.0e-4)"^
