@echo off

cd %~dp0/..

FOR /F "tokens=* delims=" %%G IN (.env) DO SET %%G

@rem activate anaconda
call %USERPROFILE%\anaconda3\Scripts\activate.bat
call activate %ANACONDA_ENV_NAME%


@rem pythonファイル実行
%USERPROFILE%\anaconda3\envs\%ANACONDA_ENV_NAME%\python.exe train.py
