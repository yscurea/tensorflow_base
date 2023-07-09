@echo off

chcp 65001

cd %~dp0/../

FOR /F "tokens=* delims=" %%G IN (.env) DO SET %%G

@REM activate anaconda
call %USERPROFILE%\anaconda3\Scripts\activate.bat
call activate %ANACONDA_ENV_NAME%

set comment="sample comment"

echo %comment%

@REM execute train.py with multirun
%USERPROFILE%\anaconda3\envs\%ANACONDA_ENV_NAME%\python.exe train.py "comment='%comment%'"


@REM Add override parameters "batch_size=2"
