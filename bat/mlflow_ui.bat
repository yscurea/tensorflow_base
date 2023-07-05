@echo off

chcp 65001

cd %~dp0/../

FOR /F "tokens=* delims=" %%G IN (.env) DO SET %%G

@rem activate anaconda
call %USERPROFILE%\anaconda3\Scripts\activate.bat
call activate %ANACONDA_ENV_NAME%

cd %~dp0/../logs/

call mlflow ui