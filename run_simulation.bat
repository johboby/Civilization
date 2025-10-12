@echo off
REM 文明演化模拟系统启动脚本
REM 提供多种启动选项，方便用户快速开始模拟

SETLOCAL ENABLEDELAYEDEXPANSION

REM 配置虚拟环境名称
SET VENV_NAME=venv

REM 检查Python是否安装
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
echo 错误: 未找到Python。请先安装Python 3.8或更高版本。
echo 访问 https://www.python.org/downloads/ 下载安装。
pause
exit /b 1
)

REM 检查虚拟环境是否存在
IF NOT EXIST %VENV_NAME% (
echo 正在创建虚拟环境...
python -m venv %VENV_NAME%
IF %ERRORLEVEL% NEQ 0 (
echo 创建虚拟环境失败！
pause
exit /b 1
)

REM 激活虚拟环境并安装依赖
echo 正在安装依赖包...
CALL %VENV_NAME%\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
echo 安装依赖失败！
pause
exit /b 1
)

) ELSE (
REM 激活虚拟环境
echo 正在激活虚拟环境...
CALL %VENV_NAME%\Scripts\activate.bat
)

REM 显示菜单
:MENU
cls
echo =======================================================
echo              文明演化模拟系统 - 启动菜单
 echo =======================================================
echo 请选择要执行的操作:
echo [1] 快速演示 (4个文明，100个周期，小型网格)
echo [2] 标准模拟 (6个文明，300个周期，中型网格)
echo [3] 大规模模拟 (10个文明，500个周期，大型网格)
echo [4] 资源稀缺模式 (更具挑战性的资源环境)
echo [5] 科技优先模式 (加速科技发展)
echo [6] 交互式命令行 (自定义参数)
echo [7] 运行测试系统 (验证功能是否正常)
echo [8] 生成示例配置 (创建自定义配置文件)
echo [9] 退出
 echo =======================================================

SET /P CHOICE=请输入选择 [1-9]: 

REM 根据用户选择执行相应操作
IF "%CHOICE%" == "1" (
echo 正在启动快速演示模式...
python simulation_cli.py --config example_config.py --preset demo
GOTO END
)

IF "%CHOICE%" == "2" (
echo 正在启动标准模拟模式...
python simulation_cli.py --config example_config.py --preset standard
GOTO END
)

IF "%CHOICE%" == "3" (
echo 正在启动大规模模拟模式...
echo 注意: 大规模模拟可能需要较长时间和较高的系统资源。
python simulation_cli.py --config example_config.py --preset large_scale
GOTO END
)

IF "%CHOICE%" == "4" (
echo 正在启动资源稀缺模式...
echo 注意: 此模式下资源更加稀缺，文明竞争更激烈。
python simulation_cli.py --config example_config.py --preset resource_scarcity
GOTO END
)

IF "%CHOICE%" == "5" (
echo 正在启动科技优先模式...
echo 注意: 此模式下科技发展速度更快。
python simulation_cli.py --config example_config.py --preset tech_focus
GOTO END
)

IF "%CHOICE%" == "6" (
echo 启动交互式命令行界面...
echo 提示: 可以使用 'python simulation_cli.py --help' 查看所有可用参数。
python simulation_cli.py --interactive
GOTO END
)

IF "%CHOICE%" == "7" (
echo 正在运行测试系统...
echo 此操作将验证系统核心功能是否正常工作。
python test_system.py
GOTO END
)

IF "%CHOICE%" == "8" (
echo 正在生成示例配置文件...
IF NOT EXIST my_config.py (
echo 请为您的配置文件命名，默认为 'my_config.py':
SET /P CONFIG_NAME=
IF "%CONFIG_NAME%" == "" (
SET CONFIG_NAME=my_config.py
)
IF EXIST %CONFIG_NAME% (
echo 警告: 文件 '%CONFIG_NAME%' 已存在，是否覆盖？[y/n]
SET /P OVERWRITE=
IF /I NOT "%OVERWRITE%" == "y" (
echo 取消操作。
GOTO MENU
)
)

REM 创建自定义配置文件
echo 正在创建配置文件 '%CONFIG_NAME%'...
(
echo # 我的自定义配置文件
echo from example_config import simulation_config

echo # 请根据需要修改以下参数
echo # 文明数量
echo simulation_config.NUM_CIVILIZATIONS = 6
echo # 模拟周期数
echo simulation_config.SIMULATION_CYCLES = 300
echo # 网格大小
echo simulation_config.GRID_SIZE = 250
echo # 初始资源量
echo simulation_config.INITIAL_RESOURCE_AMOUNT = 200.0
echo # 随机种子 (设置为固定值以复现结果)
echo # simulation_config.RANDOM_SEED = 42

echo # 输出参数
echo simulation_config.RESULTS_DIR = "my_simulation_results"
echo simulation_config.SAVE_VISUALIZATION = True
echo simulation_config.GENERATE_REPORT = True


echo # 如需使用预设配置，取消下面一行的注释
echo # from example_config import apply_preset
echo # simulation_config = apply_preset('standard')


echo # 配置完成
echo print("自定义配置已加载！")
echo config = simulation_config
) > %CONFIG_NAME%

echo 配置文件 '%CONFIG_NAME%' 创建成功！
echo 您可以编辑此文件自定义更多参数。
echo 使用命令: python simulation_cli.py --config %CONFIG_NAME% 来运行。
echo.
echo 按任意键返回菜单...
pause >nul
GOTO MENU
) ELSE (
echo 示例配置文件已存在。请先删除或重命名 'my_config.py' 文件。
echo 按任意键返回菜单...
pause >nul
GOTO MENU
)
)

IF "%CHOICE%" == "9" (
echo 感谢使用文明演化模拟系统，再见！
pause
exit /b 0
)

REM 处理无效输入
echo 无效的选择，请重新输入。
echo 按任意键继续...
pause >nul
GOTO MENU

:END
echo.
echo 模拟已完成！
echo 结果保存在 %RESULTS_DIR% 目录中。
echo 按任意键返回菜单...
pause >nul
GOTO MENU

ENDLOCAL