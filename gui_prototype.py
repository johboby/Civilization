#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - GUI界面原型

这是一个使用tkinter构建的简单GUI界面原型，用于演示如何为文明演化模拟系统
创建图形用户界面。
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class CivilizationSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文明演化模拟系统")
        self.root.geometry("1000x700")
        
        # 模拟状态
        self.simulation_running = False
        self.simulation_thread = None
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="文明演化模拟系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 参数设置
        params_frame = ttk.Frame(control_frame)
        params_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(params_frame, text="文明数量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.civ_count_var = tk.StringVar(value="5")
        civ_count_entry = ttk.Entry(params_frame, textvariable=self.civ_count_var, width=10)
        civ_count_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_frame, text="模拟周期:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.cycle_count_var = tk.StringVar(value="100")
        cycle_count_entry = ttk.Entry(params_frame, textvariable=self.cycle_count_var, width=10)
        cycle_count_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_frame, text="网格大小:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.grid_size_var = tk.StringVar(value="20")
        grid_size_entry = ttk.Entry(params_frame, textvariable=self.grid_size_var, width=10)
        grid_size_entry.grid(row=0, column=5, sticky=tk.W)
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="开始模拟", command=self.start_simulation)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="停止模拟", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.load_button = ttk.Button(button_frame, text="加载配置", command=self.load_config)
        self.load_button.grid(row=0, column=2, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="保存配置", command=self.save_config)
        self.save_button.grid(row=0, column=3, padx=(0, 10))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=4, pady=(5, 0))
        
        # 图表区域
        chart_frame = ttk.LabelFrame(main_frame, text="模拟结果", padding="10")
        chart_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="模拟统计", padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        result_frame.columnconfigure(0, weight=1)
        
        self.result_text = tk.Text(result_frame, height=8, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
    def start_simulation(self):
        """开始模拟"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("模拟进行中...")
        
        # 在单独线程中运行模拟
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def stop_simulation(self):
        """停止模拟"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟已停止")
        
    def run_simulation(self):
        """运行模拟（模拟实现）"""
        try:
            # 获取参数
            civ_count = int(self.civ_count_var.get())
            cycle_count = int(self.cycle_count_var.get())
            grid_size = int(self.grid_size_var.get())
            
            # 模拟过程
            for i in range(cycle_count):
                if not self.simulation_running:
                    break
                    
                # 更新进度
                progress = (i + 1) / cycle_count * 100
                self.progress_var.set(progress)
                self.status_var.set(f"模拟进行中... ({i+1}/{cycle_count})")
                
                # 模拟一些计算时间
                self.root.after(50)  # 每个周期50ms
                
            # 模拟完成后更新界面
            self.root.after(0, self.simulation_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.simulation_error(str(e)))
            
    def simulation_completed(self):
        """模拟完成后的处理"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟完成")
        
        # 更新图表
        self.update_chart()
        
        # 更新结果
        self.update_results()
        
    def simulation_error(self, error_msg):
        """处理模拟错误"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟出错")
        messagebox.showerror("模拟错误", f"模拟过程中发生错误：{error_msg}")
        
    def update_chart(self):
        """更新图表"""
        self.ax.clear()
        
        # 生成示例数据
        x = np.arange(0, 100)
        y1 = np.random.rand(100)
        y2 = np.random.rand(100)
        y3 = np.random.rand(100)
        
        self.ax.plot(x, y1, label="文明1")
        self.ax.plot(x, y2, label="文明2")
        self.ax.plot(x, y3, label="文明3")
        self.ax.set_title("文明演化趋势")
        self.ax.set_xlabel("周期")
        self.ax.set_ylabel("指标值")
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()
        
    def update_results(self):
        """更新结果"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # 添加示例结果
        result_text = """
模拟完成统计：
================
总周期数: 100
文明数量: 5
网格大小: 20

最终状态:
- 文明1: 资源=1200, 科技=15, 领土=25
- 文明2: 资源=980, 科技=18, 领土=30
- 文明3: 资源=1500, 科技=12, 领土=20
- 文明4: 资源=800, 科技=20, 领土=15
- 文明5: 资源=1100, 科技=16, 领土=28

事件记录:
- 周期25: 发生技术突破，文明2获得额外科技点
- 周期50: 自然灾害影响文明1和文明4
- 周期75: 文明3与文明5建立外交关系
        """
        
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
        
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 更新界面参数
                self.civ_count_var.set(str(config.get('NUM_CIVILIZATIONS', 5)))
                self.cycle_count_var.set(str(config.get('SIMULATION_CYCLES', 100)))
                self.grid_size_var.set(str(config.get('GRID_SIZE', 20)))
                
                self.status_var.set(f"配置已加载: {filename}")
            except Exception as e:
                messagebox.showerror("加载错误", f"无法加载配置文件：{str(e)}")
                
    def save_config(self):
        """保存配置"""
        filename = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                config = {
                    'NUM_CIVILIZATIONS': int(self.civ_count_var.get()),
                    'SIMULATION_CYCLES': int(self.cycle_count_var.get()),
                    'GRID_SIZE': int(self.grid_size_var.get())
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                    
                self.status_var.set(f"配置已保存: {filename}")
            except Exception as e:
                messagebox.showerror("保存错误", f"无法保存配置文件：{str(e)}")

def main():
    root = tk.Tk()
    app = CivilizationSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - GUI界面原型

这是一个使用tkinter构建的简单GUI界面原型，用于演示如何为文明演化模拟系统
创建图形用户界面。
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class CivilizationSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文明演化模拟系统")
        self.root.geometry("1000x700")
        
        # 模拟状态
        self.simulation_running = False
        self.simulation_thread = None
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="文明演化模拟系统", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 参数设置
        params_frame = ttk.Frame(control_frame)
        params_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(params_frame, text="文明数量:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.civ_count_var = tk.StringVar(value="5")
        civ_count_entry = ttk.Entry(params_frame, textvariable=self.civ_count_var, width=10)
        civ_count_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_frame, text="模拟周期:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.cycle_count_var = tk.StringVar(value="100")
        cycle_count_entry = ttk.Entry(params_frame, textvariable=self.cycle_count_var, width=10)
        cycle_count_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_frame, text="网格大小:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.grid_size_var = tk.StringVar(value="20")
        grid_size_entry = ttk.Entry(params_frame, textvariable=self.grid_size_var, width=10)
        grid_size_entry.grid(row=0, column=5, sticky=tk.W)
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="开始模拟", command=self.start_simulation)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="停止模拟", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.load_button = ttk.Button(button_frame, text="加载配置", command=self.load_config)
        self.load_button.grid(row=0, column=2, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="保存配置", command=self.save_config)
        self.save_button.grid(row=0, column=3, padx=(0, 10))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=4, pady=(5, 0))
        
        # 图表区域
        chart_frame = ttk.LabelFrame(main_frame, text="模拟结果", padding="10")
        chart_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="模拟统计", padding="10")
        result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E))
        result_frame.columnconfigure(0, weight=1)
        
        self.result_text = tk.Text(result_frame, height=8, state=tk.DISABLED)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
    def start_simulation(self):
        """开始模拟"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("模拟进行中...")
        
        # 在单独线程中运行模拟
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def stop_simulation(self):
        """停止模拟"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟已停止")
        
    def run_simulation(self):
        """运行模拟（模拟实现）"""
        try:
            # 获取参数
            civ_count = int(self.civ_count_var.get())
            cycle_count = int(self.cycle_count_var.get())
            grid_size = int(self.grid_size_var.get())
            
            # 模拟过程
            for i in range(cycle_count):
                if not self.simulation_running:
                    break
                    
                # 更新进度
                progress = (i + 1) / cycle_count * 100
                self.progress_var.set(progress)
                self.status_var.set(f"模拟进行中... ({i+1}/{cycle_count})")
                
                # 模拟一些计算时间
                self.root.after(50)  # 每个周期50ms
                
            # 模拟完成后更新界面
            self.root.after(0, self.simulation_completed)
            
        except Exception as e:
            self.root.after(0, lambda: self.simulation_error(str(e)))
            
    def simulation_completed(self):
        """模拟完成后的处理"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟完成")
        
        # 更新图表
        self.update_chart()
        
        # 更新结果
        self.update_results()
        
    def simulation_error(self, error_msg):
        """处理模拟错误"""
        self.simulation_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("模拟出错")
        messagebox.showerror("模拟错误", f"模拟过程中发生错误：{error_msg}")
        
    def update_chart(self):
        """更新图表"""
        self.ax.clear()
        
        # 生成示例数据
        x = np.arange(0, 100)
        y1 = np.random.rand(100)
        y2 = np.random.rand(100)
        y3 = np.random.rand(100)
        
        self.ax.plot(x, y1, label="文明1")
        self.ax.plot(x, y2, label="文明2")
        self.ax.plot(x, y3, label="文明3")
        self.ax.set_title("文明演化趋势")
        self.ax.set_xlabel("周期")
        self.ax.set_ylabel("指标值")
        self.ax.legend()
        self.ax.grid(True)
        
        self.canvas.draw()
        
    def update_results(self):
        """更新结果"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # 添加示例结果
        result_text = """
模拟完成统计：
================
总周期数: 100
文明数量: 5
网格大小: 20

最终状态:
- 文明1: 资源=1200, 科技=15, 领土=25
- 文明2: 资源=980, 科技=18, 领土=30
- 文明3: 资源=1500, 科技=12, 领土=20
- 文明4: 资源=800, 科技=20, 领土=15
- 文明5: 资源=1100, 科技=16, 领土=28

事件记录:
- 周期25: 发生技术突破，文明2获得额外科技点
- 周期50: 自然灾害影响文明1和文明4
- 周期75: 文明3与文明5建立外交关系
        """
        
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)
        
    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 更新界面参数
                self.civ_count_var.set(str(config.get('NUM_CIVILIZATIONS', 5)))
                self.cycle_count_var.set(str(config.get('SIMULATION_CYCLES', 100)))
                self.grid_size_var.set(str(config.get('GRID_SIZE', 20)))
                
                self.status_var.set(f"配置已加载: {filename}")
            except Exception as e:
                messagebox.showerror("加载错误", f"无法加载配置文件：{str(e)}")
                
    def save_config(self):
        """保存配置"""
        filename = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                config = {
                    'NUM_CIVILIZATIONS': int(self.civ_count_var.get()),
                    'SIMULATION_CYCLES': int(self.cycle_count_var.get()),
                    'GRID_SIZE': int(self.grid_size_var.get())
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                    
                self.status_var.set(f"配置已保存: {filename}")
            except Exception as e:
                messagebox.showerror("保存错误", f"无法保存配置文件：{str(e)}")

def main():
    root = tk.Tk()
    app = CivilizationSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()