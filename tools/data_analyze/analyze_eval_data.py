import pickle
import pandas as pd

import json
import numpy as np
import re
import torch

class DataAnalyze:
    def __init__(self, pkl_path: str = None, track_json_path: str = None, log_path: str = None, final_json_path: str = None):
        """

        Args:
            pkl_path (str): _pkl文件路径_
            json_path (str): _track的json /track/metrics_summary.json_
            log_path (str): _eval log文件路径_
            final_json_path (str): _储存最终文件的路径_
        """
        self.pkl_path = pkl_path
        self.track_path = track_json_path
        self.log_path = log_path
        self.final_json_path = final_json_path
        self.bbox_data = None
        self.track_data = None
        self.map_data = None
        self.motion_data = None
        self.occ_data = None
        self.planning_data = None
        self.pd_data = pd.DataFrame()
        self.occ_df = None
        self.planning_df = None
        self.track_df = None
        self.track_result_df = None
        self.map_df = None
        self.motion_df = None

    def load_data(self):
        """加载 .pickle 文件数据"""
        if self.final_json_path is not None:
            print("Loading data from final json file: ")
            self._load_final_json()
        else:
            print("Loading data from origin file: ")
            self._load_pkl_data()
            self._load_track_json()
            self._load_log_data()

        # 分析数据
        self._analyze_map()
        self._analyze_motion()
        self._analyze_occ()
        self._analyze_planning()
        self._analyze_track()

    
    def _load_log_data(self):
        """加载 eval log 文件数据"""
        print("Loading log data from: ", self.log_path)
        try:
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1]
                print("Log data loaded successfully!")
        except FileNotFoundError:
            print(f"文件 {self.log_path} 未找到。")
            return None
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return None
        
        # 处理掉nan的情况
        if 'nan' in last_line:
            last_line = last_line.replace('nan', '0.0')
        

        # 转换为字典
        last_line_dict = eval(last_line)

        #* map部分
        map_keys = ['lanes_iou', 'drivable_iou', 'divider_iou', 'crossing_iou']

        # 从last_line_dict读取数据，转换为pandas表格
        map_data = {}
        for key in map_keys:
            map_data[key] = last_line_dict[key]
        self.map_data = map_data

        #* motion部分
        # 使用正则表达式提取表格中的数据
        # 将 lines 合并为一个单一字符串
        log_data = "\n".join(lines)

        self.motion_data = {}

        pattern = r"\| *([\w\s]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\|"
        matches = re.findall(pattern, log_data)

        # 提取数据
        data = [{'class_name': match[0].strip(), 'min_ADE(m)↓': match[1], 'min_FDE(m)↓': match[2], 'miss_rate↓': match[3]} for match in matches]
        # 查找并打印 "car" 这一行的数据
        self.motion_data['car'] = next(item for item in data if item['class_name'] == 'car')

        # 使用正则表达式提取所有 "EPA vehicle value" 的数据
        pattern_EPA = r"EPA\s+(\w+)\s+([\d\.]+)"
        matches_EPA = re.findall(pattern_EPA, log_data)

        # 获取第一个 "car" 的数据
        for match in matches_EPA:
            vehicle, value = match
            if vehicle == 'car':
                self.motion_data['car']['EPA↑'] = float(value)
                break  # 找到第一个匹配后立即停止循环
        # 提取数据
        data = [{'class_name': match[0].strip(), 'min_ADE(m)↓': match[1], 'min_FDE(m)↓': match[2], 'miss_rate↓': match[3]} for match in matches]
        # 查找并打印 "pedestrian" 这一行的数据
        self.motion_data['pedestrian'] = next(item for item in data if item['class_name'] == 'pedestrian')


        # 获取第一个 "pedestrian" 的数据
        for match in matches_EPA:
            vehicle, value = match
            if vehicle == 'pedestrian':
                self.motion_data['pedestrian']['EPA↑'] = float(value)
                break  # 找到第一个匹配后立即停止循环
    
    def _convert_tensor(self, data):
        """将数据中的所有张量转换为列表"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, dict):
            return {k: self._convert_tensor(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_tensor(item) for item in data]
        return data

    def _load_track_json(self):
        """加载 track json 文件数据"""
        print("Loading track data from: ", self.track_path)
        try:
            with open(self.track_path, 'r') as file:
                self.track_data = json.load(file)
                print("Track data loaded successfully!")
        except FileNotFoundError:
            print(f"文件 {self.track_path} 未找到。")
            return None
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return None
    
    def _load_pkl_data(self):
        """加载 .pkl 文件数据"""
        print("Loading pkl data from: ", self.pkl_path)
        try:
            with open(self.pkl_path, 'rb') as file:
                self.data = pickle.load(file)
                # 从tensor转换为dict
                self.bbox_data = self.data['bbox_results']
                self.occ_data = self.data['occ_results_computed']
                self.planning_data = self.data['planning_results_computed']
                print("pkl data loaded successfully!")
                # 
        except FileNotFoundError:
            print(f"文件 {self.pkl_path} 未找到。")
            return None
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return None
        
        # 关闭文件
        
        self._occ_data = self._convert_tensor(self.occ_data)
        self._planning_data = self._convert_tensor(self.planning_data)
    
    def _load_final_json(self):
        """加载 final json 文件数据"""
        print("Loading final json from: ", self.final_json_path)
        try:
            with open(self.final_json_path, 'r') as file:
                final_json = json.load(file)
                self.occ_data = final_json['occ']
                self.planning_data = final_json['planning']
                self.track_data = final_json['track']
                self.map_data = final_json['map']
                self.motion_data = final_json['motion']
                print("Final json loaded successfully!")
        except FileNotFoundError:
            print(f"文件 {self.final_json_path} 未找到。")
            return None
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return None

    def _analyze_occ(self):
        """分析 OCC 数据"""
        occ = self.occ_data

        # 准备数据用于 DataFrame
        data = {
            "pq": occ['pq'],
            "sq": occ['sq'],
            "rq": occ['rq'],
            "denominator": occ['denominator'],
            "iou": occ['iou'],
            "num_occ": [occ['num_occ']] * len(occ['pq']),  # 重复 num_occ 值以匹配长度
            "ratio_occ": [occ['ratio_occ']] * len(occ['pq'])  # 重复 ratio_occ 值以匹配长度
        }

        # 转换数据为 DataFrame，设置合适的索引
        self.occ_df = pd.DataFrame(data, index=["-n (30m x 30m)", "-f (50m x 50m)"])
        self.pd_data = pd.concat([self.pd_data, self.occ_df], axis=0)

    def _analyze_planning(self):
        """分析 Planning 数据格"""
        planning = self.planning_data
        planning_keys = planning.keys()

        # 如果是tensor
        if isinstance(planning['obj_col'], torch.Tensor):
            data = {
            "obj_col(%)": planning['obj_col'].cpu().numpy() * 100,  # 从 CUDA 转换为 NumPy
            "obj_box_col(%)": planning['obj_box_col'].cpu().numpy() * 100,
            "L2(m)": planning['L2'].cpu().numpy(),
        }
        else:
            data = {
            "obj_col(%)": [item * 100 for item in planning['obj_col']],
            "obj_box_col(%)": [item * 100 for item in planning['obj_box_col']],
            "L2(m)": planning['L2'],
        }

        # 使用时间间隔作为索引
        time_intervals = [f"{(i+1) * 0.5}s" for i in range(len(planning['obj_col']))]
        self.planning_df = pd.DataFrame(data, index=time_intervals).T

        # 计算0.5s到3s的平均间隔0.5s
        avg_0_5s = self.planning_df.mean(axis=0)  
        # 计算1s到3s的平均间隔1s, 即取第2个, 第4个, 第6个...的平均值
        avg_1s = self.planning_df.iloc[1::2].mean(axis=0)

        

        # 将平均值添加到原 DataFrame 的末尾
        self.planning_df.loc['avg_0.5s'] = avg_0_5s
        self.planning_df.loc['avg_1s'] = avg_1s

        self.pd_data = pd.concat([self.pd_data, self.planning_df], axis=0)

    def _analyze_track(self):
        """分析 Track 数据并输出为表格"""
        track = self.track_data

        # 从cfg中获取class名字
        track_cfg = track['cfg']
        tracking_names = track_cfg['tracking_names']

        # 从metrics_summary中获取数据创建表格
        track_data = track['label_metrics']

        self.track_df = pd.DataFrame(track_data, index=tracking_names)
        # 将列标签转换为大写
        self.track_df.columns = self.track_df.columns.str.upper()

        
        # tracking_names中提取track_df列tag作为key绘制pandas表格
        track_result_list = self.track_df.columns.str.lower()

        track_result = {}
        for tag in track_result_list:
            track_result[tag] = track[tag]

        self.track_result_df = pd.DataFrame(track_result, index={'tracking_result'})    

        self.pd_data = pd.concat([self.pd_data, self.track_df], axis=0)
        self.pd_data = pd.concat([self.pd_data, self.track_result_df], axis=0)

    def _analyze_map(self):
        """分析 Map 数据并输出为表格"""
        map_data = self.map_data

        self.map_df = pd.DataFrame(map_data, index={'map_result'})
        self.pd_data = pd.concat([self.pd_data,  self.map_df], axis=0)
    
    def _analyze_motion(self):
        """分析 Motion 数据并输出为表格"""
        motion_data = self.motion_data

        self.motion_df = pd.DataFrame(motion_data).T
        self.pd_data = pd.concat([self.pd_data, self.motion_df], axis=0)

    def show_occ(self):
        print("OCC 数据：")
        # 输出 OCC 表格
        print("\nOCC 表格(越大越好)：")
        print(self.occ_df)

    def show_planning(self):
        print("Planning 数据：")
        planning = self.planning_data
        planning_keys = planning.keys()

        print("Planning 数据分析：")
        for key in planning_keys:
            print(f"{key}: {planning[key]}")

                # 输出转置后的表格
        print("\n规划结果：")
        print(self.planning_df)  # 转置数据

    def show_track(self):
        track = self.track_data
        print("Track 数据：")
        print("评估用时: ", track['eval_time'], "s")
        print("使用传感器 ", track['meta'])

        print("各类agent的track result: ")

        print(self.track_df, "\n")

        print("自车的track结果(论文中评价指标): ")
        print(self.track_result_df, "\n")
    
    def show_map(self):
        print("Map结果: ")
        print( self.map_df)
    
    def show_motion(self):
        print("Motion 数据：")
        print("Motion的Car和Pedestrian result: ")
        print(self.motion_df)

    def show_all(self):
        self.show_occ()
        self.show_planning()
        self.show_track()
        self.show_map()
        self.show_motion()

    def show_keys(self):
        """显示数据的键"""
        print("数据的key：")
        print("bbox keys: ", self.bbox_data[0].keys())
        print("occ keys: ", self.occ_data.keys())
        print("planning keys: ", self.planning_data.keys())
        print("track keys: ", self.track_data.keys())
        print("map keys: ", self.map_data.keys())
        print("motion keys: ", self.motion_data.keys())

    def save_final_data(self, output_json_path: str, output_pd_path: str):
        """保存最终的json文件"""
        print("Saving final json to: ", output_json_path)
        final_json = {
            "occ": self._convert_tensor(self.occ_data),
            "planning": self._convert_tensor(self.planning_data),
            "track": self._convert_tensor(self.track_data),
            "map": self._convert_tensor(self.map_data),
            "motion": self._convert_tensor(self.motion_data)
        }

        with open(output_json_path, 'w') as file:
            json.dump(final_json, file, indent=4)
        print("Final json saved successfully!")

        print("Saving pandas data to: ", output_pd_path)
        self.pd_data.to_csv(output_pd_path)
