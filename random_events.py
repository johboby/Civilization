#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文明演化模拟系统 - 随机事件模块

此模块实现了随机事件系统，为模拟增加不确定性和真实性
包括自然灾害、技术突破、社会变革等事件
"""

import numpy as np
import random
from collections import defaultdict

class RandomEventManager:
    """随机事件管理器"""
    
    def __init__(self, config=None):
        """初始化随机事件管理器
        
        Args:
            config: 模拟配置对象
        """
        self.config = config
        self.events = self._initialize_events()
        self.event_history = []
        
    def _initialize_events(self):
        """初始化事件库"""
        events = {
            # 自然灾害事件
            "minor_disaster": {
                "name": "小规模自然灾害",
                "description": "发生了一次小规模的自然灾害，对部分文明造成影响",
                "probability": 0.15,  # 基础发生概率
                "effect": self._minor_disaster_effect,
                "severity": "minor",
                "categories": ["natural_disaster"]
            },
            "major_disaster": {
                "name": "重大自然灾害",
                "description": "发生了一次重大的自然灾害，对多个文明造成严重影响",
                "probability": 0.05,
                "effect": self._major_disaster_effect,
                "severity": "major",
                "categories": ["natural_disaster"]
            },
            
            # 技术突破事件
            "tech_breakthrough": {
                "name": "技术突破",
                "description": "一项重要技术被意外发现，加速了科技发展",
                "probability": 0.08,
                "effect": self._tech_breakthrough_effect,
                "severity": "positive",
                "categories": ["technology"]
            },
            
            # 社会变革事件
            "social_reform": {
                "name": "社会改革",
                "description": "一次重要的社会改革提升了文明的组织效率",
                "probability": 0.07,
                "effect": self._social_reform_effect,
                "severity": "positive",
                "categories": ["social"]
            },
            
            # 资源发现事件
            "resource_discovery": {
                "name": "资源发现",
                "description": "发现了新的资源储备，提升了资源产量",
                "probability": 0.1,
                "effect": self._resource_discovery_effect,
                "severity": "positive",
                "categories": ["resource"]
            }
        }
        return events
    
    def trigger_event(self, agents, cycle):
        """触发随机事件
        
        Args:
            agents: 文明智能体列表
            cycle: 当前周期
            
        Returns:
            dict: 触发的事件信息，如果没有事件则返回None
        """
        # 根据配置调整事件概率
        event_probability_modifier = getattr(self.config, 'EVENT_PROBABILITY_MODIFIER', 1.0)
        
        # 遍历所有事件，检查是否触发
        for event_id, event_info in self.events.items():
            adjusted_probability = event_info["probability"] * event_probability_modifier
            
            if random.random() < adjusted_probability:
                # 触发事件
                affected_agents = event_info["effect"](agents)
                
                event_record = {
                    "cycle": cycle,
                    "event_id": event_id,
                    "event_name": event_info["name"],
                    "description": event_info["description"],
                    "affected_agents": affected_agents,
                    "severity": event_info["severity"]
                }
                
                self.event_history.append(event_record)
                
                # 打印事件信息（如果启用了日志）
                if hasattr(self.config, 'PRINT_LOGS') and self.config.PRINT_LOGS:
                    print(f"周期 {cycle}: {event_info['name']} - {event_info['description']}")
                    if affected_agents:
                        print(f"  影响文明: {affected_agents}")
                
                return event_record
        
        return None
    
    def _minor_disaster_effect(self, agents):
        """小规模自然灾害效果"""
        # 随机选择1-2个文明受到影响
        affected_agents = random.sample(agents, min(random.randint(1, 2), len(agents)))
        
        for agent in affected_agents:
            # 减少资源和人口
            agent.resources *= random.uniform(0.7, 0.9)
            agent.population *= random.uniform(0.8, 0.95)
            agent.strength *= random.uniform(0.85, 0.95)
            
        return [agent.agent_id for agent in affected_agents]
    
    def _major_disaster_effect(self, agents):
        """重大自然灾害效果"""
        # 随机选择一半以上的文明受到影响
        affected_count = max(1, len(agents) // 2 + random.randint(0, len(agents) // 2))
        affected_agents = random.sample(agents, min(affected_count, len(agents)))
        
        for agent in affected_agents:
            # 严重减少资源、人口和军事力量
            agent.resources *= random.uniform(0.4, 0.7)
            agent.population *= random.uniform(0.6, 0.8)
            agent.strength *= random.uniform(0.5, 0.7)
            
        return [agent.agent_id for agent in affected_agents]
    
    def _tech_breakthrough_effect(self, agents):
        """技术突破效果"""
        # 随机选择1个文明获得技术突破
        affected_agent = random.choice(agents)
        
        # 增加科技点数或直接提升科技等级
        tech_to_boost = random.choice(list(affected_agent.technology.keys()))
        current_level = affected_agent.technology[tech_to_boost]
        affected_agent.technology[tech_to_boost] = min(5, current_level + random.randint(1, 2))  # 最多提升2级
        
        # 更新科技加成
        affected_agent.update_tech_bonuses()
        
        return [affected_agent.agent_id]
    
    def _social_reform_effect(self, agents):
        """社会改革效果"""
        # 随机选择1个文明进行社会改革
        affected_agent = random.choice(agents)
        
        # 提升效率相关属性
        affected_agent.infrastructure *= random.uniform(1.1, 1.3)
        affected_agent.stability = min(1.5, affected_agent.stability * random.uniform(1.1, 1.2))
        affected_agent.decision_quality *= random.uniform(1.1, 1.2)
        
        return [affected_agent.agent_id]
    
    def _resource_discovery_effect(self, agents):
        """资源发现效果"""
        # 随机选择1-2个文明发现新资源
        affected_count = random.randint(1, min(2, len(agents)))
        affected_agents = random.sample(agents, affected_count)
        
        for agent in affected_agents:
            # 增加资源储备
            agent.resources *= random.uniform(1.3, 1.8)
            
        return [agent.agent_id for agent in affected_agents]
    
    def get_event_history(self):
        """获取事件历史记录"""
        return self.event_history
    
    def save_event_history(self, filename="event_history.json"):
        """保存事件历史到文件"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.event_history, f, ensure_ascii=False, indent=2)

# 示例使用
if __name__ == "__main__":
    # 创建一个简单的配置对象
    class DummyConfig:
        EVENT_PROBABILITY_MODIFIER = 1.0
        PRINT_LOGS = True
    
    # 初始化事件管理器
    config = DummyConfig()
    event_manager = RandomEventManager(config)
    
    print("随机事件系统测试:")
    print("可用事件:")
    for event_id, event_info in event_manager.events.items():
        print(f"  - {event_info['name']}: {event_info['description']}")