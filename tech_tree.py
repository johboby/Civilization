import numpy as np

class TechTree:
    """Civilization Technology Tree System"""
    def __init__(self):
        # 定义科技树结构
        self.techs = {
            # 基础科技
            "agriculture": {"level": 1, "cost": 50, "description": "Agriculture Technology", "prerequisites": []},
            "military": {"level": 1, "cost": 80, "description": "Military Technology", "prerequisites": []},
            "trade": {"level": 1, "cost": 60, "description": "Trade Technology", "prerequisites": []},
            "science": {"level": 1, "cost": 100, "description": "Scientific Research", "prerequisites": []},
            
            # 中级科技
            "irrigation": {"level": 2, "cost": 200, "description": "Irrigation System", "prerequisites": ["agriculture"]},
            "fortification": {"level": 2, "cost": 250, "description": "Fortification", "prerequisites": ["military"]},
            "currency": {"level": 2, "cost": 220, "description": "Currency System", "prerequisites": ["trade"]},
            "engineering": {"level": 2, "cost": 300, "description": "Engineering", "prerequisites": ["science"]},
            
            # 高级科技
            "industrial_agriculture": {"level": 3, "cost": 800, "description": "Industrial Agriculture", "prerequisites": ["irrigation", "engineering"]},
            "advanced_tactics": {"level": 3, "cost": 900, "description": "Advanced Tactics", "prerequisites": ["fortification", "engineering"]},
            "global_trade": {"level": 3, "cost": 750, "description": "Global Trade", "prerequisites": ["currency", "engineering"]},
            "advanced_science": {"level": 3, "cost": 1000, "description": "Advanced Science", "prerequisites": ["engineering", "science"]},
            
            # 顶级科技（新增）
            "genetic_engineering": {"level": 4, "cost": 2000, "description": "Genetic Engineering", "prerequisites": ["industrial_agriculture", "advanced_science"]},
            "nuclear_technology": {"level": 4, "cost": 2500, "description": "Nuclear Technology", "prerequisites": ["advanced_tactics", "advanced_science"]},
            "space_colonization": {"level": 4, "cost": 3000, "description": "Space Colonization", "prerequisites": ["global_trade", "advanced_science"]},
            "artificial_intelligence": {"level": 4, "cost": 3500, "description": "Artificial Intelligence", "prerequisites": ["advanced_science"]}
        }
        
        # Technology effects on civilization attributes
        self.tech_effects = {
            "agriculture": {"resources": 0.1, "territory_growth": 0.05},
            "military": {"strength": 0.15, "defense": 0.1},
            "trade": {"resources": 0.08, "diplomacy": 0.15},
            "science": {"research_speed": 0.2, "tech_discovery": 0.1},
            "irrigation": {"resources": 0.15, "territory_value": 0.1},
            "fortification": {"defense": 0.2, "stability": 0.1},
            "currency": {"resources": 0.12, "trade_efficiency": 0.15},
            "engineering": {"research_speed": 0.15, "infrastructure": 0.2},
            "industrial_agriculture": {"resources": 0.3, "population_growth": 0.2},
            "advanced_tactics": {"strength": 0.3, "tactical_advantage": 0.25},
            "global_trade": {"resources": 0.25, "diplomacy": 0.2},
            "advanced_science": {"research_speed": 0.35, "innovation": 0.3},
            # 顶级科技效果（新增）
            "genetic_engineering": {"resources": 0.5, "population_growth": 0.35, "health": 0.3},
            "nuclear_technology": {"strength": 0.6, "defense": 0.4, "energy_efficiency": 0.5},
            "space_colonization": {"territory_growth": 0.8, "resource_acquisition": 0.6, "global_influence": 0.5},
            "artificial_intelligence": {"research_speed": 0.8, "innovation": 0.6, "decision_quality": 0.7}
        }

    def get_tech_category(self, tech_name):
        """获取科技的分类"""
        if tech_name in ["agriculture", "irrigation", "industrial_agriculture"]:
            return "农业"
        elif tech_name in ["military", "fortification", "advanced_tactics"]:
            return "军事"
        elif tech_name in ["trade", "currency", "global_trade"]:
            return "贸易"
        elif tech_name in ["science", "engineering", "advanced_science"]:
            return "科学"
        elif tech_name in ["genetic_engineering", "nuclear_technology", "space_colonization", "artificial_intelligence"]:
            return "顶级科技"
        else:
            return "其他"

    def get_tech_impact_description(self, tech_name):
        """获取科技影响的描述"""
        if tech_name not in self.tech_effects:
            return "无特殊效果"
            
        effects = self.tech_effects[tech_name]
        descriptions = []
        
        for attribute, bonus in effects.items():
            if attribute == "resources":
                descriptions.append(f"资源产出+{bonus*100:.0f}%")
            elif attribute == "strength":
                descriptions.append(f"军事力量+{bonus*100:.0f}%")
            elif attribute == "defense":
                descriptions.append(f"防御能力+{bonus*100:.0f}%")
            elif attribute == "research_speed":
                descriptions.append(f"研发速度+{bonus*100:.0f}%")
            elif attribute == "population_growth":
                descriptions.append(f"人口增长+{bonus*100:.0f}%")
            elif attribute == "territory_growth":
                descriptions.append(f"领土扩张+{bonus*100:.0f}%")
            elif attribute == "health":
                descriptions.append(f"健康水平+{bonus*100:.0f}%")
            elif attribute == "energy_efficiency":
                descriptions.append(f"能源效率+{bonus*100:.0f}%")
            elif attribute == "resource_acquisition":
                descriptions.append(f"资源获取+{bonus*100:.0f}%")
            elif attribute == "global_influence":
                descriptions.append(f"全球影响+{bonus*100:.0f}%")
            elif attribute == "decision_quality":
                descriptions.append(f"决策质量+{bonus*100:.0f}%")
                
        return "，".join(descriptions)

    def can_research(self, tech_name, current_techs):
        """Check if specified technology can be researched"""
        if tech_name not in self.techs:
            return False, "Technology does not exist"
        
        tech = self.techs[tech_name]
        
        # 检查前置科技
        for prerequisite in tech["prerequisites"]:
            if prerequisite not in current_techs or current_techs[prerequisite] < 1:
                return False, f"Missing prerequisite technology: {prerequisite}"
        
        return True, "Can be researched"
    
    def get_research_cost(self, tech_name, current_level):
        """Calculate the cost of researching specified technology"""
        if tech_name not in self.techs:
            return 0
        
        base_cost = self.techs[tech_name]["cost"]
        # 成本随等级指数增长
        cost_multiplier = 1.5 ** (current_level - 1)
        
        return int(base_cost * cost_multiplier)
    
    def calculate_tech_bonuses(self, current_techs):
        """Calculate attribute bonuses from current technologies"""
        bonuses = {
            "resources": 1.0,
            "strength": 1.0,
            "defense": 1.0,
            "research_speed": 1.0,
            "diplomacy": 1.0,
            "territory_growth": 1.0,
            "territory_value": 1.0,
            "trade_efficiency": 1.0,
            "infrastructure": 1.0,
            "population_growth": 1.0,
            "tactical_advantage": 1.0,
            "tech_discovery": 1.0,
            "stability": 1.0,
            "innovation": 1.0,
            # 新增属性
            "health": 1.0,
            "energy_efficiency": 1.0,
            "resource_acquisition": 1.0,
            "global_influence": 1.0,
            "decision_quality": 1.0
        }
        
        for tech_name, level in current_techs.items():
            if tech_name in self.tech_effects and level > 0:
                for attribute, bonus_per_level in self.tech_effects[tech_name].items():
                    if attribute in bonuses:
                        # 每个科技等级提供的加成累加
                        bonuses[attribute] += bonus_per_level * level
        
        return bonuses
    
    def get_available_techs(self, current_techs):
        """Get list of currently available technologies for research"""
        available = []
        
        for tech_name in self.techs.keys():
            if tech_name not in current_techs or current_techs[tech_name] < self.techs[tech_name]["level"]:
                can_research, _ = self.can_research(tech_name, current_techs)
                if can_research:
                    available.append({
                        "name": tech_name,
                        "level": self.techs[tech_name]["level"],
                        "cost": self.get_research_cost(tech_name, current_techs.get(tech_name, 0) + 1),
                        "description": self.techs[tech_name]["description"]
                    })
        
        return available
    
    def get_tech_tree_summary(self):
        """Get technology tree summary"""
        summary = "Technology Tree System\n"
        summary += "=" * 50 + "\n"
        
        # 按等级分组显示科技
        techs_by_level = {1: [], 2: [], 3: []}
        for tech_name, tech_info in self.techs.items():
            techs_by_level[tech_info["level"]].append((tech_name, tech_info))
        
        for level in sorted(techs_by_level.keys()):
            summary += f"\nLevel {level} Technologies:\n"
            for tech_name, tech_info in techs_by_level[level]:
                prereqs = ", ".join(tech_info["prerequisites"]) if tech_info["prerequisites"] else "None"
                summary += f"  - {tech_name}: {tech_info['description']} (Prerequisites: {prereqs})\n"
        
        return summary

# 测试代码
if __name__ == "__main__":
    tech_tree = TechTree()
    
    # 测试初始科技状态
    initial_techs = {"agriculture": 1, "military": 1, "trade": 1}
    
    # 获取可研究的科技
    available = tech_tree.get_available_techs(initial_techs)
    print("可研究的科技:")
    for tech in available:
        print(f"  {tech['name']}: {tech['description']} (成本: {tech['cost']})")
    
    # 计算科技加成
    bonuses = tech_tree.calculate_tech_bonuses(initial_techs)
    print("\n当前科技加成:")
    for attr, bonus in bonuses.items():
        if bonus > 1.0:
            print(f"  {attr}: {bonus:.2f}x")
    
    # 打印科技树摘要
    print(tech_tree.get_tech_tree_summary())