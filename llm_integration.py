"""
LLM集成模块 - 支持OpenAI API和兼容接口
"""

import os
import yaml
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
import json
from abc import ABC, abstractmethod

# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retry_times: int = 3
    retry_delay: int = 1

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    llm: LLMConfig
    system: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 解析LLM配置
        llm_data = data.get('llm', {})
        llm_config = LLMConfig(
            provider=llm_data.get('provider', 'openai'),
            base_url=llm_data.get('base_url', 'https://api.openai.com/v1'),
            api_key=llm_data.get('api_key', os.getenv('OPENAI_API_KEY', '')),
            model=llm_data.get('model', 'gpt-4'),
            temperature=llm_data.get('temperature', 0.7),
            max_tokens=llm_data.get('max_tokens', 2000),
            timeout=llm_data.get('timeout', 60),
            retry_times=llm_data.get('retry_times', 3),
            retry_delay=llm_data.get('retry_delay', 1)
        )
        
        return cls(
            name=data.get('name', 'default_experiment'),
            description=data.get('description', ''),
            llm=llm_config,
            system=data.get('system', {}),
            agents=data.get('agents', {}),
            tasks=data.get('tasks', [])
        )

# ============================================================================
# LLM接口抽象
# ============================================================================

class LLMInterface(ABC):
    """LLM接口抽象基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    async def generate_with_schema(self, prompt: str, response_schema: Dict[str, Any], 
                                  system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """生成符合特定模式的响应"""
        pass

# ============================================================================
# OpenAI API实现
# ============================================================================

class OpenAIInterface(LLMInterface):
    """OpenAI API接口实现"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """调用OpenAI API生成响应"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # 添加其他可选参数
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            request_data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_data["presence_penalty"] = kwargs["presence_penalty"]
        
        # 重试机制
        for attempt in range(self.config.retry_times):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.base_url}/chat/completions",
                        headers=self.headers,
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API error: {response.status} - {error_text}")
                            
                            if response.status == 429:  # Rate limit
                                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                                continue
                            
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout (attempt {attempt + 1}/{self.config.retry_times})")
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
            
            if attempt < self.config.retry_times - 1:
                await asyncio.sleep(self.config.retry_delay)
        
        raise Exception("Failed to get response from LLM after all retries")
    
    async def generate_with_schema(self, prompt: str, response_schema: Dict[str, Any], 
                                  system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """生成符合JSON模式的响应"""
        schema_prompt = f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(response_schema, indent=2)}"
        
        full_prompt = prompt + schema_prompt
        
        response = await self.generate(full_prompt, system_prompt, **kwargs)
        
        # 尝试解析JSON
        try:
            # 清理响应（移除可能的markdown标记）
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            return json.loads(cleaned_response.strip())
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Raw response: {response}")
            raise

# ============================================================================
# LLM管理器
# ============================================================================

class LLMManager:
    """LLM管理器 - 管理所有LLM接口"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("LLMManager")
        
        # 根据provider创建接口
        if config.provider == "openai":
            self.interface = OpenAIInterface(config)
        else:
            # 可以扩展其他provider
            self.interface = OpenAIInterface(config)  # 默认使用OpenAI接口
        
        self.logger.info(f"LLM Manager initialized with {config.provider} provider")
    
    async def generate_code(self, task_type: str, requirements: Dict[str, Any], 
                           approach: str = "standard", agent_profile: Dict[str, Any] = None) -> str:
        """生成Verilog代码"""
        
        system_prompt = self._build_code_generation_system_prompt(agent_profile)
        
        prompt = f"""Generate Verilog code for the following task:

Task Type: {task_type}
Approach: {approach}

Requirements:
{json.dumps(requirements, indent=2)}

Please generate clean, well-commented Verilog code that meets all requirements.
"""
        
        return await self.interface.generate(prompt, system_prompt)
    
    async def review_code(self, code: str, requirements: Dict[str, Any], 
                         reviewer_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """审查Verilog代码"""
        
        system_prompt = self._build_review_system_prompt(reviewer_profile)
        
        prompt = f"""Review the following Verilog code:

```verilog
{code}
```

Requirements that the code should meet:
{json.dumps(requirements, indent=2)}

Please provide a detailed review including:
1. Syntax issues
2. Functional correctness
3. Style and best practices
4. Optimization suggestions
"""
        
        response_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["approved", "needs_improvement", "rejected"]},
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "severity": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                            "description": {"type": "string"},
                            "line": {"type": "number"},
                            "suggestion": {"type": "string"}
                        }
                    }
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        
        return await self.interface.generate_with_schema(prompt, response_schema, system_prompt)
    
    async def analyze_task(self, task_description: str, agent_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析任务并提供决策建议"""
        
        system_prompt = self._build_analysis_system_prompt(agent_profile)
        
        prompt = f"""Analyze the following task and provide decision recommendations:

Task: {task_description}

Consider:
1. Task complexity
2. Required capabilities
3. Potential collaboration needs
4. Approach recommendations
"""
        
        response_schema = {
            "type": "object",
            "properties": {
                "complexity": {"type": "number", "minimum": 0, "maximum": 1},
                "should_accept": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "required_capabilities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "collaboration_needed": {"type": "boolean"},
                "suggested_approach": {"type": "string"}
            }
        }
        
        return await self.interface.generate_with_schema(prompt, response_schema, system_prompt)
    
    async def generate_testbench(self, module_code: str, test_requirements: Dict[str, Any] = None) -> str:
        """生成测试台"""
        
        system_prompt = "You are an expert Verilog testbench generator."
        
        prompt = f"""Generate a comprehensive testbench for the following Verilog module:

```verilog
{module_code}
```

Test Requirements:
{json.dumps(test_requirements, indent=2) if test_requirements else "Standard functional verification"}

Please include:
1. Clock generation
2. Reset sequence
3. Comprehensive test cases
4. Result checking
5. Waveform generation setup
"""
        
        return await self.interface.generate(prompt, system_prompt)
    
    def _build_code_generation_system_prompt(self, agent_profile: Dict[str, Any] = None) -> str:
        """构建代码生成的系统提示"""
        base_prompt = "You are an expert Verilog code generator."
        
        if agent_profile:
            traits = agent_profile.get("personality_traits", {})
            
            if traits.get("creative", 0) > 0.7:
                base_prompt += " You approach problems with creativity and innovation."
            
            if traits.get("detail_oriented", 0) > 0.7:
                base_prompt += " You pay meticulous attention to detail and code quality."
            
            style = agent_profile.get("communication_style", "")
            if style == "analytical":
                base_prompt += " You provide thorough analysis and reasoning in your code comments."
        
        return base_prompt
    
    def _build_review_system_prompt(self, reviewer_profile: Dict[str, Any] = None) -> str:
        """构建代码审查的系统提示"""
        base_prompt = "You are an expert Verilog code reviewer."
        
        if reviewer_profile:
            traits = reviewer_profile.get("personality_traits", {})
            
            if traits.get("critical", 0) > 0.7:
                base_prompt += " You are thorough and critical in your reviews, catching even minor issues."
            
            if traits.get("helpful", 0) > 0.7:
                base_prompt += " You provide constructive feedback with clear improvement suggestions."
        
        return base_prompt
    
    def _build_analysis_system_prompt(self, agent_profile: Dict[str, Any] = None) -> str:
        """构建任务分析的系统提示"""
        base_prompt = "You are an AI agent specialized in analyzing Verilog design tasks."
        
        if agent_profile:
            role = agent_profile.get("role", "")
            if role == "code_generator":
                base_prompt += " You excel at identifying code generation opportunities."
            elif role == "code_reviewer":
                base_prompt += " You focus on quality and correctness aspects."
        
        return base_prompt

# ============================================================================
# 实验管理器
# ============================================================================

class ExperimentManager:
    """实验管理器 - 管理整个实验流程"""
    
    def __init__(self, config_path: str):
        self.config = ExperimentConfig.from_yaml(config_path)
        self.llm_manager = LLMManager(self.config.llm)
        self.logger = logging.getLogger("ExperimentManager")
        
        self.logger.info(f"Loaded experiment: {self.config.name}")
    
    async def run_experiment(self):
        """运行实验"""
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Description: {self.config.description}")
        
        # 这里可以集成到UnifiedAutonomousFramework
        # 示例：运行配置中定义的任务
        for task_config in self.config.tasks:
            await self._run_task(task_config)
    
    async def _run_task(self, task_config: Dict[str, Any]):
        """运行单个任务"""
        self.logger.info(f"Running task: {task_config.get('name', 'unnamed')}")
        
        # 示例：使用LLM生成代码
        if task_config.get("type") == "code_generation":
            code = await self.llm_manager.generate_code(
                task_type=task_config.get("task_type"),
                requirements=task_config.get("requirements"),
                approach=task_config.get("approach", "standard")
            )
            
            self.logger.info(f"Generated code:\n{code}")