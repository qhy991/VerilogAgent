"""
集成LLM的统一自主Verilog框架
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from unified_autonomous_framework import *
from llm_integration import LLMManager, ExperimentConfig

# ============================================================================
# 增强的AutonomousAgent - 集成LLM
# ============================================================================

class LLMEnabledAutonomousAgent(AutonomousAgent):
    """集成LLM能力的自主智能体"""
    
    def __init__(self, profile: UnifiedAgentProfile, 
                 system_coordinator: SystemCoordinator,
                 llm_manager: LLMManager):
        super().__init__(profile, system_coordinator)
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"LLMAgent.{profile.name}")
    
    async def _decide_on_task(self, task: UnifiedTask) -> bool:
        """使用LLM辅助任务决策"""
        # 获取LLM分析
        analysis = await self.llm_manager.analyze_task(
            task.description,
            agent_profile={
                "role": self.profile.role.value,
                "personality_traits": {
                    k.value if hasattr(k, 'value') else k: v 
                    for k, v in self.profile.personality_traits.items()
                },
                "current_workload": self.profile.current_workload,
                "capabilities": {
                    k.value: v for k, v in self.profile.capabilities.items()
                }
            }
        )
        
        # 结合LLM建议和个性特征做决策
        base_decision = analysis.get("should_accept", False)
        
        # 个性调整
        if self.profile.personality_traits.get(PersonalityTrait.CAUTIOUS, 0.5) > 0.7:
            # 谨慎的智能体需要更高的信心
            if analysis.get("complexity", 0.5) > 0.7:
                base_decision = False
        
        if self.profile.personality_traits.get(PersonalityTrait.AMBITIOUS, 0.5) > 0.7:
            # 有野心的智能体更愿意接受挑战
            if analysis.get("complexity", 0.5) > 0.5:
                base_decision = True
        
        # 记录决策
        self.logger.info(f"Task decision for {task.task_id}: {base_decision} (LLM complexity: {analysis.get('complexity', 'N/A')})")
        
        return base_decision
    
    async def _generate_response(self, original_message: UnifiedMessage, 
                                response_decision: Dict[str, Any]) -> Optional[UnifiedMessage]:
        """使用LLM生成响应"""
        # 对于某些响应类型，使用LLM
        if response_decision.get("use_llm", False):
            # 构建上下文
            context = {
                "message_type": original_message.type,
                "sender": original_message.sender,
                "content": original_message.content,
                "agent_role": self.profile.role.value,
                "communication_style": self.profile.communication_style.value
            }
            
            # 使用LLM生成响应内容
            llm_response = await self.llm_manager.interface.generate(
                prompt=f"Generate a response to: {original_message.content}",
                system_prompt=f"You are {self.profile.name}, a {self.profile.role.value} with {self.profile.communication_style.value} communication style."
            )
            
            response_decision["response_content"] = llm_response
        
        return await super()._generate_response(original_message, response_decision)

# ============================================================================
# LLM增强的CoderAgent
# ============================================================================

class LLMCoderAgent(CoderAgent, LLMEnabledAutonomousAgent):
    """集成LLM的代码生成智能体"""
    
    def __init__(self, name: str, system_coordinator: SystemCoordinator, llm_manager: LLMManager):
        # 初始化CoderAgent的profile
        CoderAgent.__init__(self, name, system_coordinator)
        # 添加LLM能力
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"LLMCoderAgent.{name}")
    
    async def _execute_coding_task(self, task_id: str, task: UnifiedTask):
        """使用LLM执行编码任务"""
        self.logger.info(f"Starting LLM-powered code generation for task {task_id}")
        
        # 更新任务状态
        self.active_tasks[task_id]["status"] = "in_progress"
        
        try:
            # 分析任务需求
            analysis = await self._analyze_requirements(task)
            
            # 选择设计方法
            approach = self._select_design_approach(task, analysis)
            
            # 使用LLM生成代码
            generated_code = await self.llm_manager.generate_code(
                task_type=task.task_type.value,
                requirements=task.requirements,
                approach=approach,
                agent_profile={
                    "name": self.profile.name,
                    "personality_traits": {
                        k.value if hasattr(k, 'value') else k: v 
                        for k, v in self.profile.personality_traits.items()
                    },
                    "communication_style": self.profile.communication_style.value
                }
            )
            
            # 自我审查（如果性格倾向于此）
            if self.profile.personality_traits.get(PersonalityTrait.DETAIL_ORIENTED, 0.5) > 0.6:
                generated_code = await self._self_review_code(generated_code)
            
            # 记录到历史
            self.generation_history.append({
                "task_id": task_id,
                "approach": approach,
                "success": True,
                "timestamp": time.time(),
                "llm_used": True
            })
            
            # 准备交付
            result_message = UnifiedMessage(
                type="task_result",
                content={
                    "task_id": task_id,
                    "status": "completed",
                    "artifacts": {
                        "verilog_code": generated_code,
                        "design_approach": approach,
                        "metadata": {
                            "lines_of_code": len(generated_code.split('\n')),
                            "generation_time": time.time() - self.active_tasks[task_id]["started_at"],
                            "llm_model": self.llm_manager.config.model
                        }
                    },
                    "agent": self.profile.name,
                    "ready_for_review": True
                },
                sender=self.profile.name,
                receivers=["SystemCoordinator", "ReviewerAgent"],
                priority=MessagePriority.HIGH,
                intent=MessageIntent.TASK_RESULT
            )
            
            # 发送结果
            await self.message_router.route_message(result_message, self.profile.name)
            
            # 更新任务状态
            self.active_tasks[task_id]["status"] = "completed"
            
            # 学习记录
            self.experience_buffer.append({
                "task_type": task.task_type.value,
                "complexity": task.complexity,
                "approach": approach,
                "success": True,
                "duration": time.time() - self.active_tasks[task_id]["started_at"],
                "llm_model": self.llm_manager.config.model
            })
            
        except Exception as e:
            self.logger.error(f"Failed to generate code for task {task_id}: {e}")
            self.active_tasks[task_id]["status"] = "failed"
            
            # 请求帮助
            if self.profile.personality_traits.get(PersonalityTrait.COLLABORATIVE, 0.5) > 0.6:
                await self._request_help_for_task(task_id, str(e))

# ============================================================================
# LLM增强的ReviewerAgent
# ============================================================================

class LLMReviewerAgent(ReviewerAgent, LLMEnabledAutonomousAgent):
    """集成LLM的代码审查智能体"""
    
    def __init__(self, name: str, system_coordinator: SystemCoordinator, llm_manager: LLMManager):
        ReviewerAgent.__init__(self, name, system_coordinator)
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"LLMReviewerAgent.{name}")
    
    async def _start_review_process(self, message: UnifiedMessage):
        """使用LLM进行代码审查"""
        content = message.content
        task_id = content.get("task_id")
        code = content.get("artifacts", {}).get("verilog_code")
        
        if not code:
            self.logger.warning(f"No code found for review in task {task_id}")
            return
        
        self.logger.info(f"Starting LLM-powered review for task {task_id}")
        
        # 使用LLM审查代码
        review_result = await self.llm_manager.review_code(
            code=code,
            requirements=content.get("requirements", {}),
            reviewer_profile={
                "name": self.profile.name,
                "personality_traits": {
                    k.value if hasattr(k, 'value') else k: v 
                    for k, v in self.profile.personality_traits.items()
                },
                "communication_style": self.profile.communication_style.value
            }
        )
        
        # 生成审查报告
        report = self._generate_review_report(task_id, review_result)
        
        # 记录到历史
        self.review_history.append({
            "task_id": task_id,
            "issues_found": len(review_result.get("issues", [])),
            "severity_distribution": self._calculate_severity_distribution(review_result.get("issues", [])),
            "timestamp": time.time(),
            "reviewed_agent": message.sender,
            "llm_used": True
        })
        
        # 准备审查结果消息
        review_message = UnifiedMessage(
            type="review_result",
            content={
                "task_id": task_id,
                "review_status": review_result.get("status", "needs_improvement"),
                "issues": review_result.get("issues", []),
                "suggestions": review_result.get("suggestions", []),
                "score": review_result.get("score", 50),
                "detailed_report": report,
                "reviewer": self.profile.name,
                "llm_model": self.llm_manager.config.model
            },
            sender=self.profile.name,
            receivers=[message.sender, "ExecutorAgent", "SystemCoordinator"],
            priority=MessagePriority.HIGH if review_result.get("status") == "rejected" else MessagePriority.NORMAL,
            intent=MessageIntent.REVIEW_RESULT
        )
        
        # 发送审查结果
        await self.message_router.route_message(review_message, self.profile.name)
        
        # 学习记录
        self.experience_buffer.append({
            "task_type": "code_review",
            "complexity": self._estimate_review_complexity(content),
            "issues_found": len(review_result.get("issues", [])),
            "success": True,
            "duration": time.time() - message.timestamp if hasattr(message, 'timestamp') else 0,
            "llm_model": self.llm_manager.config.model
        })
    
    def _calculate_severity_distribution(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算严重性分布"""
        distribution = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in issues:
            severity = issue.get("severity", "LOW")
            if severity in distribution:
                distribution[severity] += 1
        return distribution

# ============================================================================
# LLM增强的ExecutorAgent
# ============================================================================

class LLMExecutorAgent(ExecutorAgent, LLMEnabledAutonomousAgent):
    """集成LLM的代码执行智能体"""
    
    def __init__(self, name: str, system_coordinator: SystemCoordinator, llm_manager: LLMManager):
        ExecutorAgent.__init__(self, name, system_coordinator)
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(f"LLMExecutorAgent.{name}")
    
    async def generate_testbench(self, dut_code: str, task_id: str) -> str:
        """使用LLM生成测试台"""
        # 分析DUT接口
        interface = self._analyze_interface(dut_code)
        
        # 使用LLM生成测试台
        testbench = await self.llm_manager.generate_testbench(
            module_code=dut_code,
            test_requirements={
                "coverage_target": "functional",
                "test_scenarios": ["normal_operation", "edge_cases", "error_conditions"],
                "simulation_time": 1000,
                "agent_profile": {
                    "name": self.profile.name,
                    "focus": "comprehensive_testing"
                }
            }
        )
        
        return testbench

# ============================================================================
# 集成LLM的统一框架
# ============================================================================

class LLMUnifiedAutonomousFramework(UnifiedAutonomousFramework):
    """集成LLM的统一自主框架"""
    
    def __init__(self, experiment_config_path: str):
        super().__init__()
        
        # 加载实验配置
        self.experiment_config = ExperimentConfig.from_yaml(experiment_config_path)
        
        # 创建LLM管理器
        self.llm_manager = LLMManager(self.experiment_config.llm)
        
        self.logger.info(f"LLM-enabled framework initialized with experiment: {self.experiment_config.name}")
    
    async def _create_default_agent_team(self):
        """创建支持LLM的默认智能体团队"""
        # 从配置文件读取智能体配置
        agents_config = self.experiment_config.agents
        
        for agent_key, agent_config in agents_config.items():
            name = agent_config.get("name", agent_key)
            role_str = agent_config.get("role", "code_generator")
            
            # 转换角色
            role_map = {
                "code_generator": AgentRole.CODE_GENERATOR,
                "code_reviewer": AgentRole.CODE_REVIEWER,
                "code_executor": AgentRole.CODE_EXECUTOR
            }
            role = role_map.get(role_str, AgentRole.CODE_GENERATOR)
            
            # 创建profile
            profile = UnifiedAgentProfile(
                name=name,
                role=role,
                communication_style=CommunicationStyle(agent_config.get("communication_style", "balanced")),
                learning_rate=agent_config.get("learning_rate", 0.15),
                proactivity=agent_config.get("personality", {}).get("proactive", 0.6)
            )
            
            # 设置个性特征
            personality_config = agent_config.get("personality", {})
            for trait_name, value in personality_config.items():
                # 尝试匹配PersonalityTrait枚举
                for trait in PersonalityTrait:
                    if trait.value == trait_name or trait_name in trait.value:
                        profile.personality_traits[trait] = value
                        break
            
            # 添加角色特定能力
            self._add_role_capabilities(profile)
            
            # 根据角色创建相应的LLM增强智能体
            if role == AgentRole.CODE_GENERATOR:
                agent = LLMCoderAgent(name, self.system_coordinator, self.llm_manager)
            elif role == AgentRole.CODE_REVIEWER:
                agent = LLMReviewerAgent(name, self.system_coordinator, self.llm_manager)
            elif role == AgentRole.CODE_EXECUTOR:
                agent = LLMExecutorAgent(name, self.system_coordinator, self.llm_manager)
            else:
                # 默认使用基础LLM智能体
                agent = LLMEnabledAutonomousAgent(profile, self.system_coordinator, self.llm_manager)
            
            # 注册到系统
            self.system_coordinator.register_agent(agent)
            self.agents[name] = agent
            
            self.logger.info(f"Created LLM-enabled agent: {name} ({role.value})")
    
    async def run_experiment_tasks(self):
        """运行实验配置中定义的任务"""
        self.logger.info(f"Running tasks from experiment: {self.experiment_config.name}")
        
        for task_config in self.experiment_config.tasks:
            task_name = task_config.get("name", "unnamed_task")
            self.logger.info(f"Submitting task: {task_name}")
            
            # 提交任务
            task_id = await self.submit_task(
                description=task_name,
                task_type=task_config.get("task_type", "simple_logic"),
                requirements=task_config.get("requirements", {}),
                priority=task_config.get("priority", 0.5)
            )
            
            self.logger.info(f"Task {task_name} submitted with ID: {task_id}")
            
            # 如果配置了任务间隔，等待
            if "task_interval" in self.experiment_config.system:
                await asyncio.sleep(self.experiment_config.system["task_interval"])

# ============================================================================
# 使用示例
# ============================================================================

async def main():
    """主函数示例 - 使用LLM增强的框架"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用配置文件创建框架
    framework = LLMUnifiedAutonomousFramework("configs/experiment_basic.yaml")
    
    # 初始化
    await framework.initialize()
    
    # 启动
    await framework.start()
    
    # 运行实验任务
    await framework.run_experiment_tasks()
    
    # 运行一段时间
    run_duration = framework.experiment_config.system.get("experiment_duration", 60)
    await asyncio.sleep(run_duration)
    
    # 检查状态
    status = framework.get_system_status()
    print(f"\nExperiment Results:")
    print(f"- Experiment: {framework.experiment_config.name}")
    print(f"- LLM Model: {framework.llm_manager.config.model}")
    print(f"- Active agents: {len(status['agents'])}")
    print(f"- Tasks completed: {status['system_metrics']['completed_tasks']}")
    print(f"- Average task duration: {status['system_metrics'].get('average_task_duration', 0):.2f}s")
    
    # 停止
    await framework.stop()

if __name__ == "__main__":
    asyncio.run(main())