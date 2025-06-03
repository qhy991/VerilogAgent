from unified_autonomous_framework import *
import asyncio
import logging
from typing import Dict, Any, Optional, List

class CoderAgent(AutonomousAgent):
    """重构后的代码生成智能体"""
    
    def __init__(self, name: str = "CoderAgent", system_coordinator: SystemCoordinator = None):
        # 创建Coder专属的Profile
        profile = UnifiedAgentProfile(
            name=name,
            role=AgentRole.CODE_GENERATOR,
            communication_style=CommunicationStyle.CREATIVE,
            personality_traits={
                PersonalityTrait.CREATIVE: 0.9,
                PersonalityTrait.PROACTIVE: 0.8,
                PersonalityTrait.CURIOUS: 0.7,
                PersonalityTrait.HELPFUL: 0.8,
                PersonalityTrait.AMBITIOUS: 0.7
            },
            capabilities={
                AgentCapability.CODE_GENERATION: 0.95,
                AgentCapability.CODE_DEBUGGING: 0.8,
                AgentCapability.DOCUMENTATION: 0.7,
                AgentCapability.DESIGN_OPTIMIZATION: 0.6
            },
            proactivity=0.8,
            learning_rate=0.2
        )
        
        # 初始化父类
        super().__init__(profile, system_coordinator)
        
        # Coder特有的组件
        self.code_generator = VerilogCodeGenerator(self)
        self.design_patterns = self._load_design_patterns()
        self.generation_history = deque(maxlen=50)
        
        self.logger = logging.getLogger(f"CoderAgent.{name}")
    
    def _load_design_patterns(self) -> Dict[str, Any]:
        """加载设计模式库"""
        return {
            "counter": {
                "template": "always @(posedge clk) begin\n  if (reset) count <= 0;\n  else count <= count + 1;\nend",
                "parameters": ["width", "reset_value"]
            },
            "mux": {
                "template": "assign out = sel ? in1 : in0;",
                "parameters": ["width", "inputs"]
            },
            "fsm": {
                "template": "always @(posedge clk) state <= next_state;",
                "parameters": ["states", "transitions"]
            }
        }
    
    async def _perception_loop(self):
        """Coder的感知循环 - 重写父类方法"""
        while self.behavior_loop_active:
            try:
                # 感知可用的编码任务
                available_tasks = await self.system_coordinator.get_available_tasks()
                
                # 筛选适合的任务
                coding_tasks = [
                    task for task in available_tasks
                    if task.task_type in [TaskType.SIMPLE_LOGIC, TaskType.COMBINATIONAL, 
                                         TaskType.SEQUENTIAL, TaskType.COMPLEX_MODULE]
                ]
                
                # 更新决策上下文
                self.decision_context["available_coding_tasks"] = coding_tasks
                self.decision_context["design_opportunities"] = len(coding_tasks)
                
                # 感知协作机会
                if coding_tasks:
                    for task in coding_tasks[:3]:  # 检查前3个任务
                        if task.complexity > 0.7:
                            # 复杂任务可能需要协作
                            self.decision_context["collaboration_opportunities"].append({
                                "task": task,
                                "reason": "high_complexity",
                                "suggested_partners": ["ReviewerAgent", "ExecutorAgent"]
                            })
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in Coder perception loop: {e}")
                await asyncio.sleep(5)
    
    async def _task_execution_loop(self):
        """Coder的任务执行循环"""
        while self.behavior_loop_active:
            try:
                # 检查当前任务
                if self.active_tasks:
                    for task_id, task_info in list(self.active_tasks.items()):
                        if task_info["status"] == "accepted":
                            # 开始代码生成
                            await self._execute_coding_task(task_id, task_info["task"])
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in task execution loop: {e}")
                await asyncio.sleep(2)
    
    async def _execute_coding_task(self, task_id: str, task: UnifiedTask):
        """执行编码任务"""
        self.logger.info(f"Starting code generation for task {task_id}")
        
        # 更新任务状态
        self.active_tasks[task_id]["status"] = "in_progress"
        
        try:
            # 分析任务需求
            analysis = await self._analyze_requirements(task)
            
            # 选择设计方法
            approach = self._select_design_approach(task, analysis)
            
            # 生成代码
            generated_code = await self.code_generator.generate(
                task_type=task.task_type,
                requirements=task.requirements,
                approach=approach
            )
            
            # 自我审查
            if self.profile.personality_traits.get(PersonalityTrait.DETAIL_ORIENTED, 0.5) > 0.6:
                generated_code = await self._self_review_code(generated_code)
            
            # 记录到历史
            self.generation_history.append({
                "task_id": task_id,
                "approach": approach,
                "success": True,
                "timestamp": time.time()
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
                            "generation_time": time.time() - self.active_tasks[task_id]["started_at"]
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
                "duration": time.time() - self.active_tasks[task_id]["started_at"]
            })
            
        except Exception as e:
            self.logger.error(f"Failed to generate code for task {task_id}: {e}")
            self.active_tasks[task_id]["status"] = "failed"
            
            # 请求帮助
            if self.profile.personality_traits.get(PersonalityTrait.COLLABORATIVE, 0.5) > 0.6:
                await self._request_help_for_task(task_id, str(e))
    
    async def _analyze_requirements(self, task: UnifiedTask) -> Dict[str, Any]:
        """分析任务需求"""
        analysis = {
            "module_type": self._identify_module_type(task),
            "complexity_factors": [],
            "suggested_patterns": [],
            "potential_challenges": []
        }
        
        # 复杂度因素
        if task.requirements.get("state_count", 0) > 4:
            analysis["complexity_factors"].append("multiple_states")
        
        if task.requirements.get("interface_count", 0) > 5:
            analysis["complexity_factors"].append("complex_interface")
        
        # 建议的设计模式
        if task.task_type == TaskType.SEQUENTIAL:
            analysis["suggested_patterns"].extend(["fsm", "counter"])
        
        return analysis
    
    def _select_design_approach(self, task: UnifiedTask, analysis: Dict[str, Any]) -> str:
        """选择设计方法"""
        # 基于个性选择方法
        if self.profile.personality_traits.get(PersonalityTrait.CREATIVE, 0.5) > 0.7:
            # 创造性方法
            if "fsm" in analysis["suggested_patterns"]:
                return "innovative_fsm_design"
            return "creative_structural_design"
        else:
            # 传统方法
            if task.task_type == TaskType.SIMPLE_LOGIC:
                return "gate_level_design"
            return "rtl_design"
    
    async def _self_review_code(self, code: str) -> str:
        """自我审查代码"""
        # 简单的自我审查
        issues = []
        
        # 检查基本问题
        if "always @" in code and "begin" not in code:
            issues.append("missing_begin_end")
        
        if issues:
            self.logger.info(f"Self-review found issues: {issues}")
            # 尝试修复
            code = await self._fix_code_issues(code, issues)
        
        return code
    
    async def _request_help_for_task(self, task_id: str, error: str):
        """请求协作帮助"""
        help_request = UnifiedMessage(
            type="help_request",
            content={
                "task_id": task_id,
                "issue": error,
                "request_type": "code_generation_assistance",
                "current_progress": self.active_tasks[task_id].get("progress", 0)
            },
            sender=self.profile.name,
            receivers=["ReviewerAgent", "ExecutorAgent"],
            priority=MessagePriority.HIGH,
            intent=MessageIntent.HELP_REQUEST
        )
        
        await self.message_router.route_message(help_request, self.profile.name)
    
    # 重写消息处理以支持Coder特定的消息类型
    async def _generate_response(self, original_message: UnifiedMessage, 
                                response_decision: Dict[str, Any]) -> Optional[UnifiedMessage]:
        """生成Coder特定的响应"""
        response_type = response_decision["response_type"]
        
        if response_type == "code_improvement_suggestion":
            return await self._create_code_improvement_response(original_message)
        elif response_type == "design_discussion":
            return await self._create_design_discussion_response(original_message)
        else:
            # 使用父类的通用响应
            return await super()._generate_response(original_message, response_decision)
    
    def _identify_module_type(self, task: UnifiedTask) -> str:
        """识别模块类型"""
        desc = task.description.lower()
        
        if "counter" in desc:
            return "counter"
        elif "mux" in desc or "multiplex" in desc:
            return "multiplexer"
        elif "fsm" in desc or "state machine" in desc:
            return "state_machine"
        else:
            return "generic"


class VerilogCodeGenerator:
    """Verilog代码生成器"""
    
    def __init__(self, agent: CoderAgent):
        self.agent = agent
        self.logger = logging.getLogger(f"CodeGenerator.{agent.profile.name}")
    
    async def generate(self, task_type: TaskType, requirements: Dict[str, Any], 
                      approach: str) -> str:
        """生成Verilog代码"""
        # 构建提示
        prompt = self._build_generation_prompt(task_type, requirements, approach)
        
        # 调用LLM生成代码
        code = await self._call_llm_for_generation(prompt)
        
        # 后处理
        code = self._post_process_code(code, requirements)
        
        return code
    
    def _build_generation_prompt(self, task_type: TaskType, requirements: Dict[str, Any], 
                                approach: str) -> str:
        """构建生成提示"""
        prompt_parts = [
            f"Generate Verilog code using {approach} approach.",
            f"Task type: {task_type.value}",
            f"Requirements:",
        ]
        
        for key, value in requirements.items():
            prompt_parts.append(f"  - {key}: {value}")
        
        # 添加个性化指导
        if self.agent.profile.personality_traits.get(PersonalityTrait.CREATIVE, 0.5) > 0.7:
            prompt_parts.append("\nBe creative and innovative in your design approach.")
        
        if self.agent.profile.personality_traits.get(PersonalityTrait.DETAIL_ORIENTED, 0.5) > 0.7:
            prompt_parts.append("\nInclude detailed comments explaining the design.")
        
        return "\n".join(prompt_parts)
    
    async def _call_llm_for_generation(self, prompt: str) -> str:
        """调用LLM生成代码（模拟）"""
        await asyncio.sleep(2)  # 模拟LLM调用延迟
        
        # 这里应该是实际的LLM调用
        # 现在返回模拟的代码
        return """module generated_module(
    input wire clk,
    input wire reset,
    input wire enable,
    output reg [3:0] count
);

    // Generated by CoderAgent with creative approach
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count <= 4'b0000;
        end else if (enable) begin
            count <= count + 1'b1;
        end
    end

endmodule"""
    
    def _post_process_code(self, code: str, requirements: Dict[str, Any]) -> str:
        """后处理生成的代码"""
        # 添加文件头
        header = f"""// Generated by {self.agent.profile.name}
// Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
// Approach: {self.agent.profile.communication_style.value}

"""
        return header + code