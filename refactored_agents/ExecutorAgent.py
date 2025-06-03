from unified_autonomous_framework import *
import asyncio
import logging
from typing import Dict, Any, Optional, List
import subprocess
import tempfile
import os

class ExecutorAgent(AutonomousAgent):
    """重构后的代码执行智能体"""
    
    def __init__(self, name: str = "ExecutorAgent", system_coordinator: SystemCoordinator = None):
        # 创建Executor专属的Profile
        profile = UnifiedAgentProfile(
            name=name,
            role=AgentRole.CODE_EXECUTOR,
            communication_style=CommunicationStyle.DIRECTIVE,
            personality_traits={
                PersonalityTrait.FOCUSED: 0.9,
                PersonalityTrait.RELIABLE: 0.95,
                PersonalityTrait.EFFICIENT: 0.9,
                PersonalityTrait.DETAIL_ORIENTED: 0.8,
                PersonalityTrait.HELPFUL: 0.7
            },
            capabilities={
                AgentCapability.COMPILATION: 0.95,
                AgentCapability.SIMULATION: 0.9,
                AgentCapability.TEST_GENERATION: 0.7,
                AgentCapability.PERFORMANCE_ANALYSIS: 0.8,
                AgentCapability.ERROR_ANALYSIS: 0.85
            },
            proactivity=0.6,
            learning_rate=0.1
        )
        
        # 初始化父类
        super().__init__(profile, system_coordinator)
        
        # Executor特有的组件
        self.simulator = VerilogSimulator(self)
        self.test_generator = TestbenchGenerator(self)
        self.execution_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(list)
        
        # 执行环境设置
        self.work_dir = tempfile.mkdtemp(prefix="verilog_executor_")
        self.available_tools = self._detect_available_tools()
        
        self.logger = logging.getLogger(f"ExecutorAgent.{name}")
    
    def _detect_available_tools(self) -> Dict[str, bool]:
        """检测可用的工具"""
        tools = {
            "iverilog": self._check_tool("iverilog", "--version"),
            "verilator": self._check_tool("verilator", "--version"),
            "gtkwave": self._check_tool("gtkwave", "--version")
        }
        
        self.logger.info(f"Available tools: {[k for k, v in tools.items() if v]}")
        return tools
    
    def _check_tool(self, tool_name: str, version_arg: str) -> bool:
        """检查工具是否可用"""
        try:
            subprocess.run([tool_name, version_arg], 
                         capture_output=True, 
                         check=False, 
                         timeout=5)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    async def _perception_loop(self):
        """Executor的感知循环"""
        while self.behavior_loop_active:
            try:
                # 感知待执行的代码
                pending_executions = await self._scan_for_execution_requests()
                
                # 更新决策上下文
                self.decision_context["pending_executions"] = pending_executions
                self.decision_context["execution_queue_length"] = len(pending_executions)
                
                # 感知系统资源
                resource_status = self._check_system_resources()
                if resource_status["cpu_usage"] > 80:
                    self.decision_context["resource_constraints"].append({
                        "type": "high_cpu_usage",
                        "impact": "delayed_execution"
                    })
                
                # 感知测试需求
                test_opportunities = await self._identify_test_opportunities()
                self.decision_context["test_opportunities"] = test_opportunities
                
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error in Executor perception loop: {e}")
                await asyncio.sleep(3)
    
    async def _task_execution_loop(self):
        """Executor的任务执行循环"""
        while self.behavior_loop_active:
            try:
                # 处理待执行的代码
                if self.decision_context.get("pending_executions"):
                    execution = self.decision_context["pending_executions"].pop(0)
                    await self._execute_verilog_code(execution)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(2)
    
    async def _proactive_behavior_loop(self):
        """Executor的主动行为循环"""
        while self.behavior_loop_active:
            try:
                # 主动生成测试用例
                if self.profile.proactivity > 0.5:
                    if self._should_generate_tests():
                        await self._proactive_test_generation()
                
                # 主动进行性能分析
                if len(self.execution_history) > 10:
                    performance_insights = self._analyze_performance_trends()
                    if performance_insights:
                        await self._share_performance_insights(performance_insights)
                
                await asyncio.sleep(180)  # 3分钟检查一次
                
            except Exception as e:
                self.logger.error(f"Error in proactive behavior loop: {e}")
                await asyncio.sleep(180)
    
    async def _scan_for_execution_requests(self) -> List[Dict[str, Any]]:
        """扫描待执行的请求"""
        pending = []
        
        # 检查审查通过的代码
        for msg_info in self.decision_context.get("pending_messages", []):
            message = msg_info["message"]
            
            # 审查通过的代码
            if message.intent == MessageIntent.REVIEW_RESULT:
                if message.content.get("review_status") == "approved":
                    pending.append({
                        "task_id": message.content.get("task_id"),
                        "type": "simulation",
                        "priority": "high"
                    })
            
            # 直接执行请求
            elif message.intent == MessageIntent.TASK_RESULT:
                if message.content.get("ready_for_execution", False):
                    pending.append({
                        "task_id": message.content.get("task_id"),
                        "code": message.content.get("artifacts", {}).get("verilog_code"),
                        "type": "direct_execution",
                        "priority": "normal"
                    })
        
        return sorted(pending, key=lambda x: x["priority"] == "high", reverse=True)
    
    def _check_system_resources(self) -> Dict[str, float]:
        """检查系统资源"""
        try:
            # 简化的资源检查
            import psutil
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_free": psutil.disk_usage(self.work_dir).free / (1024**3)  # GB
            }
        except:
            # 如果psutil不可用，返回默认值
            return {
                "cpu_usage": 50.0,
                "memory_usage": 50.0,
                "disk_free": 10.0
            }
    
    async def _execute_verilog_code(self, execution_request: Dict[str, Any]):
        """执行Verilog代码"""
        task_id = execution_request.get("task_id")
        self.logger.info(f"Starting execution for task {task_id}")
        
        start_time = time.time()
        
        try:
            # 获取代码
            code = await self._retrieve_code_for_task(task_id)
            if not code:
                raise Exception("No code found for execution")
            
            # 生成测试台
            testbench = await self.test_generator.generate_testbench(code, task_id)
            
            # 执行仿真
            simulation_result = await self.simulator.simulate(code, testbench)
            
            # 分析结果
            analysis = self._analyze_simulation_result(simulation_result)
            
            # 记录执行历史
            self.execution_history.append({
                "task_id": task_id,
                "execution_time": time.time() - start_time,
                "success": simulation_result["success"],
                "warnings": len(simulation_result.get("warnings", [])),
                "timestamp": time.time()
            })
            
            # 准备执行结果消息
            result_message = UnifiedMessage(
                type="execution_result",
                content={
                    "task_id": task_id,
                    "status": "success" if simulation_result["success"] else "failed",
                    "simulation_output": simulation_result.get("output", ""),
                    "waveform_file": simulation_result.get("waveform_file"),
                    "analysis": analysis,
                    "execution_time": time.time() - start_time,
                    "executor": self.profile.name
                },
                sender=self.profile.name,
                receivers=["SystemCoordinator", "CoderAgent", "ReviewerAgent"],
                priority=MessagePriority.HIGH,
                intent=MessageIntent.EXECUTION_RESULT
            )
            
            await self.message_router.route_message(result_message, self.profile.name)
            
            # 学习记录
            self.experience_buffer.append({
                "task_type": "code_execution",
                "success": simulation_result["success"],
                "duration": time.time() - start_time,
                "tool_used": simulation_result.get("tool", "unknown")
            })
            
        except Exception as e:
            self.logger.error(f"Execution failed for task {task_id}: {e}")
            
            # 发送失败通知
            error_message = UnifiedMessage(
                type="execution_error",
                content={
                    "task_id": task_id,
                    "error": str(e),
                    "executor": self.profile.name
                },
                sender=self.profile.name,
                receivers=["SystemCoordinator", "CoderAgent"],
                priority=MessagePriority.HIGH,
                intent=MessageIntent.ERROR_REPORT
            )
            
            await self.message_router.route_message(error_message, self.profile.name)
    
    async def _retrieve_code_for_task(self, task_id: str) -> Optional[str]:
        """获取任务的代码"""
        # 从系统协调器或知识库获取代码
        # 这里简化处理，实际应该查询系统
        return """module counter(
    input clk,
    input reset,
    output [3:0] count
);
    reg [3:0] count;
    always @(posedge clk) begin
        if (reset) count <= 0;
        else count <= count + 1;
    end
endmodule"""
    
    def _analyze_simulation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """分析仿真结果"""
        analysis = {
            "functional_correctness": "unknown",
            "timing_analysis": {},
            "resource_usage": {},
            "recommendations": []
        }
        
        if result["success"]:
            analysis["functional_correctness"] = "verified"
            
            # 分析输出
            output = result.get("output", "")
            if "error" in output.lower():
                analysis["functional_correctness"] = "errors_detected"
                analysis["recommendations"].append("Review error conditions")
            
            # 提取时序信息（如果有）
            if "timing" in result:
                analysis["timing_analysis"] = result["timing"]
        
        return analysis
    
    async def _identify_test_opportunities(self) -> List[Dict[str, Any]]:
        """识别测试机会"""
        opportunities = []
        
        # 查找最近执行但测试覆盖不足的代码
        recent_executions = list(self.execution_history)[-10:]
        for execution in recent_executions:
            if execution.get("test_coverage", 0) < 80:
                opportunities.append({
                    "task_id": execution["task_id"],
                    "reason": "low_test_coverage",
                    "current_coverage": execution.get("test_coverage", 0)
                })
        
        return opportunities
    
    def _should_generate_tests(self) -> bool:
        """决定是否应该生成测试"""
        # 基于个性和测试覆盖率
        if self.profile.personality_traits.get(PersonalityTrait.DETAIL_ORIENTED, 0.5) > 0.7:
            # 检查最近的测试覆盖率
            if self.decision_context.get("test_opportunities"):
                return True
        
        return False
    
    async def _proactive_test_generation(self):
        """主动生成测试用例"""
        opportunities = self.decision_context.get("test_opportunities", [])
        
        if opportunities:
            # 选择一个任务生成测试
            target = opportunities[0]
            
            self.logger.info(f"Proactively generating tests for task {target['task_id']}")
            
            # 生成增强测试
            enhanced_tests = await self.test_generator.generate_enhanced_tests(
                target["task_id"],
                target.get("current_coverage", 0)
            )
            
            # 分享测试
            test_message = UnifiedMessage(
                type="enhanced_tests",
                content={
                    "task_id": target["task_id"],
                    "test_cases": enhanced_tests,
                    "expected_coverage": 95,
                    "executor": self.profile.name
                },
                sender=self.profile.name,
                receivers=["CoderAgent", "ReviewerAgent"],
                priority=MessagePriority.LOW,
                intent=MessageIntent.INFORMATION_SHARING
            )
            
            await self.message_router.route_message(test_message, self.profile.name)
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.execution_history) < 10:
            return {}
        
        recent = list(self.execution_history)[-20:]
        
        # 计算指标
        avg_execution_time = sum(e["execution_time"] for e in recent) / len(recent)
        success_rate = sum(1 for e in recent if e["success"]) / len(recent)
        
        insights = {}
        
        # 识别性能问题
        if avg_execution_time > 10:  # 秒
            insights["performance_issue"] = {
                "type": "slow_execution",
                "average_time": avg_execution_time,
                "recommendation": "Consider optimization or tool upgrade"
            }
        
        if success_rate < 0.8:
            insights["reliability_issue"] = {
                "type": "low_success_rate",
                "rate": success_rate,
                "recommendation": "Review common failure patterns"
            }
        
        return insights


class VerilogSimulator:
    """Verilog仿真器"""
    
    def __init__(self, agent: ExecutorAgent):
        self.agent = agent
        self.logger = logging.getLogger(f"Simulator.{agent.profile.name}")
    
    async def simulate(self, code: str, testbench: str) -> Dict[str, Any]:
        """执行仿真"""
        result = {
            "success": False,
            "output": "",
            "warnings": [],
            "errors": [],
            "tool": "none"
        }
        
        # 选择仿真工具
        if self.agent.available_tools.get("iverilog"):
            result = await self._simulate_with_iverilog(code, testbench)
        elif self.agent.available_tools.get("verilator"):
            result = await self._simulate_with_verilator(code, testbench)
        else:
            # 模拟仿真
            result = await self._simulate_mock(code, testbench)
        
        return result
    
    async def _simulate_mock(self, code: str, testbench: str) -> Dict[str, Any]:
        """模拟仿真（当没有实际工具时）"""
        await asyncio.sleep(1)  # 模拟执行时间
        
        return {
            "success": True,
            "output": "Simulation completed successfully\nAll tests passed",
            "warnings": [],
            "errors": [],
            "tool": "mock_simulator",
            "waveform_file": None
        }


class TestbenchGenerator:
    """测试台生成器"""
    
    def __init__(self, agent: ExecutorAgent):
        self.agent = agent
        self.logger = logging.getLogger(f"TestGenerator.{agent.profile.name}")
    
    async def generate_testbench(self, dut_code: str, task_id: str) -> str:
        """生成测试台"""
        # 分析DUT接口
        interface = self._analyze_interface(dut_code)
        
        # 生成测试台模板
        testbench = f"""module tb_{task_id};
    // Generated by {self.agent.profile.name}
    
    // DUT signals
{self._generate_signal_declarations(interface)}
    
    // Clock generation
    always #5 clk = ~clk;
    
    // DUT instantiation
    {interface['module_name']} dut(
{self._generate_port_connections(interface)}
    );
    
    // Test sequence
    initial begin
        $dumpfile("wave.vcd");
        $dumpvars(0, tb_{task_id});
        
        // Initialize
        clk = 0;
        reset = 1;
        
        // Test cases
        #20 reset = 0;
        
        // Add more test cases
        #100 $finish;
    end
    
endmodule"""
        
        return testbench
    
    def _analyze_interface(self, code: str) -> Dict[str, Any]:
        """分析模块接口"""
        # 简化的接口分析
        return {
            "module_name": "counter",
            "inputs": ["clk", "reset"],
            "outputs": ["count[3:0]"]
        }
    
    def _generate_signal_declarations(self, interface: Dict[str, Any]) -> str:
        """生成信号声明"""
        declarations = []
        
        for input_sig in interface.get("inputs", []):
            declarations.append(f"    reg {input_sig};")
        
        for output_sig in interface.get("outputs", []):
            declarations.append(f"    wire {output_sig};")
        
        return "\n".join(declarations)
    
    def _generate_port_connections(self, interface: Dict[str, Any]) -> str:
        """生成端口连接"""
        connections = []
        
        all_ports = interface.get("inputs", []) + interface.get("outputs", [])
        for i, port in enumerate(all_ports):
            # 提取端口名（去掉位宽）
            port_name = port.split('[')[0]
            connections.append(f"        .{port_name}({port_name})")
        
        return ",\n".join(connections)