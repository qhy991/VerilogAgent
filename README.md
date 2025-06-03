# 自动对话多智能体系统 - 完整使用指南

## 🎯 系统概述

这是一个完全自动化的多智能体对话系统，专门用于Verilog设计任务。系统包含三类智能体：

- **🧠 Coder Agent**: 负责代码生成和优化
- **🔍 Reviewer Agent**: 负责代码审查和质量控制
- **⚡ Executor Agent**: 负责编译、仿真和测试

### 核心特性

✅ **完整的Verilog工具链集成** - Icarus Verilog, Verilator, Yosys支持  
✅ **智能对话管理** - 自动协调智能体间的协作  
✅ **持久化存储** - 任务历史、学习结果、智能体状态保存  
✅ **实时Web监控** - 可视化界面监控系统状态  
✅ **高级学习机制** - 强化学习、模式识别、知识迁移  
✅ **自动结束条件** - 任务完成或达到最大轮次自动停止  

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目文件
# 运行安装脚本
chmod +x setup.sh
./setup.sh

# 或手动安装依赖
pip install -r requirements.txt

# 安装Verilog工具（Ubuntu/Debian）
sudo apt-get install iverilog verilator gtkwave yosys

# 安装Verilog工具（macOS）
brew install icarus-verilog verilator gtkwave yosys
```

### 2. 配置设置

编辑配置文件 `configs/basic_auto_dialogue.yaml`:

```yaml
llm:
  provider: "openai"
  api_key: "your-openai-api-key-here"  # 必须设置
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

dialogue:
  max_rounds: 10        # 最大对话轮次
  timeout_minutes: 30   # 对话超时时间
  auto_escalation: true # 自动干预机制

agents:
  coder:
    personality:
      creative: 0.8      # 创造性
      proactive: 0.7     # 主动性
      focused: 0.6       # 专注度
  reviewer:
    personality:
      analytical: 0.9    # 分析性
      detail_oriented: 0.9  # 细节导向
      cautious: 0.8      # 谨慎性
  executor:
    personality:
      focused: 0.9       # 专注度
      reliable: 0.9      # 可靠性
      proactive: 0.6     # 主动性
```

### 3. 启动系统

```bash
# 交互模式 - 命令行界面提交任务
python run_auto_dialogue.py interactive

# 演示模式 - 运行预定义示例任务
python run_auto_dialogue.py demo

# 服务器模式 - 后台运行，使用Web界面
python run_auto_dialogue.py server
```

### 4. Web监控界面

启动后访问: http://localhost:5000

界面功能：
- 📊 实时系统状态监控
- 🤖 智能体状态和性能指标
- 💬 活动对话进度追踪
- ➕ 在线任务提交
- 🔴 实时事件日志

## 📝 使用示例

### 示例1：简单逻辑设计

```python
# 在交互模式中提交任务
> task

Problem description: Design a 4-bit counter with synchronous reset and enable

Test code (optional, press Enter twice to finish):
module counter_tb;
    reg clk, reset, enable;
    wire [3:0] count;
    
    counter dut(.clk(clk), .reset(reset), .enable(enable), .count(count));
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("counter.vcd");
        $dumpvars(0, counter_tb);
        
        reset = 1; enable = 0;
        #20 reset = 0;
        #10 enable = 1;
        #100 $finish;
    end
endmodule


Custom prompt (optional): Focus on clean, readable code with proper reset behavior

🚀 Starting dialogue...
✅ Dialogue dialogue_1234567890_0 completed successfully!
```

### 示例2：复杂模块设计

```python
# 通过API提交任务
import asyncio
from auto_dialogue_framework import AutoDialogueFramework

async def submit_complex_task():
    framework = AutoDialogueFramework("configs/advanced_auto_dialogue.yaml")
    await framework.initialize()
    await framework.start()
    
    dialogue_id = await framework.start_auto_dialogue(
        problem_description="Design a UART transmitter with configurable baud rate",
        test_code='''
        module uart_tx_tb;
            reg clk, reset, tx_start;
            reg [7:0] tx_data;
            wire tx_out, tx_done;
            
            uart_tx dut(
                .clk(clk), .reset(reset), 
                .tx_start(tx_start), .tx_data(tx_data),
                .tx_out(tx_out), .tx_done(tx_done)
            );
            
            initial begin
                clk = 0;
                forever #5 clk = ~clk;
            end
            
            initial begin
                reset = 1; tx_start = 0;
                #20 reset = 0;
                #10 tx_data = 8'h55; tx_start = 1;
                #10 tx_start = 0;
                wait(tx_done);
                #100 $finish;
            end
        endmodule
        ''',
        custom_prompt="Include start bit, 8 data bits, and stop bit. Use 9600 baud rate."
    )
    
    print(f"Task completed: {dialogue_id}")

# 运行
asyncio.run(submit_complex_task())
```

## 🔄 对话流程

### 典型的对话轮次

1. **任务初始化**
   - 系统解析问题描述和测试代码
   - 向所有智能体广播任务信息

2. **代码生成阶段** (Coder Agent)
   - 分析需求和约束条件
   - 调用LLM生成Verilog代码
   - 应用个性化的设计风格

3. **代码审查阶段** (Reviewer Agent)
   - 检查语法和功能正确性
   - 评估代码质量和最佳实践
   - 提供改进建议

4. **执行测试阶段** (Executor Agent)
   - 使用Verilog工具链编译代码
   - 运行仿真测试
   - 生成测试报告和波形文件

5. **迭代改进**
   - 如果测试失败，回到步骤2
   - 智能体学习和调整策略
   - 最多进行max_rounds轮迭代

### 自动结束条件

✅ **成功完成**：
- 代码编译无错误
- 仿真测试通过
- 满足所有需求规范

❌ **达到最大轮次**：
- 超过配置的max_rounds
- 系统自动记录失败原因
- 提供改进建议

⚠️ **干预机制**：
- 连续3轮失败自动触发
- 调整智能体策略
- 简化问题复杂度

## 🎛️ 高级配置

### 智能体个性调整

```yaml
agents:
  coder:
    personality:
      creative: 0.9      # 提高创造性 -> 更创新的设计
      cautious: 0.3      # 降低谨慎性 -> 更大胆的尝试
      detail_oriented: 0.7  # 适中的细节关注
    capabilities:
      - code_generation
      - debugging
      - optimization
    
  reviewer:
    personality:
      analytical: 0.95   # 极高分析性 -> 深度代码分析
      critical: 0.8      # 高批判性 -> 严格审查
      helpful: 0.9       # 高帮助性 -> 建设性反馈
    specialization: "timing_analysis"  # 专业领域
```

### 学习系统配置

```yaml
learning:
  enable_reinforcement_learning: true
  pattern_recognition: true
  knowledge_transfer: true
  collective_intelligence: true
  learning_rate: 0.1
  
  # 强化学习参数
  reinforcement:
    discount_factor: 0.9
    exploration_rate: 0.1
    
  # 模式识别
  pattern_recognition:
    success_pattern_threshold: 0.8
    failure_pattern_threshold: 0.7
    
  # 知识迁移
  knowledge_transfer:
    similarity_threshold: 0.6
    transfer_confidence: 0.7
```

### 工具链配置

```yaml
toolchain:
  work_dir: "./verilog_workspace"
  cleanup_after_task: true
  simulation_timeout: 30
  
  # 编译器偏好
  preferred_compiler: "icarus"  # icarus, verilator
  
  # 仿真设置
  simulation:
    default_timeout: 1000
    waveform_format: "vcd"
    enable_coverage: true
    
  # 综合设置
  synthesis:
    target_device: "ice40"
    optimization_level: 2
```

## 📊 监控和调试

### 实时指标

系统提供以下监控指标：

- **任务成功率**：完成任务的百分比
- **平均轮次数**：任务完成所需的平均对话轮次
- **智能体工作负载**：各智能体的当前任务负载
- **错误分布**：不同类型错误的统计
- **学习进展**：智能体能力的提升轨迹

### 日志分析

```bash
# 查看系统日志
tail -f logs/auto_dialogue.log

# 查看特定智能体日志
tail -f logs/coder_agent.log

# 查看对话历史
tail -f logs/dialogue_1234567890.log
```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单步调试模式
framework = AutoDialogueFramework(
    config_path="configs/debug_config.yaml",
    max_dialogue_rounds=3  # 限制轮次便于调试
)
```

## 🔧 故障排除

### 常见问题

**Q1: LLM API调用失败**
```
Error: Failed to get response from LLM after all retries
```
- 检查API密钥是否正确
- 确认网络连接正常
- 验证API配额是否充足

**Q2: Verilog工具未找到**
```
Error: No Verilog compiler available
```
- 安装Icarus Verilog: `sudo apt-get install iverilog`
- 检查PATH环境变量
- 运行工具检测: `iverilog --version`

**Q3: 对话无限循环**
```
Warning: Task not completing after multiple rounds
```
- 降低任务复杂度
- 增加max_rounds限制
- 检查测试代码是否过于严格

**Q4: 内存使用过高**
```
Warning: High memory usage detected
```
- 减少并发对话数量
- 启用cleanup_after_task
- 调整工作目录大小限制

### 性能优化

**提高成功率**：
- 提供清晰详细的问题描述
- 包含完整的测试代码
- 使用分步骤的需求描述

**减少对话轮次**：
- 提高Coder Agent的细节关注度
- 降低Reviewer Agent的批判性
- 提供参考实现示例

**加速执行**：
- 使用更快的LLM模型（如GPT-3.5）
- 减少仿真时间限制
- 启用并行测试

## 📈 扩展和定制

### 添加新智能体角色

```python
class OptimizationAgent(AutonomousAgent):
    """优化专家智能体"""
    
    def __init__(self, name: str, system_coordinator: SystemCoordinator):
        profile = UnifiedAgentProfile(
            name=name,
            role=AgentRole.OPTIMIZATION_EXPERT,  # 新角色
            personality_traits={
                PersonalityTrait.ANALYTICAL: 0.9,
                PersonalityTrait.EFFICIENCY_FOCUSED: 0.95,
                PersonalityTrait.DETAIL_ORIENTED: 0.8
            },
            capabilities={
                AgentCapability.CODE_OPTIMIZATION: 0.95,
                AgentCapability.PERFORMANCE_ANALYSIS: 0.9,
                AgentCapability.RESOURCE_ANALYSIS: 0.85
            }
        )
        super().__init__(profile, system_coordinator)
```

### 自定义任务类型

```python
# 在TaskType枚举中添加新类型
class TaskType(Enum):
    SIMPLE_LOGIC = "simple_logic"
    COMBINATIONAL = "combinational"
    SEQUENTIAL = "sequential"
    COMPLEX_MODULE = "complex_module"
    TESTBENCH = "testbench"
    OPTIMIZATION = "optimization"
    # 新增类型
    PROTOCOL_IMPLEMENTATION = "protocol_implementation"
    SYSTEM_INTEGRATION = "system_integration"
    VERIFICATION = "verification"
```

### 集成外部工具

```python
class CustomToolchain(VerilogToolchain):
    """自定义工具链"""
    
    async def run_custom_analysis(self, code: str) -> ToolResult:
        """运行自定义分析工具"""
        # 集成静态分析工具
        # 集成形式化验证工具
        # 集成时序分析工具
        pass
```

## 📚 API参考

### 核心API

```python
# 创建框架实例
framework = AutoDialogueFramework(
    config_path="config.yaml",
    max_dialogue_rounds=10
)

# 初始化和启动
await framework.initialize()
await framework.start()

# 提交任务
dialogue_id = await framework.start_auto_dialogue(
    problem_description="设计一个...",
    test_code="module test...",
    custom_prompt="要求..."
)

# 获取系统状态
status = framework.get_system_status()

# 关闭系统
await framework.shutdown()
```

### Web API端点

```http
# 获取系统状态
GET /api/system/status

# 获取智能体信息
GET /api/agents

# 获取任务列表
GET /api/tasks

# 提交新任务
POST /api/submit_task
Content-Type: application/json
{
    "description": "设计描述",
    "task_type": "sequential",
    "test_code": "测试代码",
    "priority": 0.8
}

# 获取指标数据
GET /api/metrics
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 克隆开发分支
git clone -b develop project-repo
cd project-repo

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码风格检查
flake8 src/
black src/
```

### 添加新功能

1. **创建功能分支**：`git checkout -b feature/new-feature`
2. **实现功能**：遵循现有代码风格
3. **编写测试**：确保测试覆盖率 > 80%
4. **更新文档**：包括API文档和使用示例
5. **提交PR**：包含详细的功能描述

### 报告问题

请在GitHub Issues中报告问题，包含：
- 详细的错误描述
- 重现步骤
- 系统环境信息
- 相关日志文件

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下开源项目和工具：
- [Icarus Verilog](http://iverilog.icarus.com/)
- [Verilator](https://verilator.org/)
- [Yosys](https://yosyshq.net/yosys/)
- [OpenAI API](https://openai.com/api/)
- [Flask](https://flask.palletsprojects.com/)

---

🎉 **开始你的自动化Verilog设计之旅吧！**