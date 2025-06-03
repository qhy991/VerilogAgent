#!/usr/bin/env python3
"""
自动对话多智能体系统 - 快速演示脚本
这个脚本展示了如何使用系统完成简单的Verilog设计任务
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

# 简化版的框架，用于演示
class QuickDemo:
    """快速演示类"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.work_dir = Path(tempfile.mkdtemp(prefix="verilog_demo_"))
        self.demo_tasks = self._load_demo_tasks()
        
        print("🚀 Auto-Dialogue Multi-Agent System - Quick Demo")
        print("=" * 50)
        print(f"📁 Working directory: {self.work_dir}")
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("QuickDemo")
    
    def _load_demo_tasks(self):
        """加载演示任务"""
        return [
            {
                'name': '2-to-1 Multiplexer',
                'description': 'Design a 2-to-1 multiplexer with 4-bit inputs',
                'expected_solution': '''module mux2to1(
    input [3:0] a,
    input [3:0] b,
    input sel,
    output [3:0] out
);
    assign out = sel ? b : a;
endmodule''',
                'test_code': '''module mux_tb;
    reg [3:0] a, b;
    reg sel;
    wire [3:0] out;
    
    mux2to1 dut(.a(a), .b(b), .sel(sel), .out(out));
    
    initial begin
        $dumpfile("mux.vcd");
        $dumpvars(0, mux_tb);
        
        a = 4'b1010; b = 4'b0101; sel = 0;
        #10;
        if (out !== a) $display("ERROR: Expected %b, got %b", a, out);
        
        sel = 1;
        #10;
        if (out !== b) $display("ERROR: Expected %b, got %b", b, out);
        
        $display("Test completed");
        $finish;
    end
endmodule'''
            },
            {
                'name': '4-bit Counter',
                'description': 'Design a 4-bit counter with synchronous reset and enable',
                'expected_solution': '''module counter(
    input clk,
    input reset,
    input enable,
    output reg [3:0] count
);
    always @(posedge clk) begin
        if (reset)
            count <= 4'b0000;
        else if (enable)
            count <= count + 1;
    end
endmodule''',
                'test_code': '''module counter_tb;
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
        #50;
        if (count == 4'b0101) $display("PASS: Counter working correctly");
        else $display("FAIL: Expected count=5, got %d", count);
        
        $finish;
    end
endmodule'''
            },
            {
                'name': 'D Flip-Flop',
                'description': 'Design a D flip-flop with asynchronous reset',
                'expected_solution': '''module dff(
    input clk,
    input reset,
    input d,
    output reg q,
    output qn
);
    assign qn = ~q;
    
    always @(posedge clk or posedge reset) begin
        if (reset)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule''',
                'test_code': '''module dff_tb;
    reg clk, reset, d;
    wire q, qn;
    
    dff dut(.clk(clk), .reset(reset), .d(d), .q(q), .qn(qn));
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("dff.vcd");
        $dumpvars(0, dff_tb);
        
        reset = 1; d = 0;
        #15 reset = 0;
        #10 d = 1;
        #10;
        if (q == 1 && qn == 0) $display("PASS: DFF working correctly");
        else $display("FAIL: DFF not working");
        
        $finish;
    end
endmodule'''
            }
        ]
    
    async def run_demo(self):
        """运行完整演示"""
        print("\n🎯 Starting Auto-Dialogue Demo...")
        print("This demo simulates the multi-agent dialogue process")
        
        for i, task in enumerate(self.demo_tasks, 1):
            print(f"\n{'='*60}")
            print(f"📋 Demo Task {i}/{len(self.demo_tasks)}: {task['name']}")
            print(f"{'='*60}")
            
            success = await self._simulate_dialogue(task)
            
            if success:
                print(f"✅ Task {i} completed successfully!")
            else:
                print(f"❌ Task {i} failed!")
            
            # 等待一下，让用户看到输出
            await asyncio.sleep(2)
        
        print(f"\n🎉 Demo completed! Check results in: {self.work_dir}")
        self._show_summary()
    
    async def _simulate_dialogue(self, task):
        """模拟智能体对话过程"""
        print(f"\n📝 Problem: {task['description']}")
        
        max_rounds = 5
        current_code = ""
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n🔄 Round {round_num}/{max_rounds}")
            
            # 模拟Coder Agent
            print("🧠 Coder Agent: Generating code...")
            await asyncio.sleep(1)  # 模拟思考时间
            
            if round_num == 1:
                # 第一轮生成基础代码
                current_code = self._generate_initial_code(task)
                print("📝 Generated initial code structure")
            else:
                # 后续轮次改进代码
                current_code = self._improve_code(current_code, task)
                print("🔧 Improved code based on feedback")
            
            # 模拟Reviewer Agent
            print("🔍 Reviewer Agent: Analyzing code...")
            await asyncio.sleep(1)
            
            review_result = self._simulate_code_review(current_code, task)
            
            if review_result['approved']:
                print("✅ Code approved by reviewer")
            else:
                print(f"⚠️  Issues found: {', '.join(review_result['issues'])}")
                if round_num < max_rounds:
                    print("🔄 Sending back to coder for improvements...")
                    continue
            
            # 模拟Executor Agent
            print("⚡ Executor Agent: Testing code...")
            await asyncio.sleep(1)
            
            test_result = await self._simulate_execution(current_code, task)
            
            if test_result['success']:
                print("🎯 All tests passed!")
                self._save_successful_design(task, current_code)
                return True
            else:
                print(f"🔴 Test failed: {test_result['error']}")
                if round_num < max_rounds:
                    print("🔄 Sending feedback to coder...")
        
        print("❌ Maximum rounds reached without success")
        return False
    
    def _generate_initial_code(self, task):
        """生成初始代码（模拟LLM）"""
        # 这里可以集成真实的LLM调用
        # 现在使用预定义的解决方案作为演示
        return task['expected_solution']
    
    def _improve_code(self, current_code, task):
        """改进代码（模拟迭代优化）"""
        # 模拟代码改进过程
        improvements = [
            "// Added detailed comments",
            "// Improved signal naming",
            "// Enhanced error handling",
            "// Optimized logic structure"
        ]
        
        # 随机添加一个改进注释
        import random
        improvement = random.choice(improvements)
        return f"{improvement}\n{current_code}"
    
    def _simulate_code_review(self, code, task):
        """模拟代码审查"""
        issues = []
        
        # 简单的静态检查
        if 'module' not in code:
            issues.append("Missing module declaration")
        
        if 'endmodule' not in code:
            issues.append("Missing endmodule")
        
        if task['name'] == '4-bit Counter':
            if 'always @(posedge clk)' not in code:
                issues.append("Missing clock edge sensitivity")
            
            if 'reset' in code and 'if (reset)' not in code:
                issues.append("Reset logic not implemented")
        
        return {
            'approved': len(issues) == 0,
            'issues': issues,
            'score': max(0, 100 - len(issues) * 20)
        }
    
    async def _simulate_execution(self, code, task):
        """模拟代码执行和测试"""
        # 写入代码文件
        module_name = self._extract_module_name(code)
        code_file = self.work_dir / f"{module_name}.v"
        test_file = self.work_dir / f"{module_name}_tb.v"
        
        with open(code_file, 'w') as f:
            f.write(code)
        
        with open(test_file, 'w') as f:
            f.write(task['test_code'])
        
        # 尝试编译（如果有iverilog）
        try:
            import subprocess
            
            # 检查是否有iverilog
            result = subprocess.run(['which', 'iverilog'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # 有iverilog，进行真实编译
                compile_result = subprocess.run([
                    'iverilog', '-o', str(self.work_dir / f'{module_name}_sim'),
                    str(code_file), str(test_file)
                ], capture_output=True, text=True)
                
                if compile_result.returncode == 0:
                    print("✅ Compilation successful")
                    
                    # 运行仿真
                    sim_result = subprocess.run([
                        str(self.work_dir / f'{module_name}_sim')
                    ], capture_output=True, text=True, timeout=10)
                    
                    if sim_result.returncode == 0:
                        output = sim_result.stdout
                        if